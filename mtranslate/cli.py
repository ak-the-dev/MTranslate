"""CLI entrypoint for MTranslate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from .constants import ALLOWED_EXPORT_FORMATS
from .model_manager import (
    DEFAULT_INPAINT_REPO,
    DEFAULT_TRANSLATE_REPO,
    DEFAULT_TRANSLATE_VLLM_REPO,
    pull_inpaint_model,
    pull_profile,
    pull_translate_model,
)
from .ocr_backends import _select_model_files
from .paths import models_dir
from .pipeline import PipelineRunner, audit_job, load_manifest, summarize_manifest


def _repo_root() -> str:
    return os.getcwd()


def _parse_exports(value: str) -> list[str]:
    exports = [x.strip().lower() for x in value.split(",") if x.strip()]
    if not exports:
        raise ValueError("At least one export format is required")
    unsupported = sorted({x for x in exports if x not in ALLOWED_EXPORT_FORMATS})
    if unsupported:
        allowed = ",".join(sorted(ALLOWED_EXPORT_FORMATS))
        raise ValueError(f"Unsupported export format(s): {', '.join(unsupported)}. Allowed: {allowed}")
    return exports


def _default_translate_model_ref() -> str:
    override = os.getenv("MTRANSLATE_VLLM_MODEL", "").strip()
    if override:
        return override

    root = models_dir()
    candidates = [
        root / "mlx_community_gemma_3_4b_it_qat_4bit",
        root / "translator_llm_mlx",
        root / "translator_llm_vllm",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return DEFAULT_TRANSLATE_VLLM_REPO


def cmd_models_pull(args: argparse.Namespace) -> int:
    registry = pull_profile(args.profile)
    print(f"Profile: {registry.profile}")
    for model in registry.models:
        print(
            f"- {model['id']} ({model['kind']}, {model['runtime']}, {model['size_gb']}GB): "
            f"{model['state']} -> {model['path']}"
        )
    missing = [m for m in registry.models if m["state"] != "ready"]
    if missing:
        print("\nMissing model artifacts (place local files in the paths above):")
        for model in missing:
            print(f"  - {model['id']}: {model['notes']}")
    return 0


def _verify_inpaint_dir(root: Path) -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True

    if not root.exists():
        return False, [f"Inpaint model path does not exist: {root}"]

    model_index = root / "model_index.json"
    if not model_index.exists():
        ok = False
        messages.append(f"Missing file: {model_index}")

    required_subdirs = ["unet", "vae", "scheduler"]
    for name in required_subdirs:
        d = root / name
        if not d.exists() or not d.is_dir():
            ok = False
            messages.append(f"Missing subdirectory: {d}")

    if ok:
        messages.append(f"Inpaint model looks valid: {root}")
    return ok, messages


def cmd_models_pull_inpaint(args: argparse.Namespace) -> int:
    path = pull_inpaint_model(repo_id=args.repo, dest=args.path)
    print(f"Downloaded inpaint model: {path}")
    print(f"Set: export MTRANSLATE_INPAINT_MODEL='{path}'")
    ok, msgs = _verify_inpaint_dir(path)
    for m in msgs:
        print(m)
    return 0 if ok else 1


def cmd_models_verify_inpaint(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser().resolve()
    ok, messages = _verify_inpaint_dir(root)
    for m in messages:
        print(m)
    print("Result: OK" if ok else "Result: FAILED")
    return 0 if ok else 1


def _verify_translate_dir(root: Path) -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True

    if not root.exists():
        return False, [f"Translation model path does not exist: {root}"]

    config_path = root / "config.json"
    if not config_path.exists():
        ok = False
        messages.append(f"Missing file: {config_path}")
    elif _is_lfs_pointer(config_path):
        ok = False
        messages.append(f"LFS pointer detected in file: {config_path}")

    tokenizer_candidates = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
    if not any((root / name).exists() for name in tokenizer_candidates):
        ok = False
        messages.append(f"Missing tokenizer files in {root}")
    for tok_name in tokenizer_candidates:
        tok_path = root / tok_name
        if tok_path.exists() and _is_lfs_pointer(tok_path):
            ok = False
            messages.append(f"LFS pointer detected in tokenizer file: {tok_path}")

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        if _is_lfs_pointer(index_path):
            return False, [f"LFS pointer detected in index file: {index_path}"]
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return False, [f"Failed to parse index file {index_path}: {exc}"]

        weight_map = index.get("weight_map") or {}
        if not isinstance(weight_map, dict) or not weight_map:
            ok = False
            messages.append(f"Invalid or empty weight_map in {index_path}")
        else:
            shard_names = sorted(set(str(name) for name in weight_map.values()))
            missing_shards: list[Path] = []
            for name in shard_names:
                shard_path = root / name
                if not shard_path.exists():
                    missing_shards.append(shard_path)
                elif _is_lfs_pointer(shard_path):
                    ok = False
                    messages.append(f"LFS pointer detected in shard file: {shard_path}")
            if missing_shards:
                merged = root / "model.safetensors"
                if merged.exists() and not _is_lfs_pointer(merged):
                    messages.append(
                        "Index references sharded weights, but consolidated model.safetensors exists; "
                        "accepting MLX-converted layout."
                    )
                else:
                    ok = False
                    for shard_path in missing_shards:
                        messages.append(f"Missing shard file: {shard_path}")
    else:
        safetensors = list(root.glob("*.safetensors"))
        if not safetensors and not list(root.glob("pytorch_model*.bin")):
            ok = False
            messages.append(
                "No model weights found (expected model.safetensors.index.json, *.safetensors, or pytorch_model*.bin)"
            )
        for path in safetensors:
            if _is_lfs_pointer(path):
                ok = False
                messages.append(f"LFS pointer detected in weights file: {path}")

    if ok:
        messages.append(f"Translation model directory looks valid: {root}")
    return ok, messages


def cmd_models_pull_translate(args: argparse.Namespace) -> int:
    try:
        path = pull_translate_model(repo_id=args.repo, dest=args.path)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        lowered = msg.lower()
        if "gated repo" in lowered or "access to model" in lowered or "401" in lowered:
            print(
                "Translation model access is gated. Authenticate with Hugging Face first "
                "(set HF_TOKEN or run `huggingface-cli login`) and retry."
            )
            return 1
        raise
    print(f"Downloaded translation model: {path}")
    print(f"Set: export MTRANSLATE_VLLM_MODEL='{path}'")
    ok, msgs = _verify_translate_dir(path)
    for m in msgs:
        print(m)
    return 0 if ok else 1


def cmd_models_verify_translate(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser().resolve()
    ok, messages = _verify_translate_dir(root)
    for line in messages:
        print(line)
    print("Result: OK" if ok else "Result: FAILED")
    return 0 if ok else 1


def _verify_ocr_root(root: Path) -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True

    if not root.exists():
        return False, [f"OCR root does not exist: {root}"]

    required = ["det", "cls", "rec"]
    subdirs = {name: root / name for name in required}

    for name, path in subdirs.items():
        if not path.exists():
            ok = False
            messages.append(f"missing subdirectory: {name} ({path})")

    for name, path in subdirs.items():
        if not path.exists():
            continue
        try:
            model, params, fmt = _select_model_files(path)
            messages.append(f"{name}: OK [{fmt}] model={model} params={params}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            messages.append(f"{name}: ERROR {exc}")

    return ok, messages


def cmd_models_verify_ocr(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    ok, messages = _verify_ocr_root(root)
    for line in messages:
        print(line)
    print("Result: OK" if ok else "Result: FAILED")
    return 0 if ok else 1


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.read(256)
    except OSError:
        return False
    return "git-lfs.github.com/spec/v1" in head


def _verify_qwen_dir(root: Path) -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True

    if not root.exists():
        return False, [f"Qwen path does not exist: {root}"]

    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        return False, [f"Missing index file: {index_path}"]

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, [f"Failed to parse {index_path}: {exc}"]

    weight_map = index.get("weight_map") or {}
    if not isinstance(weight_map, dict) or not weight_map:
        return False, [f"Invalid or empty weight_map in {index_path}"]

    shard_names = sorted(set(str(name) for name in weight_map.values()))
    for name in shard_names:
        shard_path = root / name
        if not shard_path.exists():
            ok = False
            messages.append(f"Missing shard file: {shard_path}")
            continue
        if _is_lfs_pointer(shard_path):
            ok = False
            messages.append(
                f"LFS pointer detected in shard {shard_path}. "
                "Run: git lfs install && git lfs pull"
            )

    # Basic tokenizer sanity check
    for tok_name in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]:
        tok_path = root / tok_name
        if tok_path.exists() and _is_lfs_pointer(tok_path):
            ok = False
            messages.append(
                f"LFS pointer detected in tokenizer file {tok_path}. "
                "Run: git lfs install && git lfs pull"
            )

    if ok:
        messages.append("Qwen-VL weights and tokenizer look OK")
    return ok, messages


def cmd_models_verify_qwen(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser().resolve()
    ok, messages = _verify_qwen_dir(root)
    for line in messages:
        print(line)
    if ok:
        print("Result: OK")
        return 0
    print("Result: FAILED")
    return 1


def cmd_models_verify_vllm(args: argparse.Namespace) -> int:
    messages: list[str] = []
    ok = True

    plugin = args.plugin.strip()
    if plugin:
        os.environ.setdefault("VLLM_PLUGINS", plugin)
        messages.append(f"VLLM_PLUGINS={os.getenv('VLLM_PLUGINS', '')}")

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    messages.append(f"Python: {py_ver}")

    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec is None:
        ok = False
        messages.append("Missing dependency: vllm")
    else:
        messages.append("vllm import path: OK")

    if not ok:
        for line in messages:
            print(line)
        print("Result: FAILED")
        return 1

    try:
        import vllm  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"vLLM import failed: {exc}")
        print("Result: FAILED")
        return 1

    messages.append(f"vLLM version: {getattr(vllm, '__version__', 'unknown')}")
    if args.smoke:
        model = args.model
        messages.append(f"Running smoke test with model: {model}")
        smoke_code = (
            "import sys\n"
            "from mtranslate.translator_backends import VLLMTranslatorBackend\n"
            "model = sys.argv[1]\n"
            "backend = VLLMTranslatorBackend(glossary={}, model_path=model)\n"
            "text = backend.translate(\n"
            "    '\\u304a\\u306f\\u3088\\u3046\\u3054\\u3056\\u3044\\u307e\\u3059\\u3002',\n"
            "    {'prev': [], 'next': [], 'history': [], 'role': 'dialogue', 'orientation': 'horizontal'},\n"
            ")\n"
            "if not text:\n"
            "    raise RuntimeError('smoke output was empty')\n"
            "print(f'SMOKE_OUTPUT::{text}')\n"
        )
        try:
            smoke_env = dict(os.environ)
            if plugin:
                smoke_env["VLLM_PLUGINS"] = plugin
            smoke_env["MTRANSLATE_VLLM_DTYPE"] = args.dtype
            smoke_env["MTRANSLATE_VLLM_GPU_MEMORY_UTIL"] = str(args.gpu_memory_util)
            smoke_env["MTRANSLATE_VLLM_MAX_MODEL_LEN"] = str(args.max_model_len)
            cp = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    smoke_code,
                    model,
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=args.timeout_sec,
                env=smoke_env,
            )
        except subprocess.TimeoutExpired:
            print(f"Smoke test timed out after {args.timeout_sec}s")
            print("Result: FAILED")
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"Smoke test failed to start: {exc}")
            print("Result: FAILED")
            return 1

        if cp.returncode != 0:
            detail = (cp.stderr or cp.stdout or "").strip()
            if len(detail) > 1200:
                detail = detail[:1200] + "... <truncated>"
            print(f"Smoke test failed: {detail or f'exit code {cp.returncode}'}")
            print("Result: FAILED")
            return 1

        lines = [ln.strip() for ln in (cp.stdout or "").splitlines() if ln.strip()]
        smoke_lines = [ln for ln in lines if ln.startswith("SMOKE_OUTPUT::")]
        smoke_text = smoke_lines[-1].split("SMOKE_OUTPUT::", 1)[1].strip() if smoke_lines else ""
        if not smoke_text:
            detail = (cp.stderr or cp.stdout or "").strip()
            if len(detail) > 1200:
                detail = detail[:1200] + "... <truncated>"
            print(f"Smoke test failed: no smoke output marker found. {detail}")
            print("Result: FAILED")
            return 1
        messages.append(f"Smoke output: {smoke_text}")

    for line in messages:
        print(line)
    print("Result: OK")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    exports = _parse_exports(args.export)
    runner = PipelineRunner(
        input_path=args.input,
        output_path=args.output,
        export_formats=exports,
        glossary_path=args.glossary,
        pre_dict_path=args.pre_dict,
        post_dict_path=args.post_dict,
        max_workers=args.max_workers,
        font_path=args.font_path,
        repo_root=_repo_root(),
    )
    manifest = runner.run()
    summary = summarize_manifest(manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"manifest: {runner.manifest_file}")
    return 0 if manifest.status != "failed" else 2


def cmd_status(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.job)
    summary = summarize_manifest(manifest)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Job: {summary['job_id']}")
        print(f"Status: {summary['status']}")
        print(f"Input: {summary['input_path']}")
        print(f"Output: {summary['output_path']}")
        print(f"Export: {','.join(summary['export'])}")
        pages = summary["pages"]
        print(
            "Pages: total={total} done={done} failed={failed} running={running} pending={pending}".format(
                **pages
            )
        )
        if summary["notes"].get("pdf_output"):
            print(f"PDF: {summary['notes']['pdf_output']}")
        if summary["notes"].get("pdf_error"):
            print(f"PDF error: {summary['notes']['pdf_error']}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    report = audit_job(args.job)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("checks", {}).get("ok") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mtranslate", description="Fully automated offline manga translation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser("models", help="Model management")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    pull = models_sub.add_parser("pull", help="Prepare model profile")
    pull.add_argument("--profile", default="hq", choices=["hq"])
    pull.set_defaults(func=cmd_models_pull)

    pull_inpaint = models_sub.add_parser("pull-inpaint", help="Download an inpaint model from Hugging Face")
    pull_inpaint.add_argument("--repo", default=DEFAULT_INPAINT_REPO, help="Hugging Face repo id")
    pull_inpaint.add_argument("--path", default=None, help="Local destination directory")
    pull_inpaint.set_defaults(func=cmd_models_pull_inpaint)

    pull_translate = models_sub.add_parser("pull-translate", help="Download the translation LLM from Hugging Face")
    pull_translate.add_argument("--repo", default=DEFAULT_TRANSLATE_REPO, help="Hugging Face repo id")
    pull_translate.add_argument("--path", default=None, help="Local destination directory")
    pull_translate.set_defaults(func=cmd_models_pull_translate)

    verify_ocr = models_sub.add_parser("verify-ocr", help="Verify OCR FastDeploy model trio")
    verify_ocr.add_argument("--root", required=True, help="Root containing det/cls/rec")
    verify_ocr.set_defaults(func=cmd_models_verify_ocr)

    verify_inpaint = models_sub.add_parser("verify-inpaint", help="Verify local diffusion inpaint model directory")
    verify_inpaint.add_argument("--path", required=True, help="Path to local inpaint model directory")
    verify_inpaint.set_defaults(func=cmd_models_verify_inpaint)

    verify_translate = models_sub.add_parser("verify-translate", help="Verify local translation model directory")
    verify_translate.add_argument("--path", required=True, help="Path to local translation model directory")
    verify_translate.set_defaults(func=cmd_models_verify_translate)

    verify_qwen = models_sub.add_parser("verify-qwen", help="Verify Qwen-VL model directory")
    verify_qwen.add_argument("--path", required=True, help="Path to Qwen-VL repository")
    verify_qwen.set_defaults(func=cmd_models_verify_qwen)

    verify_vllm = models_sub.add_parser("verify-vllm", help="Verify vLLM installation")
    verify_vllm.add_argument(
        "--model",
        default=_default_translate_model_ref(),
        help="Model path or Hugging Face model id for smoke test",
    )
    verify_vllm.add_argument("--smoke", action="store_true", help="Run one generation smoke test")
    verify_vllm.add_argument("--dtype", default="auto", help="vLLM dtype for smoke test")
    verify_vllm.add_argument("--max-model-len", type=int, default=1024, help="Max model len for smoke test")
    verify_vllm.add_argument("--gpu-memory-util", type=float, default=0.85, help="GPU memory utilization")
    verify_vllm.add_argument("--timeout-sec", type=int, default=300, help="Smoke test timeout in seconds")
    verify_vllm.add_argument(
        "--plugin",
        default="",
        help="Optional vLLM platform plugin to activate (sets VLLM_PLUGINS)",
    )
    verify_vllm.set_defaults(func=cmd_models_verify_vllm)

    run = sub.add_parser("run", help="Run translation job")
    run.add_argument("--input", required=True, help="Input image folder")
    run.add_argument("--output", required=True, help="Output folder")
    run.add_argument("--glossary", default=None, help="Glossary yaml path")
    run.add_argument("--pre-dict", default=None, help="Regex replacement dictionary applied before translation")
    run.add_argument("--post-dict", default=None, help="Regex replacement dictionary applied after translation")
    run.add_argument("--export", default="folder,pdf", help="Comma-separated: folder,pdf")
    run.add_argument("--max-workers", type=int, default=None)
    run.add_argument("--font-path", default=None)
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="Job status")
    status.add_argument("--job", required=True, help="Job id")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    audit = sub.add_parser("audit", help="Audit a completed or in-progress job for manifest/output integrity")
    audit.add_argument("--job", required=True, help="Job id")
    audit.set_defaults(func=cmd_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        if os.getenv("MTRANSLATE_DEBUG"):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
