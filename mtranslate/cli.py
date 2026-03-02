"""CLI entrypoint for MTranslate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from .constants import STAGES
from .model_manager import (
    DEFAULT_INPAINT_REPO,
    DEFAULT_TRANSLATE_REPO,
    DEFAULT_TRANSLATE_VLLM_REPO,
    pull_inpaint_model,
    pull_profile,
    pull_translate_model,
)
from .ocr_backends import _select_model_files
from .pipeline import PipelineRunner, load_manifest, retry_job, summarize_manifest


def _repo_root() -> str:
    return os.getcwd()


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
            "from vllm import LLM, SamplingParams\n"
            "model = sys.argv[1]\n"
            "dtype = sys.argv[2]\n"
            "gpu_util = float(sys.argv[3])\n"
            "max_model_len = int(sys.argv[4])\n"
            "engine = LLM(\n"
            "    model=model,\n"
            "    tokenizer=model,\n"
            "    dtype=dtype,\n"
            "    tensor_parallel_size=1,\n"
            "    enforce_eager=True,\n"
            "    gpu_memory_utilization=gpu_util,\n"
            "    max_model_len=max_model_len,\n"
            ")\n"
            "sampling = SamplingParams(temperature=0.0, max_tokens=32)\n"
            "out = engine.generate(\n"
            "    ['Translate Japanese to natural English:\\nJapanese: \\u304a\\u306f\\u3088\\u3046\\u3054\\u3056\\u3044\\u307e\\u3059\\u3002'],\n"
            "    sampling_params=sampling,\n"
            "    use_tqdm=False,\n"
            ")\n"
            "text = (out[0].outputs[0].text or '').strip() if out and out[0].outputs else ''\n"
            "if not text:\n"
            "    raise RuntimeError('smoke output was empty')\n"
            "print(f'SMOKE_OUTPUT::{text}')\n"
        )
        try:
            smoke_env = dict(os.environ)
            if plugin:
                smoke_env["VLLM_PLUGINS"] = plugin
            cp = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    smoke_code,
                    model,
                    args.dtype,
                    str(args.gpu_memory_util),
                    str(args.max_model_len),
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
    exports = [x.strip() for x in args.export.split(",") if x.strip()]
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
    manifest = runner.run(end_stage=args.until_stage)
    summary = summarize_manifest(manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"manifest: {runner.manifest_file}")
    return 0 if manifest.status != "failed" else 2


def cmd_create(args: argparse.Namespace) -> int:
    exports = [x.strip() for x in args.export.split(",") if x.strip()]
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
    runner.save_manifest()
    print(runner.job_id)
    return 0


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


def cmd_retry(args: argparse.Namespace) -> int:
    manifest = retry_job(
        job_id=args.job,
        pages_spec=args.pages,
        stage=args.stage,
        repo_root=_repo_root(),
    )
    summary = summarize_manifest(manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if manifest.status == "done" else 2


def _resolve_page_selection(manifest, page_arg: str | None, all_pages: bool) -> set[str]:
    page_ids = sorted(manifest.pages.keys())
    if all_pages:
        return set(page_ids)

    if not page_arg:
        raise ValueError("Provide --page <page_id|index> or --all-pages")

    if page_arg in manifest.pages:
        return {page_arg}

    if page_arg.isdigit():
        idx = int(page_arg)
        if idx < 1 or idx > len(page_ids):
            raise ValueError(f"Page index out of range: {idx} (1..{len(page_ids)})")
        return {page_ids[idx - 1]}

    raise ValueError(f"Unknown page selection: {page_arg}")


def cmd_step(args: argparse.Namespace) -> int:
    if args.page and args.all_pages:
        raise ValueError("Use either --page or --all-pages, not both")
    runner = PipelineRunner.from_job(job_id=args.job, repo_root=_repo_root())
    selected = _resolve_page_selection(runner.manifest, args.page, args.all_pages)
    manifest = runner.run(
        selected_pages=selected,
        start_stage=args.stage,
        end_stage=args.stage,
        force=True,
    )
    summary = summarize_manifest(manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if manifest.status != "failed" else 2


def cmd_review(args: argparse.Namespace) -> int:
    runner = PipelineRunner.from_job(job_id=args.job, repo_root=_repo_root())
    if args.apply:
        manifest = runner.apply_review_edits(args.apply)
        print(json.dumps(summarize_manifest(manifest), ensure_ascii=False, indent=2))
        return 0 if manifest.status == "done" else 2

    manifest = runner.manifest
    print(f"Review artifacts: {runner.paths['review']}")
    for page_id in sorted(manifest.pages):
        page = manifest.pages[page_id]
        if page.output_paths.get("review"):
            print(f"- {page_id}: {page.output_paths['review']}")
    print("To apply edits: mtranslate review --job <job_id> --apply <edits.json>")
    print(
        "Edit format: "
        '{"pages":{"001":{"regions":[{"region_id":"ocr_0","enabled":false}],"blocks":[{"region_id":"ocr_1","text":"...","bbox":[x,y,w,h],"font_size":24}]}}}'
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mtranslate", description="Offline manga translation pipeline")
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
        default=(os.getenv("MTRANSLATE_VLLM_MODEL", "").strip() or DEFAULT_TRANSLATE_VLLM_REPO),
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
    run.add_argument("--until-stage", default=None, choices=STAGES, help="Stop after this stage")
    run.set_defaults(func=cmd_run)

    create = sub.add_parser("create", help="Create a job manifest without running stages")
    create.add_argument("--input", required=True, help="Input image folder")
    create.add_argument("--output", required=True, help="Output folder")
    create.add_argument("--glossary", default=None, help="Glossary yaml path")
    create.add_argument("--pre-dict", default=None, help="Regex replacement dictionary applied before translation")
    create.add_argument("--post-dict", default=None, help="Regex replacement dictionary applied after translation")
    create.add_argument("--export", default="folder,pdf", help="Comma-separated: folder,pdf")
    create.add_argument("--max-workers", type=int, default=None)
    create.add_argument("--font-path", default=None)
    create.set_defaults(func=cmd_create)

    status = sub.add_parser("status", help="Job status")
    status.add_argument("--job", required=True, help="Job id")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    retry = sub.add_parser("retry", help="Retry failed pages/stage")
    retry.add_argument("--job", required=True)
    retry.add_argument("--pages", required=True, help="Page ranges (1,3-5)")
    retry.add_argument("--stage", required=True)
    retry.set_defaults(func=cmd_retry)

    step = sub.add_parser("step", help="Run exactly one pipeline stage")
    step.add_argument("--job", required=True, help="Job id")
    step.add_argument("--stage", required=True, choices=STAGES, help="Stage to execute")
    step.add_argument("--page", default=None, help="Page id (e.g. 001) or 1-based index")
    step.add_argument("--all-pages", action="store_true", help="Run this stage for all pages")
    step.set_defaults(func=cmd_step)

    review = sub.add_parser("review", help="Review artifacts and apply edits")
    review.add_argument("--job", required=True)
    review.add_argument("--apply", default=None, help="JSON edits file")
    review.set_defaults(func=cmd_review)

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
