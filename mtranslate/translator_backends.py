"""Translator backend selection and adapters."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


class TranslatorBackend:
    name = "base"

    def warmup(self) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    def translate(self, text: str, context: Dict[str, Any]) -> str:  # pragma: no cover - interface only
        raise NotImplementedError

    def translate_batch(self, texts: Sequence[str], contexts: Sequence[Dict[str, Any]]) -> List[str]:
        if len(texts) != len(contexts):
            raise ValueError("texts and contexts must have the same length")
        return [self.translate(t, c) for t, c in zip(texts, contexts)]

    def drain_debug_events(self) -> List[Dict[str, Any]]:
        return []


BatchItem = Tuple[int, str, Dict[str, Any]]


class VLLMTranslatorBackend(TranslatorBackend):
    """JP->EN translation backend using Gemma 3 via vLLM."""

    name = "vllm"

    _lock = threading.Lock()
    _engine_cache: Dict[str, Any] = {}
    _small_kana_map = str.maketrans(
        {
            "ァ": "ア",
            "ィ": "イ",
            "ゥ": "ウ",
            "ェ": "エ",
            "ォ": "オ",
            "ッ": "ツ",
            "ャ": "ヤ",
            "ュ": "ユ",
            "ョ": "ヨ",
            "ぁ": "あ",
            "ぃ": "い",
            "ぅ": "う",
            "ぇ": "え",
            "ぉ": "お",
            "っ": "つ",
            "ゃ": "や",
            "ゅ": "ゆ",
            "ょ": "よ",
        }
    )

    def __init__(self, glossary: Optional[Dict[str, Any]] = None, model_path: str | None = None) -> None:
        self.glossary = glossary or {}
        self.model_path = (
            model_path
            or os.getenv("MTRANSLATE_VLLM_MODEL", "").strip()
            or os.getenv("MTRANSLATE_TRANSLATOR_MODEL", "").strip()
            or "google/gemma-3-4b-it"
        )
        self.vllm_plugin = os.getenv("MTRANSLATE_VLLM_PLUGIN", "").strip()
        if self.vllm_plugin:
            os.environ.setdefault("VLLM_PLUGINS", self.vllm_plugin)
        plugin_name = (self.vllm_plugin or os.getenv("VLLM_PLUGINS", "")).strip().lower()

        self.temperature = float(os.getenv("MTRANSLATE_VLLM_TEMPERATURE", "0.0"))
        self.top_p = float(os.getenv("MTRANSLATE_VLLM_TOP_P", "0.95"))
        self.max_tokens = int(os.getenv("MTRANSLATE_VLLM_MAX_TOKENS", "220"))
        self.repetition_penalty = float(os.getenv("MTRANSLATE_VLLM_REPETITION_PENALTY", "1.02"))
        self.tensor_parallel_size = int(os.getenv("MTRANSLATE_VLLM_TP", "1"))
        self.dtype = os.getenv("MTRANSLATE_VLLM_DTYPE", "auto").strip() or "auto"
        self.max_model_len = int(os.getenv("MTRANSLATE_VLLM_MAX_MODEL_LEN", "0"))
        self.gpu_memory_utilization = float(os.getenv("MTRANSLATE_VLLM_GPU_MEMORY_UTIL", "0.85"))

        self.context_lines = max(0, int(os.getenv("MTRANSLATE_VLLM_CONTEXT_LINES", "2")))
        self.context_chars = max(64, int(os.getenv("MTRANSLATE_VLLM_CONTEXT_CHARS", "260")))
        self.context_line_chars = max(24, int(os.getenv("MTRANSLATE_VLLM_CONTEXT_LINE_CHARS", "88")))
        self.min_source_chars_for_context = max(
            0, int(os.getenv("MTRANSLATE_VLLM_MIN_SOURCE_CHARS_FOR_CONTEXT", "6"))
        )

        default_batch_enabled = "0" if "vllm_mlx" in plugin_name or "mlx" in plugin_name else "1"
        self.batch_enabled = os.getenv("MTRANSLATE_VLLM_BATCH_ENABLED", default_batch_enabled).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.batch_size = max(1, int(os.getenv("MTRANSLATE_VLLM_BATCH_SIZE", "6")))
        self.batch_retries = max(0, int(os.getenv("MTRANSLATE_VLLM_BATCH_RETRIES", "2")))
        self.batch_split_depth = max(0, int(os.getenv("MTRANSLATE_VLLM_BATCH_SPLIT_DEPTH", "3")))
        self.region_retries = max(0, int(os.getenv("MTRANSLATE_VLLM_REGION_RETRIES", "1")))

        self.max_glossary_terms = max(1, int(os.getenv("MTRANSLATE_VLLM_GLOSSARY_MAX_TERMS", "12")))
        self.glossary_fuzzy_max_distance = max(0, int(os.getenv("MTRANSLATE_VLLM_GLOSSARY_FUZZY_DIST", "2")))

        self.enable_reasoning = os.getenv("MTRANSLATE_VLLM_ENABLE_REASONING", "1").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.reasoning_model_hint = os.getenv("MTRANSLATE_VLLM_REASONING_MODEL_HINT", "gemma").strip().lower()
        self.enforce_eager = os.getenv("MTRANSLATE_VLLM_ENFORCE_EAGER", "1").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.trust_remote_code = os.getenv("MTRANSLATE_VLLM_TRUST_REMOTE_CODE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }

        self.hallucination_guard = os.getenv("MTRANSLATE_VLLM_HALLUCINATION_GUARD", "1").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        symbols = os.getenv("MTRANSLATE_VLLM_SUSPICIOUS_SYMBOLS", "ହ,ി,ഹ")
        self.suspicious_symbols = [x.strip() for x in symbols.split(",") if x.strip()]
        self.max_char_expansion = max(2.0, float(os.getenv("MTRANSLATE_VLLM_MAX_CHAR_EXPANSION", "8.0")))

        self.debug_translation = os.getenv("MTRANSLATE_DEBUG_TRANSLATION", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.debug_preview_chars = max(120, int(os.getenv("MTRANSLATE_DEBUG_TRANSLATION_CHARS", "320")))
        self._debug_events: List[Dict[str, Any]] = []
        self._debug_lock = threading.Lock()

        self._memo: Dict[str, str] = {}

    def warmup(self) -> None:
        _ = self._engine()

    def _cache_key(self) -> str:
        return "|".join(
            [
                self.model_path,
                self.dtype,
                str(self.tensor_parallel_size),
                str(self.max_model_len),
                str(self.gpu_memory_utilization),
                str(int(self.enforce_eager)),
                str(int(self.trust_remote_code)),
            ]
        )

    def _engine(self):
        cache_key = self._cache_key()
        with VLLMTranslatorBackend._lock:
            cached = VLLMTranslatorBackend._engine_cache.get(cache_key)
            if cached is not None:
                return cached

            try:
                from vllm import LLM  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "vLLM translator backend requires `vllm` in the active environment."
                ) from exc

            kwargs: Dict[str, Any] = {
                "model": self.model_path,
                "tokenizer": self.model_path,
                "dtype": self.dtype,
                "tensor_parallel_size": self.tensor_parallel_size,
                "trust_remote_code": self.trust_remote_code,
                "enforce_eager": self.enforce_eager,
                "gpu_memory_utilization": self.gpu_memory_utilization,
            }
            if self.max_model_len > 0:
                kwargs["max_model_len"] = self.max_model_len

            engine = LLM(**kwargs)
            VLLMTranslatorBackend._engine_cache[cache_key] = engine
            return engine

    def _sampling_params(self):
        try:
            from vllm import SamplingParams  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("vLLM SamplingParams is unavailable") from exc

        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
            stop=["\nSOURCE_TEXT", "\nSOURCES", "\nJP:", "\nJapanese:"],
        )

    def _record_debug(self, event: Dict[str, Any]) -> None:
        if not self.debug_translation:
            return
        payload = dict(event)
        for key in ("prompt", "response", "source", "translation"):
            val = payload.get(key)
            if isinstance(val, str) and len(val) > self.debug_preview_chars:
                payload[key] = val[: self.debug_preview_chars] + "... <truncated>"
        with self._debug_lock:
            self._debug_events.append(payload)

    def drain_debug_events(self) -> List[Dict[str, Any]]:
        if not self.debug_translation:
            return []
        with self._debug_lock:
            out = list(self._debug_events)
            self._debug_events.clear()
        return out

    def _contains_japanese(self, text: str) -> bool:
        for ch in text:
            code = ord(ch)
            if (0x3040 <= code <= 0x30FF) or (0x4E00 <= code <= 0x9FFF):
                return True
        return False

    def _normalize_japanese(self, text: str) -> str:
        text = (text or "").translate(self._small_kana_map)
        out: List[str] = []
        for ch in text:
            code = ord(ch)
            if 0x30A0 <= code <= 0x30FF:
                out.append(chr(code - 0x60))
            else:
                out.append(ch)
        return "".join(out)

    def _normalize_term(self, text: str) -> str:
        norm = self._normalize_japanese((text or "").lower().strip())
        norm = re.sub(r"[^\w\u3040-\u30ff\u4e00-\u9fff]+", "", norm)
        return norm

    def _tokenize_for_fuzzy(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9\u3040-\u30ff\u4e00-\u9fff]+", text or "")
        out: List[str] = []
        for token in tokens:
            norm = self._normalize_term(token)
            if norm:
                out.append(norm)
        return out

    def _levenshtein_distance(self, s1: str, s2: str, max_distance: int | None = None) -> int:
        if s1 == s2:
            return 0
        if not s1:
            return len(s2)
        if not s2:
            return len(s1)
        if max_distance is not None and abs(len(s1) - len(s2)) > max_distance:
            return max_distance + 1

        if len(s1) < len(s2):
            s1, s2 = s2, s1

        previous = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1, start=1):
            current = [i]
            row_min = current[0]
            for j, c2 in enumerate(s2, start=1):
                insertions = previous[j] + 1
                deletions = current[j - 1] + 1
                substitutions = previous[j - 1] + (0 if c1 == c2 else 1)
                value = min(insertions, deletions, substitutions)
                current.append(value)
                row_min = min(row_min, value)
            if max_distance is not None and row_min > max_distance:
                return max_distance + 1
            previous = current
        return previous[-1]

    def _glossary_entries(self) -> List[tuple[str, str, str]]:
        entries: List[tuple[str, str, str]] = []
        for src, tgt in (self.glossary.get("characters", {}) or {}).items():
            entries.append(("character", str(src), str(tgt)))
        for src, tgt in (self.glossary.get("catchphrases", {}) or {}).items():
            entries.append(("catchphrase", str(src), str(tgt)))
        for src, tgt in (self.glossary.get("banned_literals", {}) or {}).items():
            entries.append(("banned", str(src), str(tgt)))
        return entries

    def _match_score(self, term: str, focus_text: str, focus_normalized: str, focus_tokens: List[str]) -> int | None:
        clean_term = (term or "").strip()
        if not clean_term:
            return None

        compact_focus = re.sub(r"\s+", "", focus_text)
        if clean_term in focus_text or clean_term.replace(" ", "") in compact_focus:
            return 0

        norm_term = self._normalize_term(clean_term)
        if not norm_term:
            return None
        if norm_term in focus_normalized:
            return 1

        if self._contains_japanese(clean_term):
            if len(norm_term) <= 2:
                max_dist = 0
            elif len(norm_term) <= 4:
                max_dist = 1
            else:
                max_dist = self.glossary_fuzzy_max_distance
        else:
            max_dist = max(0, min(3, len(norm_term) // 8))

        for token in focus_tokens:
            if abs(len(token) - len(norm_term)) > max_dist:
                continue
            if self._levenshtein_distance(token, norm_term, max_distance=max_dist) <= max_dist:
                return 2

        try:
            if re.search(clean_term, focus_text, flags=re.IGNORECASE):
                return 4
        except re.error:
            pass
        return None

    def _extract_relevant_terms(self, text: str, context: Dict[str, Any]) -> List[tuple[str, str, str]]:
        focus_parts = [text]
        for key in ("prev", "next", "history"):
            values = context.get(key) or []
            if isinstance(values, list):
                focus_parts.extend(str(v) for v in values if str(v).strip())
        focus_text = "\n".join(focus_parts)
        focus_norm = self._normalize_term(focus_text)
        focus_tokens = self._tokenize_for_fuzzy(focus_text)

        kind_priority = {"character": 0, "catchphrase": 1, "banned": 2}
        ranked: List[tuple[int, int, int, str, str, str]] = []
        for kind, src, tgt in self._glossary_entries():
            score = self._match_score(src, focus_text, focus_norm, focus_tokens)
            if score is None:
                continue
            ranked.append((score, kind_priority.get(kind, 9), -len(src), kind, src, tgt))

        ranked.sort()
        seen: set[tuple[str, str]] = set()
        selected: List[tuple[str, str, str]] = []
        for _, _, _, kind, src, tgt in ranked:
            key = (kind, src)
            if key in seen:
                continue
            selected.append((kind, src, tgt))
            seen.add(key)
            if len(selected) >= self.max_glossary_terms:
                break
        return selected

    def _format_glossary_lines(self, selected: Sequence[tuple[str, str, str]]) -> List[str]:
        lines: List[str] = []
        honorific = self.glossary.get("honorific_policy", "keep")
        lines.append(f"- Honorific policy: {honorific}")

        chars = [x for x in selected if x[0] == "character"]
        phrases = [x for x in selected if x[0] == "catchphrase"]
        banned = [x for x in selected if x[0] == "banned"]

        if chars:
            lines.append("- Character names:")
            for _, src, tgt in chars:
                lines.append(f"  - {src} => {tgt}")
        if phrases:
            lines.append("- Catchphrases:")
            for _, src, tgt in phrases:
                lines.append(f"  - {src} => {tgt}")
        if banned:
            lines.append("- Avoid literal forms:")
            for _, src, tgt in banned:
                lines.append(f"  - avoid '{src}', use '{tgt}'")
        return lines

    def _trim_line(self, text: str, limit: int) -> str:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(0, limit - 3)].rstrip() + "..."

    def _condense_context_lines(self, values: Any, tail: bool) -> List[str]:
        if self.context_lines <= 0:
            return []
        if not isinstance(values, list):
            return []
        raw = [self._trim_line(str(v), self.context_line_chars) for v in values if str(v).strip()]
        if not raw:
            return []
        selected = raw[-self.context_lines :] if tail else raw[: self.context_lines]
        out: List[str] = []
        remaining = self.context_chars
        for line in selected:
            cost = len(line) + 2
            if cost > remaining and out:
                break
            if cost > remaining:
                line = self._trim_line(line, max(8, remaining))
                cost = len(line) + 2
            out.append(line)
            remaining -= cost
            if remaining <= 0:
                break
        return out

    def _reasoning_line(self) -> str:
        model_lc = self.model_path.lower()
        hint = self.reasoning_model_hint
        reasoning_on = self.enable_reasoning and (
            hint in {"", "any"} or hint in model_lc or ("gemma" in model_lc and hint == "gemma")
        )
        return "Use internal step-by-step reasoning, but never reveal reasoning in output.\n" if reasoning_on else ""

    def _prompt(self, text: str, context: Dict[str, Any]) -> str:
        prev = context.get("prev") or []
        nxt = context.get("next") or []
        history = context.get("history") or []
        role = str(context.get("role", "dialogue"))
        orientation = str(context.get("orientation", "horizontal"))

        include_context = len((text or "").strip()) >= self.min_source_chars_for_context and role != "sfx"
        prev_items = self._condense_context_lines(prev, tail=True) if include_context else []
        next_items = self._condense_context_lines(nxt, tail=False) if include_context else []
        history_items = self._condense_context_lines(history, tail=True) if include_context else []

        prev_lines = "\n".join(f"- {x}" for x in prev_items)
        next_lines = "\n".join(f"- {x}" for x in next_items)
        history_lines = "\n".join(f"- {x}" for x in history_items)

        relevant = self._extract_relevant_terms(text, context)
        glossary_block = "\n".join(self._format_glossary_lines(relevant))

        return (
            "You are a professional Japanese-to-English manga translator.\n"
            "Translate only the provided Japanese text into natural English.\n"
            "Do not add explanations, notes, or alternatives.\n"
            "Keep character voice, humor, tone, and context continuity.\n"
            f"{self._reasoning_line()}"
            "For SFX, output concise natural English SFX wording.\n"
            "Respect glossary constraints exactly.\n\n"
            f"Context role: {role}\n"
            f"Original orientation: {orientation}\n\n"
            "Glossary (relevant terms only):\n"
            f"{glossary_block or '- Honorific policy: keep'}\n\n"
            "Series history (previous translated lines):\n"
            f"{history_lines or '- none'}\n\n"
            "Previous lines (for context only):\n"
            f"{prev_lines or '- none'}\n\n"
            "Next lines (for context only):\n"
            f"{next_lines or '- none'}\n\n"
            f"SOURCE_TEXT:\n{text}\n\n"
            "English translation:"
        )

    def _prompt_batch(self, items: Sequence[BatchItem]) -> str:
        first_ctx = items[0][2] if items else {}
        prev = first_ctx.get("prev") or []
        nxt = first_ctx.get("next") or []
        history = first_ctx.get("history") or []

        prev_items = self._condense_context_lines(prev, tail=True)
        next_items = self._condense_context_lines(nxt, tail=False)
        history_items = self._condense_context_lines(history, tail=True)

        aggregated: List[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for _, src, ctx in items:
            for kind, term, tgt in self._extract_relevant_terms(src, ctx):
                key = (kind, term)
                if key in seen:
                    continue
                aggregated.append((kind, term, tgt))
                seen.add(key)
                if len(aggregated) >= self.max_glossary_terms:
                    break
            if len(aggregated) >= self.max_glossary_terms:
                break
        glossary_block = "\n".join(self._format_glossary_lines(aggregated))

        source_lines: List[str] = []
        for local_idx, (_, src, ctx) in enumerate(items, start=1):
            role = str(ctx.get("role", "dialogue"))
            orientation = str(ctx.get("orientation", "horizontal"))
            source_lines.append(f"<|{local_idx}|> role={role}; orientation={orientation}; text={src}")

        return (
            "You are a professional Japanese-to-English manga translator.\n"
            "Translate each numbered SOURCE line into natural English.\n"
            "Return exactly one line per source using format: <|n|>translation\n"
            "Do not skip indices. Do not add notes.\n"
            f"{self._reasoning_line()}"
            "Respect glossary constraints exactly.\n\n"
            "Glossary (relevant terms only):\n"
            f"{glossary_block or '- Honorific policy: keep'}\n\n"
            "Series history (previous translated lines):\n"
            + ("\n".join(f"- {x}" for x in history_items) or "- none")
            + "\n\nPrevious lines:\n"
            + ("\n".join(f"- {x}" for x in prev_items) or "- none")
            + "\n\nNext lines:\n"
            + ("\n".join(f"- {x}" for x in next_items) or "- none")
            + "\n\nSOURCES:\n"
            + "\n".join(source_lines)
            + "\n\nOUTPUT:\n"
        )

    def _generate(self, prompt: str) -> str:
        engine = self._engine()
        sampling = self._sampling_params()
        outputs = engine.generate([prompt], sampling_params=sampling, use_tqdm=False)
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no translation output")
        return outputs[0].outputs[0].text or ""

    def _clean(self, text: str) -> str:
        out = text.strip()
        out = re.sub(r"^```[a-zA-Z]*", "", out).strip()
        out = re.sub(r"```$", "", out).strip()
        out = re.sub(r"^(English\s*translation\s*:|EN\s*:)", "", out, flags=re.IGNORECASE).strip()
        out = out.strip('"').strip("'").strip()
        out = out.split("\n\n")[0].strip()
        return out

    def _memo_key(self, source: str, context: Dict[str, Any]) -> str:
        role = str(context.get("role", "dialogue"))
        return f"{source}|{role}"

    def _has_repetition_hallucination(self, text: str) -> bool:
        if re.search(r"(.)\1{6,}", text):
            return True
        if re.search(r"\b(\w+)(?:\s+\1){3,}\b", text.lower()):
            return True
        if re.search(r"(.{2,12})\1{3,}", text):
            return True
        return False

    def _jp_ratio(self, text: str) -> float:
        cleaned = re.sub(r"\s+", "", text or "")
        if not cleaned:
            return 0.0
        jp = 0
        for ch in cleaned:
            code = ord(ch)
            if (0x3040 <= code <= 0x30FF) or (0x4E00 <= code <= 0x9FFF):
                jp += 1
        return jp / float(len(cleaned))

    def _translation_invalid(self, source: str, candidate: str) -> bool:
        if not self.hallucination_guard:
            return False
        text = (candidate or "").strip()
        if not text:
            return True
        if any(symbol in text for symbol in self.suspicious_symbols):
            return True
        if self._has_repetition_hallucination(text):
            return True
        if self._contains_japanese(source) and self._jp_ratio(text) >= 0.45:
            return True

        src_len = max(1, len(re.sub(r"\s+", "", source)))
        out_len = len(re.sub(r"\s+", "", text))
        if out_len > int(src_len * self.max_char_expansion):
            return True
        return False

    def translate(self, text: str, context: Dict[str, Any]) -> str:
        source = (text or "").strip()
        if not source:
            return source
        if source.startswith("__UNRESOLVED_REGION_"):
            return ""
        key = self._memo_key(source, context)
        cached = self._memo.get(key)
        if cached:
            return cached

        attempts = max(1, self.region_retries + 1)
        last_candidate = source
        for attempt in range(1, attempts + 1):
            prompt = self._prompt(source, context)
            raw = self._generate(prompt)
            candidate = self._clean(raw) or source
            invalid = self._translation_invalid(source, candidate)
            self._record_debug(
                {
                    "event": "single_translate_attempt",
                    "attempt": attempt,
                    "page_id": context.get("page_id"),
                    "region_id": context.get("region_id"),
                    "source": source,
                    "translation": candidate,
                    "invalid": invalid,
                    "prompt": prompt,
                    "response": raw,
                }
            )
            if not invalid:
                self._memo[key] = candidate
                return candidate
            last_candidate = candidate

        self._memo[key] = last_candidate
        return last_candidate

    def _parse_batch_response(self, raw: str, expected: int) -> List[str]:
        matches = list(re.finditer(r"<\|(\d+)\|>", raw))
        if not matches:
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            if len(lines) == expected:
                return lines
            raise ValueError("Missing <|n|> markers in batch response")

        parsed: Dict[int, str] = {}
        for i, match in enumerate(matches):
            idx = int(match.group(1))
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            segment = raw[start:end].strip()
            segment = re.sub(r"^[:：\-]\s*", "", segment)
            if idx in parsed:
                raise ValueError(f"Duplicate index in batch response: {idx}")
            parsed[idx] = segment

        expected_set = set(range(1, expected + 1))
        if set(parsed.keys()) != expected_set:
            raise ValueError(
                f"Batch response indices mismatch: got={sorted(parsed.keys())}, expected={sorted(expected_set)}"
            )
        return [parsed[i] for i in range(1, expected + 1)]

    def _translate_batch_recursive(self, items: Sequence[BatchItem], depth: int) -> List[str]:
        if len(items) == 1:
            _, src, ctx = items[0]
            return [self.translate(src, ctx)]

        attempts = max(1, self.batch_retries + 1)
        last_error = "unknown"
        for attempt in range(1, attempts + 1):
            prompt = self._prompt_batch(items)
            try:
                raw = self._generate(prompt)
                parsed = self._parse_batch_response(raw, expected=len(items))
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                self._record_debug(
                    {
                        "event": "batch_attempt_failed",
                        "depth": depth,
                        "attempt": attempt,
                        "size": len(items),
                        "error": last_error,
                        "prompt": prompt,
                    }
                )
                continue

            cleaned: List[str] = []
            invalid_slots: List[int] = []
            for slot, ((_, source, _), cand) in enumerate(zip(items, parsed), start=1):
                normalized = self._clean(cand) or source
                if self._translation_invalid(source, normalized):
                    invalid_slots.append(slot)
                cleaned.append(normalized)

            self._record_debug(
                {
                    "event": "batch_attempt",
                    "depth": depth,
                    "attempt": attempt,
                    "size": len(items),
                    "invalid_slots": invalid_slots,
                    "prompt": prompt,
                    "response": raw,
                }
            )

            if invalid_slots:
                last_error = f"hallucination/invalid output in slots {invalid_slots}"
                continue

            for (_, source, ctx), translation in zip(items, cleaned):
                self._memo[self._memo_key(source, ctx)] = translation
            return cleaned

        if depth < self.batch_split_depth and len(items) > 1:
            mid = len(items) // 2
            left = self._translate_batch_recursive(items[:mid], depth + 1)
            right = self._translate_batch_recursive(items[mid:], depth + 1)
            return left + right

        self._record_debug(
            {
                "event": "batch_fallback_to_single",
                "depth": depth,
                "size": len(items),
                "error": last_error,
            }
        )
        return [self.translate(src, ctx) for _, src, ctx in items]

    def translate_batch(self, texts: Sequence[str], contexts: Sequence[Dict[str, Any]]) -> List[str]:
        if len(texts) != len(contexts):
            raise ValueError("texts and contexts must have the same length")

        results: List[str] = [""] * len(texts)
        pending: List[BatchItem] = []
        for idx, (text, ctx) in enumerate(zip(texts, contexts)):
            source = (text or "").strip()
            if not source:
                results[idx] = ""
                continue
            if source.startswith("__UNRESOLVED_REGION_"):
                results[idx] = ""
                continue
            key = self._memo_key(source, ctx)
            cached = self._memo.get(key)
            if cached:
                results[idx] = cached
                continue
            pending.append((idx, source, ctx))

        if not pending:
            return results

        if not self.batch_enabled:
            for idx, source, ctx in pending:
                results[idx] = self.translate(source, ctx)
            return results

        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start : start + self.batch_size]
            chunk_out = self._translate_batch_recursive(chunk, depth=0)
            for (global_idx, _, _), translation in zip(chunk, chunk_out):
                results[global_idx] = translation

        return results


@dataclass
class TranslationSelection:
    backend: str
    backend_obj: TranslatorBackend


def select_translation_backend(glossary: Optional[Dict[str, Any]] = None) -> TranslationSelection:
    backend = os.getenv("MTRANSLATE_TRANSLATE_BACKEND", "vllm").strip().lower()
    if backend not in {"", "vllm"}:
        raise ValueError(
            f"Unsupported translate backend: {backend}. "
            "Only `vllm` is supported (Gemma 3)."
        )

    vllm_backend = VLLMTranslatorBackend(glossary=glossary)
    vllm_backend.warmup()
    return TranslationSelection(backend="vllm", backend_obj=vllm_backend)
