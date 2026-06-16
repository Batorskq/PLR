# utils/deepmath_utils.py
import json
import math
import random
import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from collections import OrderedDict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# Try to support both "new Cache" and "legacy tuple" past_key_values
try:
    from transformers.cache_utils import DynamicCache  # transformers >= 4.36-ish
except Exception:
    DynamicCache = None


# ============================================================
# Config
# ============================================================
K_DEFAULT = 8

BASE_PROMPT = (
    "Solve the problem. You may show your reasoning. "
    "Conclude with a line that says 'Final Answer: <answer>'. "
    "If the final answer is a number, output just this number; if it is mathematical expression, use TeX-style math typesetting (e.g., 1/2 as \\frac{1}{2}). "
    "Your answer will be considered correct if it includes the correct expression anywhere."
)

# Number detection used only for *numeric* gold answers (avoid substring false positives like gold=2 matching "12")
NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

# Simple (non-nested) fraction capture for generating a couple of alternative forms.
SIMPLE_FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")


# ============================================================
# Logging
# ============================================================
def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# Determinism helpers
# ============================================================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# JSONL helpers
# ============================================================
def read_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def msg_user_text(obj: dict) -> str:
    for m in obj.get("messages", []):
        if m.get("role") == "user":
            return str(m.get("content", "")).strip()
    for k in ("problem", "question", "query", "input", "text"):
        if k in obj and obj[k] is not None:
            s = str(obj[k]).strip()
            if s:
                return s
    return ""


def msg_assistant_text(obj: dict) -> str:
    for m in obj.get("messages", []):
        if m.get("role") == "assistant":
            return str(m.get("content", "")).strip()
    for k in ("assistant", "output", "response", "solution_text", "solution"):
        if k in obj and obj[k] is not None:
            s = str(obj[k]).strip()
            if s:
                return s
    return ""


# ============================================================
# Answer normalization (expression-aware)
# ============================================================
def normalize_answer_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def normalize_answer(obj: dict) -> Optional[str]:
    """
    DeepMath: gold answer is often in top-level "solution".
    Fallbacks: other common keys, then assistant content if it's short.
    """
    # 1) DeepMath-style: top-level solution is usually the gold final answer
    for k in ("solution", "final_answer", "answer", "target", "label", "y"):
        if k in obj and obj[k] is not None:
            v = obj[k]
            if isinstance(v, dict):
                for kk in ("answer", "value", "final", "final_answer", "solution"):
                    if kk in v:
                        vv = normalize_answer_str(v[kk])
                        if vv is not None:
                            return vv
            if isinstance(v, list) and v:
                vv = normalize_answer_str(v[-1])
                if vv is not None:
                    return vv
            vv = normalize_answer_str(v)
            if vv is not None:
                return vv

    # 2) Fallback: assistant message may itself be the short final answer (as in your examples)
    atext = msg_assistant_text(obj)
    if atext:
        m = re.search(r"final\s*answer\s*:\s*(.+)$", atext, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            vv = normalize_answer_str(m.group(1))
            if vv is not None:
                return vv

        # If it's short and single-ish line, treat it as the answer
        short = " ".join(atext.splitlines()).strip()
        if 0 < len(short) <= 200:
            return short

    return None


def _strip_tex_wrappers(s: str) -> str:
    s = (s or "").strip()
    # Remove common math wrappers
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    if s.startswith("\\(") and s.endswith("\\)") and len(s) >= 4:
        s = s[2:-2].strip()
    if s.startswith("\\[") and s.endswith("\\]") and len(s) >= 4:
        s = s[2:-2].strip()
    return s


def _normalize_number_str(s: str) -> Optional[str]:
    """
    Canonicalize a number string using Decimal-like behavior but without importing Decimal.
    We do a conservative normalization: strip leading +, normalize exponent case, strip leading zeros in integer part,
    and strip trailing zeros in fractional part.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    # must fully match NUM_RE
    if not re.fullmatch(NUM_RE, s):
        return None

    # split exponent
    if "e" in s.lower():
        # keep as-is (hard to canonicalize safely without Decimal); still usable for exact text match
        s2 = s.replace("E", "e")
        if s2.startswith("+"):
            s2 = s2[1:]
        return s2

    if s.startswith("+"):
        s = s[1:]

    sign = ""
    if s.startswith("-"):
        sign = "-"
        s = s[1:]

    if "." in s:
        a, b = s.split(".", 1)
        a = a.lstrip("0") or "0"
        b = b.rstrip("0")
        if b == "":
            return sign + a
        return sign + a + "." + b

    # integer
    s = s.lstrip("0") or "0"
    return sign + s


def _is_numeric_gold(gold: str) -> bool:
    g = _strip_tex_wrappers(gold)
    return _normalize_number_str(g) is not None


def _normalize_texish(s: str) -> str:
    """
    A lightweight normalization for TeX-ish expressions to make substring matching more robust.
    This is NOT symbolic equivalence — it's string normalization.
    """
    s = _strip_tex_wrappers(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Remove common TeX spacing commands
    for cmd in ("\\,", "\\!", "\\;", "\\:", "\\ "):
        s = s.replace(cmd, "")

    # Normalize common variants
    s = s.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")

    # Drop whitespace entirely
    s = re.sub(r"\s+", "", s)

    return s


def _gold_candidates(gold: str) -> List[str]:
    """
    Build a small set of alternative normalized strings for matching.
    This helps with a few common surface-form variations, especially simple \\frac.
    """
    g0 = _normalize_texish(gold)
    cands = []
    if g0:
        cands.append(g0)

    # If it contains simple \frac{a}{b}, add "a/b" and "(a)/(b)" variants (non-nested only)
    m = SIMPLE_FRAC_RE.fullmatch(g0)  # fullmatch after whitespace removal
    if m:
        a, b = m.group(1), m.group(2)
        cands.append(f"{a}/{b}")
        cands.append(f"({a})/({b})")

    # If gold looks like a plain a/b, add \frac{a}{b}
    if re.fullmatch(r"[^{}\\]+/[^{}\\]+", g0):
        a, b = g0.split("/", 1)
        cands.append(f"\\frac{{{a}}}{{{b}}}")

    # Also try stripping outer braces once
    if g0.startswith("{") and g0.endswith("}") and len(g0) >= 2:
        cands.append(g0[1:-1])

    # Dedup while preserving order
    seen = set()
    out = []
    for x in cands:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_all_numbers(s: str) -> List[str]:
    if s is None:
        return []
    out = []
    for m in NUM_RE.finditer(str(s)):
        norm = _normalize_number_str(m.group(0))
        if norm is not None:
            out.append(norm)
    return out


# ============================================================
# Data object
# ============================================================
@dataclass(frozen=True)
class DeepMathItem:
    problem: str
    answer: str   # gold expression string
    solution: str # demo solution text, single-line-ish

    @property
    def key(self) -> str:
        return self.problem


def _clean_solution_text(sol: str, *, answer: str) -> str:
    sol = (sol or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    sol = " ".join(sol.splitlines()).strip()
    if not sol:
        return f"Final Answer: {answer}"

    # ensure the gold answer appears somewhere (in a normalized sense)
    sol_norm = _normalize_texish(sol)
    ans_norm = _normalize_texish(answer)

    if ans_norm and ans_norm not in sol_norm:
        # if no "final answer" marker, append a proper line; else just append the answer
        if "final answer" not in sol.lower():
            sol = sol + f" Final Answer: {answer}"
        else:
            sol = sol + f" {answer}"
    return sol


def load_deepmath_items(path: str, *, show_tqdm: bool = True) -> List[DeepMathItem]:
    raw = read_jsonl(path)

    it = raw
    if show_tqdm:
        it = tqdm(raw, desc=f"Parsing {path}", dynamic_ncols=True)

    items: List[DeepMathItem] = []
    for obj in it:
        q = msg_user_text(obj)
        ans = normalize_answer(obj)
        if not q or ans is None:
            continue
        sol = msg_assistant_text(obj)
        sol_clean = _clean_solution_text(sol, answer=ans)
        items.append(DeepMathItem(problem=q, answer=ans, solution=sol_clean))

    seen = set()
    dedup: List[DeepMathItem] = []
    for it2 in items:
        if it2.key in seen:
            continue
        seen.add(it2.key)
        dedup.append(it2)

    return dedup


# ============================================================
# Prompt formatting
# ============================================================
def format_demos(examples: List[DeepMathItem]) -> str:
    lines = [BASE_PROMPT, ""]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Problem: {ex.problem}")
        lines.append(f"Solution: {ex.solution}")
        lines.append("")
    return "\n".join(lines).strip() + "\n\n"


def build_suffix_text(problem: str) -> str:
    return f"Problem: {problem}\nSolution:"


# ============================================================
# Validation splitting (generic by .key)
# ============================================================
def _split_inner_outer_validation(
    val_pool: List[Any],
    *,
    subset: str,
    seed: int,
    inner_frac: float = 0.8,
) -> Tuple[List[Any], List[Any]]:
    if not val_pool:
        return [], []

    s = str(subset).strip().lower()
    rng = random.Random(int(seed) + 1234567)

    if s == "full":
        total = list(val_pool)
    else:
        try:
            total_n_req = int(subset)
        except Exception as e:
            raise ValueError("--subset must be an integer or 'full'") from e
        total_n_req = max(1, int(total_n_req))
        take = min(total_n_req, len(val_pool))
        total = rng.sample(list(val_pool), take)

    rng2 = random.Random(int(seed) + 7654321)
    rng2.shuffle(total)

    N = len(total)
    if N <= 0:
        return [], []

    if N < 2:
        log(
            f"[VAL] Warning: total validation size={N} too small for disjoint inner/outer split; "
            f"using same set for both."
        )
        return list(total), list(total)

    inner_frac = float(max(0.0, min(1.0, inner_frac)))
    inner_size = int(round(inner_frac * N))
    inner_size = max(1, min(N - 1, inner_size))

    inner = total[:inner_size]
    outer = total[inner_size:]
    return inner, outer


def build_validation_eval_sets(
    val_pool: List[Any],
    *,
    subset: str,
    replay_size: int,
    ce_iters: int,
    seed: int,
    resample: bool,
) -> Tuple[List[List[Any]], List[Any], List[Any]]:
    """
    Returns:
      - eval_sets_inner: list of per-iteration INNER validation sets (for CE updates)
      - inner_selection_set: fixed INNER set (for diagnostics)
      - outer_val_set: fixed OUTER set (for selection)
    """
    ce_iters = int(ce_iters)
    if ce_iters <= 0:
        return [], [], []

    replay_size = max(0, int(replay_size))

    inner_set, outer_set = _split_inner_outer_validation(val_pool, subset=subset, seed=seed, inner_frac=0.8)

    total_n = len(inner_set) + len(outer_set)
    log(
        f"[VAL] nested split: total={total_n} inner={len(inner_set)} outer={len(outer_set)} "
        f"subset={subset} resample={'ON' if resample else 'OFF'} replay_size={replay_size}"
    )

    if not inner_set:
        log("[VAL] Warning: inner validation set is empty; using full val_pool as inner (outer unchanged).")
        inner_set = list(val_pool)

    inner_selection_set = list(inner_set)

    if not resample:
        eval_sets_inner = [list(inner_set) for _ in range(ce_iters)]
        return eval_sets_inner, inner_selection_set, list(outer_set)

    rng = random.Random(int(seed) + 24681357)
    inner_N = len(inner_set)
    new_per_iter = max(1, int(math.ceil(inner_N / max(1, ce_iters))))

    remaining = list(inner_set)
    per_iter_new: List[List[Any]] = []
    eval_sets_inner: List[List[Any]] = []

    log(f"[VAL] resample=ON: inner_N={inner_N} ce_iters={ce_iters} new_per_iter={new_per_iter}")

    for t in range(ce_iters):
        take = min(new_per_iter, len(remaining))
        new_items = rng.sample(remaining, take) if take > 0 else []
        new_keys = {x.key for x in new_items}
        remaining = [x for x in remaining if x.key not in new_keys]

        replay_items: List[Any] = []
        if t > 0 and replay_size > 0:
            for j in range(t):
                prev = per_iter_new[j]
                if not prev:
                    continue
                m = min(replay_size, len(prev))
                rj = random.Random(int(seed) + 900000 + 10000 * t + j)
                replay_items.extend(rj.sample(prev, m))

        seen = set()
        merged: List[Any] = []
        for it in (new_items + replay_items):
            if it.key in seen:
                continue
            seen.add(it.key)
            merged.append(it)

        per_iter_new.append(new_items)
        eval_sets_inner.append(merged)

        log(
            f"[VAL] iter={t}: new={len(new_items)} replay_added={len(replay_items)} "
            f"merged={len(merged)} remaining_inner_pool={len(remaining)}"
        )

        if len(remaining) == 0 and t + 1 < ce_iters:
            log("[VAL] Warning: inner validation pool exhausted early; later iterations may have new=0.")

    return eval_sets_inner, inner_selection_set, list(outer_set)


# ============================================================
# Evaluator (fast cached generation)
# ============================================================
class DeepMathGenerationEvaluator:
    """
    Greedy generation evaluator with demo-prefix KV caching, compatible with Qwen2 Cache objects.
    Same engine as GSM8K version; dataset-specific logic is in parsing and _is_correct().
    """

    def __init__(
        self,
        model_path: str,
        *,
        batch_size: int = 16,
        max_new_tokens: int = 96,
        show_tqdm: bool = True,
        attn_implementation: Optional[str] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self._cuda = torch.cuda.is_available()
        self._device = torch.device("cuda:0") if self._cuda else torch.device("cpu")

        if self._cuda:
            major, _minor = torch.cuda.get_device_capability(0)
            self._dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self._dtype = torch.float32

        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
        )
        if self._cuda:
            model_kwargs["device_map"] = {"": 0}

        # Choose attention implementation
        if self._cuda:
            loaded = False
            if attn_implementation is not None:
                try:
                    kw = dict(model_kwargs)
                    kw["attn_implementation"] = str(attn_implementation)
                    self.model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
                    loaded = True
                except Exception:
                    loaded = False

            if not loaded:
                for attn_impl in ("flash_attention_2", "sdpa", "eager", None):
                    try:
                        kw = dict(model_kwargs)
                        if attn_impl is not None:
                            kw["attn_implementation"] = attn_impl
                        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
                        loaded = True
                        break
                    except (ImportError, RuntimeError, ValueError, OSError, TypeError):
                        continue

            if not loaded:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            self.model.to(self._device)

        self.model.eval()

        self.batch_size = max(1, int(batch_size))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.show_tqdm = bool(show_tqdm)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # we LEFT-pad to keep last token real for all sequences
        self.tokenizer.padding_side = "left"

        if self._cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # demo_block -> (legacy_past, prefix_len_tokens)
        self._demo_cache: "OrderedDict[str, Tuple[Any, int]]" = OrderedDict()
        self._demo_cache_max = 2

        # suffix text -> token ids
        self._suffix_token_cache: Dict[str, List[int]] = {}

        # Detect whether model returns Cache object
        self._uses_cache_obj = self._detect_cache_object_usage()

    def _get_pad_id(self) -> int:
        tok = self.tokenizer
        if tok.pad_token_id is not None:
            return int(tok.pad_token_id)
        if tok.eos_token_id is not None:
            return int(tok.eos_token_id)
        return 0

    def _is_cache_obj(self, past: Any) -> bool:
        return past is not None and hasattr(past, "get_seq_length")

    @torch.inference_mode()
    def _detect_cache_object_usage(self) -> bool:
        tok = self.tokenizer
        mdl = self.model
        device = next(mdl.parameters()).device
        ids = tok("Hello", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        out = mdl(input_ids=ids, use_cache=True)
        return self._is_cache_obj(out.past_key_values)

    def _past_to_legacy(self, past: Any):
        if past is None:
            return None
        if hasattr(past, "to_legacy_cache"):
            return past.to_legacy_cache()
        return past

    def _legacy_to_past(self, legacy: Any):
        if legacy is None:
            return None
        if self._uses_cache_obj:
            if DynamicCache is None:
                raise RuntimeError(
                    "Model expects Cache objects but transformers.cache_utils.DynamicCache is unavailable."
                )
            return DynamicCache.from_legacy_cache(legacy)
        return legacy

    def _make_past_for_batch(self, legacy_prefix: Any, batch_size: int):
        """
        Expand legacy prefix KV to batch size WITHOUT copying (expand), then convert to fresh Cache.
        """
        if legacy_prefix is None:
            return None
        if batch_size == 1:
            legacy_b = legacy_prefix
        else:
            rep_layers = []
            for (k, v) in legacy_prefix:
                k_rep = k.expand(batch_size, *k.shape[1:])
                v_rep = v.expand(batch_size, *v.shape[1:])
                rep_layers.append((k_rep, v_rep))
            legacy_b = tuple(rep_layers)
        return self._legacy_to_past(legacy_b)

    @torch.inference_mode()
    def _get_demo_prefix_legacy(self, demo_block: str):
        if demo_block in self._demo_cache:
            legacy, prefix_len = self._demo_cache.pop(demo_block)
            self._demo_cache[demo_block] = (legacy, prefix_len)
            return legacy, prefix_len

        tok = self.tokenizer
        mdl = self.model
        device = next(mdl.parameters()).device

        prefix_ids = tok(demo_block, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if prefix_ids.numel() == 0:
            raise RuntimeError("Demo prefix tokenization produced 0 tokens (unexpected).")

        out = mdl(input_ids=prefix_ids, use_cache=True)
        legacy = self._past_to_legacy(out.past_key_values)
        prefix_len = int(prefix_ids.size(1))

        self._demo_cache[demo_block] = (legacy, prefix_len)
        while len(self._demo_cache) > self._demo_cache_max:
            self._demo_cache.popitem(last=False)

        return legacy, prefix_len

    @torch.inference_mode()
    def generate_for_demo_block_cached(
        self,
        demo_block: str,
        problems: List[str],
        *,
        desc: str = "Cached generate",
        show_progress: Optional[bool] = None,
    ) -> List[str]:
        if not problems:
            return []

        tok = self.tokenizer
        mdl = self.model
        device = next(mdl.parameters()).device

        pad_id = self._get_pad_id()
        eos_id = tok.eos_token_id

        legacy_prefix, prefix_len = self._get_demo_prefix_legacy(demo_block)

        suffix_texts = [build_suffix_text(p) for p in problems]

        results: List[str] = []
        bs = self.batch_size

        batches = range(0, len(suffix_texts), bs)
        use_tqdm = self.show_tqdm if show_progress is None else bool(show_progress)
        if use_tqdm:
            batches = tqdm(batches, desc=desc, dynamic_ncols=True, leave=False)

        for i in batches:
            batch_suffix = suffix_texts[i:i + bs]
            B = len(batch_suffix)

            # Tokenize suffixes (cached), collect lengths
            ids_list: List[torch.Tensor] = []
            lens: List[int] = []
            for txt in batch_suffix:
                ids = self._suffix_token_cache.get(txt)
                if ids is None:
                    ids = tok(txt, add_special_tokens=False).input_ids
                    self._suffix_token_cache[txt] = ids
                if len(ids) <= 0:
                    raise RuntimeError("Tokenized suffix length is 0; check problem text/tokenizer.")
                t = torch.tensor(ids, dtype=torch.long)
                ids_list.append(t)
                lens.append(int(t.numel()))

            max_len = max(lens)
            # Left-pad to keep last token real for all sequences
            suffix_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
            suffix_attn = torch.zeros((B, max_len), dtype=torch.long, device=device)
            for b, t in enumerate(ids_list):
                ln = int(t.numel())
                suffix_ids[b, max_len - ln:] = t.to(device)
                suffix_attn[b, max_len - ln:] = 1

            # Full attention includes prefix tokens
            prefix_attn = torch.ones((B, prefix_len), dtype=torch.long, device=device)
            full_attn = torch.cat([prefix_attn, suffix_attn], dim=1)  # (B, prefix_len+max_len)

            # Position ids for suffix tokens (pads stay at 0)
            pos = suffix_attn.cumsum(-1) - 1
            pos = torch.clamp(pos, min=0)
            pos = pos + prefix_len
            pos = pos * suffix_attn  # pads -> 0

            # Fresh per-batch past from legacy prefix
            past = self._make_past_for_batch(legacy_prefix, B)

            # Prefill suffix into cache
            out = mdl(
                input_ids=suffix_ids,
                attention_mask=full_attn,
                position_ids=pos,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values

            # First generated token (greedy) from last suffix token; with left padding it's always -1
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1)  # (B,)

            # Preallocate attention buffer: prefix+suffix + max_new_tokens
            base_len = int(full_attn.size(1))  # prefix_len + max_len
            total_len = base_len + int(self.max_new_tokens)
            attn_buf = torch.zeros((B, total_len), dtype=torch.long, device=device)
            attn_buf[:, :base_len] = full_attn

            # Per-sample next position continues after REAL suffix length
            cur_pos = torch.tensor([prefix_len + ln for ln in lens], dtype=torch.long, device=device)

            finished = torch.zeros((B,), dtype=torch.bool, device=device)
            gen_tokens: List[List[int]] = [[] for _ in range(B)]

            for step in range(int(self.max_new_tokens)):
                for b in range(B):
                    if not finished[b]:
                        gen_tokens[b].append(int(next_tok[b].item()))

                if eos_id is not None:
                    finished = finished | (next_tok == int(eos_id))
                    if bool(finished.all()):
                        break

                if eos_id is not None:
                    feed_tok = torch.where(
                        finished,
                        torch.tensor(int(eos_id), device=device, dtype=next_tok.dtype),
                        next_tok,
                    )
                else:
                    feed_tok = next_tok

                input_ids = feed_tok.view(B, 1)

                tlen = base_len + step
                attn_buf[:, tlen] = 1
                attn_step = attn_buf[:, :tlen + 1]

                pos_step = cur_pos.view(B, 1)

                out2 = mdl(
                    input_ids=input_ids,
                    attention_mask=attn_step,
                    position_ids=pos_step,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out2.past_key_values
                next_tok = torch.argmax(out2.logits[:, -1, :], dim=-1)
                cur_pos = cur_pos + 1

            for b in range(B):
                txt = tok.decode(gen_tokens[b], skip_special_tokens=True).strip()
                results.append(txt)

        return results


# ============================================================
# Accuracy evaluation
# ============================================================
def _is_correct(pred_text: str, gold_expr: str) -> bool:
    if gold_expr is None:
        return False
    gold_expr = str(gold_expr).strip()
    if not gold_expr:
        return False

    if _is_numeric_gold(gold_expr):
        gold_num = _normalize_number_str(_strip_tex_wrappers(gold_expr))
        if gold_num is None:
            return False
        preds = extract_all_numbers(pred_text)
        return gold_num in set(preds)

    pred_norm = _normalize_texish(pred_text)
    for g in _gold_candidates(gold_expr):
        if g and g in pred_norm:
            return True
    return False


def eval_accuracy_for_order(
    evaluator: DeepMathGenerationEvaluator,
    demos_ordered: List[DeepMathItem],
    items: List[DeepMathItem],
    *,
    desc: str,
) -> float:
    demo_block = format_demos(demos_ordered)
    problems = [it.problem for it in items]
    golds = [it.answer for it in items]

    preds = evaluator.generate_for_demo_block_cached(
        demo_block,
        problems,
        desc=f"Generating ({desc})",
    )

    correct = 0
    for ptxt, g in zip(preds, golds):
        if _is_correct(ptxt, g):
            correct += 1

    return correct / max(1, len(golds))


def eval_accuracy_for_many_perms(
    evaluator: DeepMathGenerationEvaluator,
    demos: List[DeepMathItem],
    perms: List[Tuple[int, ...]],
    items: List[DeepMathItem],
    *,
    desc: str,
) -> List[float]:
    problems = [it.problem for it in items]
    golds = [it.answer for it in items]

    accs: List[float] = []
    perm_it = perms
    if evaluator.show_tqdm:
        perm_it = tqdm(perms, desc=desc, dynamic_ncols=True, leave=False)

    for perm in perm_it:
        ordered = [demos[i] for i in perm]
        demo_block = format_demos(ordered)

        preds = evaluator.generate_for_demo_block_cached(
            demo_block,
            problems,
            desc=f"{desc} (perm gen)",
            show_progress=False,
        )

        correct = 0
        for ptxt, g in zip(preds, golds):
            if _is_correct(ptxt, g):
                correct += 1

        accs.append(correct / max(1, len(golds)))

    return accs


# ============================================================
# Plackett–Luce sampling + CE optimization
# ============================================================
def sample_pl_permutation_from_logits(logits: List[float], rng: random.Random) -> Tuple[int, ...]:
    scores = []
    for i, s in enumerate(logits):
        u = max(1e-12, min(1.0 - 1e-12, rng.random()))
        g = -math.log(-math.log(u))
        scores.append((s + g, i))
    scores.sort(reverse=True)
    return tuple(i for _, i in scores)


def ce_optimize_order_plackett_luce(
    demos: List[DeepMathItem],
    evaluator: DeepMathGenerationEvaluator,
    eval_sets: List[List[DeepMathItem]],
    selection_set: List[DeepMathItem],
    *,
    ce_iters: int,
    ce_batch: int,
    elite_frac: float,
    alpha: float,
    rank_temp: float,
    seed: int,
    final_draws: int,
) -> Dict:
    n = len(demos)
    if n <= 1 or ce_iters <= 0 or ce_batch <= 0:
        ident = tuple(range(n))
        val_acc = eval_accuracy_for_order(
            evaluator, [demos[i] for i in ident], selection_set, desc="identity (degenerate)"
        )
        return {
            "best_perm": ident,
            "best_val_acc": val_acc,
            "perm_draws_train": 0,
            "perm_draws_final": 0,
            "final_candidate_perms": [ident],
            "history": [],
            "final_logits": [0.0] * n,
        }

    elite_frac = float(max(0.01, min(0.99, elite_frac)))
    alpha = float(max(0.0, min(1.0, alpha)))
    rank_temp = float(max(1e-6, rank_temp))
    final_draws = int(max(1, final_draws))

    rng = random.Random(seed)
    logits = [0.0] * n
    history = []

    ce_range = range(ce_iters)
    if evaluator.show_tqdm:
        ce_range = tqdm(ce_range, desc="CE iterations (PL)", dynamic_ncols=True)

    log(f"[CE] start: iters={ce_iters} batch={ce_batch} elite_frac={elite_frac} alpha={alpha} rank_temp={rank_temp}")

    for t in ce_range:
        items_t = eval_sets[t] if t < len(eval_sets) else selection_set

        perms_t: List[Tuple[int, ...]] = [sample_pl_permutation_from_logits(logits, rng) for _ in range(ce_batch)]

        accs = eval_accuracy_for_many_perms(
            evaluator, demos, perms_t, items_t,
            desc=f"CE iter {t}: {len(perms_t)} perms on {len(items_t)} val items"
        )

        scored = list(zip(perms_t, accs))
        scored.sort(key=lambda x: x[1], reverse=True)

        best_perm_t, best_acc_t = scored[0]
        E = max(1, int(round(elite_frac * ce_batch)))
        elites = [p for p, _ in scored[:E]]
        elite_accs = [a for _, a in scored[:E]]
        elite_mean = sum(elite_accs) / max(1, len(elite_accs))

        avg_rank = [0.0] * n
        for perm in elites:
            for pos, idx in enumerate(perm):
                avg_rank[idx] += pos
        for i2 in range(n):
            avg_rank[i2] /= max(1, len(elites))

        target = [-(avg_rank[i2] / rank_temp) for i2 in range(n)]
        logits = [(1 - alpha) * logits[i2] + alpha * target[i2] for i2 in range(n)]

        history.append({"iter": t, "best_acc": best_acc_t, "elite_mean_acc": elite_mean, "best_perm": best_perm_t, "E": E, "val_size": len(items_t)})

    final_rng = random.Random(seed + 99991)
    final_candidate_perms = [sample_pl_permutation_from_logits(logits, final_rng) for _ in range(final_draws)]

    log(f"[CE] final selection: sampling final_draws={final_draws} perms and evaluating on selection_set={len(selection_set)}")

    cand_accs = eval_accuracy_for_many_perms(
        evaluator, demos, final_candidate_perms, selection_set,
        desc=f"Final select (FINAL sampling): {len(final_candidate_perms)} perms on selection_set={len(selection_set)}"
    )

    best_i = max(range(len(final_candidate_perms)), key=lambda i2: cand_accs[i2])
    best_perm = final_candidate_perms[best_i]
    best_val_acc = cand_accs[best_i]

    log(f"[CE] selected best_perm with val_acc={best_val_acc:.4f} on selection_set (size={len(selection_set)})")

    return {
        "best_perm": best_perm,
        "best_val_acc": best_val_acc,
        "perm_draws_train": ce_iters * ce_batch,
        "perm_draws_final": final_draws,
        "final_candidate_perms": final_candidate_perms,
        "history": history,
        "final_logits": list(logits),
    }


# ============================================================
# TRUE CE / MLE update for PL (same as GSM8K version)
# ============================================================
def _logsumexp_py(xs: List[float]) -> float:
    if not xs:
        return float("-inf")
    m = max(xs)
    if m == float("-inf"):
        return float("-inf")
    s = 0.0
    for x in xs:
        s += math.exp(x - m)
    return m + math.log(max(1e-300, s))


def _pl_logprob_perm(logits: List[float], perm: Tuple[int, ...]) -> float:
    remaining = list(perm)
    lp = 0.0
    for idx in perm:
        rem_scores = [logits[j] for j in remaining]
        den = _logsumexp_py(rem_scores)
        lp += logits[idx] - den
        remaining.remove(idx)
    return lp


def _normalize_probs(probs: List[float], *, eps: float = 1e-12) -> List[float]:
    s = sum(max(eps, float(p)) for p in probs)
    if s <= 0:
        return [1.0 / len(probs)] * len(probs)
    return [max(eps, float(p)) / s for p in probs]


def _pl_loglik_and_grad(
    logits_t: torch.Tensor,
    perms: List[Tuple[int, ...]],
    weights: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(logits_t.numel())
    dtype = logits_t.dtype
    device = logits_t.device

    if not perms:
        return torch.tensor(0.0, dtype=dtype, device=device), torch.zeros(n, dtype=dtype, device=device)

    if weights is None:
        weights_t = torch.ones(len(perms), dtype=dtype, device=device)
    else:
        if len(weights) != len(perms):
            raise ValueError("weights must have same length as perms")
        weights_t = torch.tensor([float(w) for w in weights], dtype=dtype, device=device)

    ll = torch.tensor(0.0, dtype=dtype, device=device)
    grad = torch.zeros(n, dtype=dtype, device=device)

    for w, perm in zip(weights_t, perms):
        remaining = list(perm)
        for idx in perm:
            rem_scores = logits_t[remaining]
            den = torch.logsumexp(rem_scores, dim=0)
            probs = torch.exp(rem_scores - den)
            for j_pos, j in enumerate(remaining):
                grad[j] = grad[j] - w * probs[j_pos]
            grad[idx] = grad[idx] + w
            ll = ll + w * (logits_t[idx] - den)
            remaining.remove(idx)

    return ll, grad


def _pl_mle_fit_adam(
    logits_init: List[float],
    elite_perms: List[Tuple[int, ...]],
    *,
    weights: Optional[List[float]] = None,
    steps: int = 50,
    lr: float = 0.5,
    l2: float = 0.0,
    clip: float = 20.0,
) -> List[float]:
    steps = int(max(1, steps))
    lr = float(max(1e-6, lr))
    l2 = float(max(0.0, l2))
    clip = float(max(1.0, clip))

    logits = torch.tensor([float(x) for x in logits_init], dtype=torch.float64, device="cpu")
    logits = logits - logits.mean()
    logits = torch.clamp(logits, -clip, clip)

    m = torch.zeros_like(logits)
    v = torch.zeros_like(logits)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for t in range(1, steps + 1):
        _ll, g = _pl_loglik_and_grad(logits, elite_perms, weights=weights)
        if l2 > 0.0:
            g = g - l2 * logits

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        mhat = m / (1.0 - beta1 ** t)
        vhat = v / (1.0 - beta2 ** t)

        step = lr * mhat / (torch.sqrt(vhat) + eps)
        logits = logits + step

        logits = logits - logits.mean()
        logits = torch.clamp(logits, -clip, clip)

    return [float(x) for x in logits.tolist()]


def ce_optimize_order_plackett_luce_mle(
    demos: List[DeepMathItem],
    evaluator: DeepMathGenerationEvaluator,
    eval_sets: List[List[DeepMathItem]],
    selection_set: List[DeepMathItem],
    *,
    ce_iters: int,
    ce_batch: int,
    elite_frac: float,
    alpha: float,
    mle_steps: int,
    mle_lr: float,
    mle_l2: float,
    mle_clip: float,
    mle_weighted: bool,
    seed: int,
    final_draws: int,
) -> Dict:
    n = len(demos)
    if n <= 1 or ce_iters <= 0 or ce_batch <= 0:
        ident = tuple(range(n))
        val_acc = eval_accuracy_for_order(
            evaluator, [demos[i] for i in ident], selection_set, desc="pl_ce_mle/identity (degenerate)"
        )
        return {
            "best_perm": ident,
            "best_val_acc": val_acc,
            "perm_draws_train": 0,
            "perm_draws_final": 0,
            "final_candidate_perms": [ident],
            "history": [],
            "final_logits": [0.0] * n,
        }

    elite_frac = float(max(0.01, min(0.99, elite_frac)))
    alpha = float(max(0.0, min(1.0, alpha)))
    final_draws = int(max(1, final_draws))

    rng = random.Random(seed)
    logits = [0.0] * n
    history = []

    ce_range = range(ce_iters)
    if evaluator.show_tqdm:
        ce_range = tqdm(ce_range, desc="CE iterations (PL MLE)", dynamic_ncols=True)

    log(
        f"[PL-CE-MLE] start: iters={ce_iters} batch={ce_batch} elite_frac={elite_frac} alpha={alpha} "
        f"mle_steps={mle_steps} lr={mle_lr} l2={mle_l2} weighted={'ON' if mle_weighted else 'OFF'}"
    )

    for t in ce_range:
        items_t = eval_sets[t] if t < len(eval_sets) else selection_set

        perms_t = [sample_pl_permutation_from_logits(logits, rng) for _ in range(ce_batch)]

        accs = eval_accuracy_for_many_perms(
            evaluator, demos, perms_t, items_t,
            desc=f"PL-CE-MLE iter {t}: {len(perms_t)} perms on {len(items_t)} val items"
        )

        scored = list(zip(perms_t, accs))
        scored.sort(key=lambda x: x[1], reverse=True)

        best_perm_t, best_acc_t = scored[0]
        E = max(1, int(round(elite_frac * ce_batch)))
        elites = [p for p, _ in scored[:E]]
        elite_accs = [a for _, a in scored[:E]]
        elite_mean = sum(elite_accs) / max(1, len(elite_accs))

        weights = None
        if mle_weighted:
            epsw = 1e-6
            w = [max(epsw, float(a)) for a in elite_accs]
            s = sum(w)
            if s > 0:
                w = [wi / s for wi in w]
            weights = w

        logits_mle = _pl_mle_fit_adam(
            logits,
            elites,
            weights=weights,
            steps=int(mle_steps),
            lr=float(mle_lr),
            l2=float(mle_l2),
            clip=float(mle_clip),
        )

        logits = [(1.0 - alpha) * logits[i2] + alpha * logits_mle[i2] for i2 in range(n)]
        meanv = sum(logits) / n
        logits = [max(-mle_clip, min(mle_clip, x - meanv)) for x in logits]

        history.append({"iter": t, "best_acc": best_acc_t, "elite_mean_acc": elite_mean, "best_perm": best_perm_t, "E": E, "val_size": len(items_t)})

    final_rng = random.Random(seed + 99992)
    final_candidate_perms = [sample_pl_permutation_from_logits(logits, final_rng) for _ in range(final_draws)]

    log(
        f"[PL-CE-MLE] final selection: sampling final_draws={final_draws} perms and evaluating on selection_set={len(selection_set)}"
    )

    cand_accs = eval_accuracy_for_many_perms(
        evaluator, demos, final_candidate_perms, selection_set,
        desc=f"PL-CE-MLE final select (FINAL sampling): {len(final_candidate_perms)} perms on selection_set={len(selection_set)}"
    )

    best_i = max(range(len(final_candidate_perms)), key=lambda i2: cand_accs[i2])
    best_perm = final_candidate_perms[best_i]
    best_val_acc = cand_accs[best_i]

    log(f"[PL-CE-MLE] selected best_perm with val_acc={best_val_acc:.4f} on selection_set (size={len(selection_set)})")

    return {
        "best_perm": best_perm,
        "best_val_acc": best_val_acc,
        "perm_draws_train": ce_iters * ce_batch,
        "perm_draws_final": final_draws,
        "final_candidate_perms": final_candidate_perms,
        "history": history,
        "final_logits": list(logits),
    }


# ============================================================
# Mixture-PL distribution
# ============================================================
def sample_mixture_pl_permutation(
    mixture_logits: List[List[float]],
    mixture_weights: List[float],
    rng: random.Random,
) -> Tuple[Tuple[int, ...], int]:
    K = len(mixture_logits)
    if K <= 0:
        raise ValueError("mixture_logits must be non-empty")
    w = _normalize_probs(mixture_weights)

    u = rng.random()
    cdf = 0.0
    k = K - 1
    for i, wi in enumerate(w):
        cdf += wi
        if u <= cdf:
            k = i
            break

    perm = sample_pl_permutation_from_logits(mixture_logits[k], rng)
    return perm, k


def mixture_pl_best_perm(
    demos: List[DeepMathItem],
    evaluator: DeepMathGenerationEvaluator,
    eval_sets: List[List[DeepMathItem]],
    selection_set: List[DeepMathItem],
    *,
    ce_iters: int,
    ce_batch: int,
    elite_frac: float,
    alpha: float,
    mix_components: int,
    mle_steps: int,
    mle_lr: float,
    mle_l2: float,
    mle_clip: float,
    mle_weighted: bool,
    seed: int,
    final_draws: int,
) -> Dict:
    n = len(demos)
    if n <= 1 or ce_iters <= 0 or ce_batch <= 0:
        ident = tuple(range(n))
        val_acc = eval_accuracy_for_order(
            evaluator, [demos[i] for i in ident], selection_set, desc="mixture_pl/identity (degenerate)"
        )
        return {
            "best_perm": ident,
            "best_val_acc": val_acc,
            "perm_draws_train": 0,
            "perm_draws_final": 0,
            "final_candidate_perms": [ident],
            "history": [],
            "mix_components": int(max(1, mix_components)),
            "final_mix_weights": [1.0],
            "final_mix_logits": [[0.0] * n],
        }

    elite_frac = float(max(0.01, min(0.99, elite_frac)))
    alpha = float(max(0.0, min(1.0, alpha)))
    final_draws = int(max(1, final_draws))

    K = int(max(1, mix_components))
    rng = random.Random(seed)

    mixture_logits: List[List[float]] = [[0.0] * n for _ in range(K)]
    mixture_weights: List[float] = [1.0 / K] * K

    history = []

    ce_range = range(ce_iters)
    if evaluator.show_tqdm:
        ce_range = tqdm(ce_range, desc="CE iterations (Mixture-PL)", dynamic_ncols=True)

    log(
        f"[MIX-PL] start: iters={ce_iters} batch={ce_batch} elite_frac={elite_frac} alpha={alpha} "
        f"K={K} mle_steps={mle_steps} lr={mle_lr} l2={mle_l2} weighted={'ON' if mle_weighted else 'OFF'}"
    )

    min_comp_weight = 1e-3

    for t in ce_range:
        items_t = eval_sets[t] if t < len(eval_sets) else selection_set

        perms_t: List[Tuple[int, ...]] = []
        comp_ids: List[int] = []
        for _ in range(ce_batch):
            perm, kk = sample_mixture_pl_permutation(mixture_logits, mixture_weights, rng)
            perms_t.append(perm)
            comp_ids.append(kk)

        accs = eval_accuracy_for_many_perms(
            evaluator, demos, perms_t, items_t,
            desc=f"MIX-PL iter {t}: {len(perms_t)} perms on {len(items_t)} val items"
        )

        scored = list(zip(perms_t, accs))
        scored.sort(key=lambda x: x[1], reverse=True)

        best_perm_t, best_acc_t = scored[0]
        E = max(1, int(round(elite_frac * ce_batch)))
        elites = [p for p, _ in scored[:E]]
        elite_accs = [a for _, a in scored[:E]]
        elite_mean = sum(elite_accs) / max(1, len(elite_accs))

        if mle_weighted:
            epsw = 1e-6
            base_w = [max(epsw, float(a)) for a in elite_accs]
        else:
            base_w = [1.0 for _ in elite_accs]

        # E-step responsibilities
        resp: List[List[float]] = []
        for perm in elites:
            logps = []
            for k in range(K):
                lp = math.log(max(1e-300, mixture_weights[k])) + _pl_logprob_perm(mixture_logits[k], perm)
                logps.append(lp)
            z = _logsumexp_py(logps)
            r = [math.exp(lp - z) for lp in logps]
            resp.append(r)

        # M-step weights
        w_new = [0.0] * K
        for m_idx in range(len(elites)):
            for k in range(K):
                w_new[k] += base_w[m_idx] * resp[m_idx][k]
        w_new = _normalize_probs([max(min_comp_weight, wk) for wk in w_new])

        # M-step logits: fit each component via weighted PL MLE
        logits_new: List[List[float]] = []
        for k in range(K):
            wk = [base_w[m_idx] * resp[m_idx][k] for m_idx in range(len(elites))]
            sumwk = sum(wk)
            if sumwk <= 1e-12:
                logits_new.append(list(mixture_logits[k]))
                continue
            wk_norm = [w / sumwk for w in wk]
            fitted = _pl_mle_fit_adam(
                mixture_logits[k],
                elites,
                weights=wk_norm,
                steps=int(mle_steps),
                lr=float(mle_lr),
                l2=float(mle_l2),
                clip=float(mle_clip),
            )
            logits_new.append(fitted)

        mixture_weights = [(1.0 - alpha) * mixture_weights[k] + alpha * w_new[k] for k in range(K)]
        mixture_weights = _normalize_probs([max(min_comp_weight, wk) for wk in mixture_weights])

        for k in range(K):
            blended = [(1.0 - alpha) * mixture_logits[k][i2] + alpha * logits_new[k][i2] for i2 in range(n)]
            meanv = sum(blended) / n
            blended = [max(-mle_clip, min(mle_clip, x - meanv)) for x in blended]
            mixture_logits[k] = blended

        history.append(
            {
                "iter": t,
                "best_acc": best_acc_t,
                "elite_mean_acc": elite_mean,
                "best_perm": best_perm_t,
                "E": E,
                "val_size": len(items_t),
                "mix_weights": list(mixture_weights),
            }
        )

    final_rng = random.Random(seed + 99993)
    final_candidate_perms: List[Tuple[int, ...]] = []
    for _ in range(final_draws):
        perm, _k = sample_mixture_pl_permutation(mixture_logits, mixture_weights, final_rng)
        final_candidate_perms.append(perm)

    log(
        f"[MIX-PL] final selection: sampling final_draws={final_draws} perms and evaluating on selection_set={len(selection_set)}"
    )

    cand_accs = eval_accuracy_for_many_perms(
        evaluator, demos, final_candidate_perms, selection_set,
        desc=f"MIX-PL final select (FINAL sampling): {len(final_candidate_perms)} perms on selection_set={len(selection_set)}"
    )
    best_i = max(range(len(final_candidate_perms)), key=lambda i2: cand_accs[i2])
    best_perm = final_candidate_perms[best_i]
    best_val_acc = cand_accs[best_i]

    log(f"[MIX-PL] selected best_perm with val_acc={best_val_acc:.4f} on selection_set (size={len(selection_set)})")

    return {
        "best_perm": best_perm,
        "best_val_acc": best_val_acc,
        "perm_draws_train": ce_iters * ce_batch,
        "perm_draws_final": final_draws,
        "final_candidate_perms": final_candidate_perms,
        "history": history,
        "mix_components": K,
        "final_mix_weights": list(mixture_weights),
        "final_mix_logits": [list(x) for x in mixture_logits],
    }