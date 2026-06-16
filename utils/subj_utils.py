# utils/subj_utils.py
import itertools
import json
import math
import random
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
K_DEFAULT = 8  # number of in-context examples

# SUBJ
LABELS = ["subjective", "objective"]

MANUAL_PROMPT = (
    "Please perform Subjectivity Classification. Given the sentence, assign a "
    "label from ['subjective', 'objective']. Return label only without any other text."
)


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
# Data loading
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
    """
    Supports both chat-style JSONL:
      {"messages":[{"role":"user","content":"..."}, ...], ...}
    and simpler formats:
      {"sentence":"..."} or {"text":"..."} or {"input":"..."} or {"query":"..."}
    """
    for m in obj.get("messages", []):
        if m.get("role") == "user":
            return str(m.get("content", "")).strip()

    for k in ("sentence", "text", "input", "query"):
        if k in obj and obj[k] is not None:
            s = str(obj[k]).strip()
            if s:
                return s

    return ""


def normalize_label(obj: dict) -> Optional[str]:
    """
    Canonicalize labels to one of: ['subjective','objective'].

    Supports:
      - exact strings: subjective/objective (case-insensitive)
      - short forms: subj/obj
      - numeric: 0/1 (0=objective, 1=subjective)
      - numeric: -1/+1 (objective/subjective)
      - common variants: "subj.", "obj."
    """
    lab = None
    for k in ("solution", "label", "subjectivity", "gold", "y"):
        if k in obj and obj[k] is not None:
            lab = obj[k]
            break
    if lab is None:
        return None

    s = str(lab).strip()
    if not s:
        return None
    low = s.lower().strip()

    # normalize punctuation
    low = low.replace(".", "").replace("_", "").replace("-", "").strip()

    if low in ("subjective", "subj"):
        return "subjective"
    if low in ("objective", "obj"):
        return "objective"

    if low.lstrip("+-").isdigit():
        v = int(low)
        if v == 0:
            return "objective"
        if v == 1:
            return "subjective"
        if v < 0:
            return "objective"
        if v > 0:
            return "subjective"

    # fallback keyword match
    if "subj" in low or "subject" in low:
        return "subjective"
    if "obj" in low or "object" in low:
        return "objective"

    return None


@dataclass(frozen=True)
class SubjItem:
    sentence: str
    label: str  # one of LABELS

    @property
    def key(self) -> str:
        return self.sentence


def load_subj_items(path: str, *, show_tqdm: bool = True) -> List[SubjItem]:
    raw = read_jsonl(path)
    items: List[SubjItem] = []

    it = raw
    if show_tqdm:
        it = tqdm(raw, desc=f"Parsing {path}", dynamic_ncols=True)

    for obj in it:
        s = msg_user_text(obj)
        y = normalize_label(obj)
        if s and y is not None:
            items.append(SubjItem(sentence=s, label=y))

    # dedup by sentence
    seen = set()
    dedup: List[SubjItem] = []
    for it2 in items:
        if it2.key in seen:
            continue
        seen.add(it2.key)
        dedup.append(it2)

    return dedup


# ============================================================
# Prompt formatting
# ============================================================
def format_demos(examples: List[SubjItem]) -> str:
    lines = [MANUAL_PROMPT, ""]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Sentence: {ex.sentence}")
        lines.append(f"Label: {ex.label}")
        lines.append("")
    return "\n".join(lines).strip() + "\n\n"


def build_user_content(demo_block: str, sentence: str) -> str:
    return f"{demo_block}Sentence: {sentence}\nLabel:"


# ============================================================
# Evaluator (fast cached scoring)
# ============================================================
class LabelScoringEvaluator:
    """
    Fast deterministic label scoring:
      - uses log-prob of labels after 'Label:' (no sampling)
      - supports KV-cache reuse for demo prefix
      - works with BOTH legacy tuple caches and new transformers Cache objects
    """

    def __init__(self, model_path: str, batch_size: int = 16, *, show_tqdm: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self._cuda = torch.cuda.is_available()
        self._device = torch.device("cuda:0") if self._cuda else torch.device("cpu")

        # IMPORTANT for RTX8000 (Turing, sm75): use FP16, not BF16
        if self._cuda:
            major, _minor = torch.cuda.get_device_capability(0)
            if major >= 8:
                self._dtype = torch.bfloat16  # Ampere+
            else:
                self._dtype = torch.float16   # Turing/Volta/Pascal
        else:
            self._dtype = torch.float32

        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
        )
        if self._cuda:
            model_kwargs["device_map"] = {"": 0}

        # Prefer FA2 if installed; else SDPA; else default.
        if self._cuda:
            loaded = False
            for attn_impl in ("flash_attention_2", "sdpa", None):
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
        self.show_tqdm = bool(show_tqdm)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pre-tokenize label strings (with leading space)
        self._label_token_ids: Dict[str, List[int]] = {
            lab: self.tokenizer(" " + lab, add_special_tokens=False).input_ids
            for lab in LABELS
        }
        self._single_token_labels = all(len(v) == 1 for v in self._label_token_ids.values())

        if self._cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # Tiny LRU demo cache: demo_block -> (past_cache, prefix_len_tokens)
        self._demo_cache: "OrderedDict[str, Tuple[Any, int]]" = OrderedDict()
        self._demo_cache_max = 2

        # Cache tokenization of suffixes ("{sentence}\nLabel:")
        self._suffix_token_cache: Dict[str, List[int]] = {}

    def _get_pad_id(self) -> int:
        tok = self.tokenizer
        if tok.pad_token_id is not None:
            return int(tok.pad_token_id)
        if tok.eos_token_id is not None:
            return int(tok.eos_token_id)
        return 0

    # ----------------------------
    # Cache compatibility helpers
    # ----------------------------
    def _is_new_cache(self, past: Any) -> bool:
        return past is not None and hasattr(past, "get_seq_length")

    def _to_legacy_cache(self, past: Any):
        if past is None:
            return None
        if hasattr(past, "to_legacy_cache"):
            return past.to_legacy_cache()
        return past

    def _from_legacy_cache(self, legacy: Any):
        if legacy is None:
            return None
        if DynamicCache is not None:
            try:
                return DynamicCache.from_legacy_cache(legacy)
            except Exception:
                pass
        return legacy

    def _ensure_cache_type(self, past: Any):
        if past is None:
            return None
        if self._is_new_cache(past):
            return past
        return self._from_legacy_cache(past)

    def _repeat_past(self, past_key_values: Any, batch_size: int):
        if past_key_values is None:
            return None

        legacy = self._to_legacy_cache(past_key_values)
        if legacy is None:
            return None

        rep_layers = []
        for (k, v) in legacy:
            k_rep = k.expand(batch_size, *k.shape[1:])
            v_rep = v.expand(batch_size, *v.shape[1:])
            rep_layers.append((k_rep, v_rep))

        rep_legacy = tuple(rep_layers)

        if self._is_new_cache(past_key_values):
            return self._from_legacy_cache(rep_legacy)

        return rep_legacy

    # ----------------------------
    # Vectorized (non-cached) scoring
    # ----------------------------
    @torch.inference_mode()
    def _score_sequences_suffix_vectorized(
        self,
        full_input_ids: List[List[int]],
        prefix_lens: List[int],
        suffix_lens: List[int],
    ) -> List[float]:
        mdl = self.model
        device = next(mdl.parameters()).device
        pad_id = self._get_pad_id()

        seqs = [torch.tensor(x, dtype=torch.long) for x in full_input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id).to(device)

        B, T = input_ids.shape
        attn = torch.zeros((B, T), device=device, dtype=torch.long)
        for b, s in enumerate(full_input_ids):
            attn[b, :len(s)] = 1

        out = mdl(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits  # (B,T,V)
        denom = torch.logsumexp(logits, dim=-1)  # (B,T)

        max_s = max(suffix_lens) if suffix_lens else 0
        if max_s <= 0:
            return [0.0] * B

        pref = torch.tensor(prefix_lens, device=device, dtype=torch.long).unsqueeze(1)  # (B,1)
        s_len = torch.tensor(suffix_lens, device=device, dtype=torch.long).unsqueeze(1)  # (B,1)

        j = torch.arange(max_s, device=device).unsqueeze(0)  # (1,max_s)
        pos = pref + j
        pred_pos = (pos - 1).clamp(min=0)

        mask = (j < s_len).float()

        pos_clamped = pos.clamp(max=T - 1)
        tok_ids = input_ids.gather(1, pos_clamped)

        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_s)

        tok_logits = logits[b_idx, pred_pos, tok_ids]
        tok_logp = tok_logits - denom[b_idx, pred_pos]

        scores = (tok_logp * mask).sum(dim=1).tolist()
        return [float(x) for x in scores]

    @torch.inference_mode()
    def score_prompts_labels(
        self,
        prompts: List[str],
        labels: List[str],
        *,
        desc: str = "Scoring",
    ) -> List[List[float]]:
        tok = self.tokenizer

        enc_prompts = tok(prompts, add_special_tokens=False)
        prompt_ids_list: List[List[int]] = enc_prompts["input_ids"]

        label_suffix_ids = {lab: tok(" " + lab, add_special_tokens=False).input_ids for lab in labels}

        full_ids: List[List[int]] = []
        prefix_lens: List[int] = []
        suffix_lens: List[int] = []

        for p_ids in prompt_ids_list:
            pref_len = len(p_ids)
            for lab in labels:
                suff = label_suffix_ids[lab]
                full_ids.append(p_ids + suff)
                prefix_lens.append(pref_len)
                suffix_lens.append(len(suff))

        all_scores: List[float] = []
        batches = range(0, len(full_ids), self.batch_size)
        if self.show_tqdm:
            batches = tqdm(batches, desc=desc, dynamic_ncols=True, leave=False)

        for i in batches:
            batch_ids = full_ids[i:i + self.batch_size]
            batch_p = prefix_lens[i:i + self.batch_size]
            batch_s = suffix_lens[i:i + self.batch_size]
            all_scores.extend(self._score_sequences_suffix_vectorized(batch_ids, batch_p, batch_s))

        L = len(labels)
        out: List[List[float]] = []
        idx = 0
        for _ in range(len(prompts)):
            out.append(all_scores[idx:idx + L])
            idx += L
        return out

    @torch.inference_mode()
    def predict_labels(self, prompts: List[str], *, desc: str = "Predict") -> List[str]:
        scores = self.score_prompts_labels(prompts, LABELS, desc=desc)
        preds: List[str] = []
        for row in scores:
            j = max(range(len(row)), key=lambda t: row[t])
            preds.append(LABELS[j])
        return preds

    # ----------------------------
    # Cached scoring (demo prefix KV)
    # ----------------------------
    @torch.inference_mode()
    def _get_demo_past(self, demo_block: str):
        if demo_block in self._demo_cache:
            past, prefix_len = self._demo_cache.pop(demo_block)
            self._demo_cache[demo_block] = (past, prefix_len)
            return past, prefix_len

        tok = self.tokenizer
        mdl = self.model
        device = next(mdl.parameters()).device

        prefix_text = f"{demo_block}Sentence: "
        prefix_ids = tok(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        out = mdl(input_ids=prefix_ids, use_cache=True)
        past = self._ensure_cache_type(out.past_key_values)
        prefix_len = int(prefix_ids.size(1))

        self._demo_cache[demo_block] = (past, prefix_len)
        while len(self._demo_cache) > self._demo_cache_max:
            _, (old_past, _) = self._demo_cache.popitem(last=False)
            del old_past

        return past, prefix_len

    @torch.inference_mode()
    def score_labels_for_demo_block_cached(
        self,
        demo_block: str,
        sentences: List[str],
        *,
        desc: str = "Cached scoring",
        show_progress: Optional[bool] = None,
    ) -> List[List[float]]:
        tok = self.tokenizer
        mdl = self.model
        device = next(mdl.parameters()).device
        pad_id = self._get_pad_id()

        past_prefix, prefix_len = self._get_demo_past(demo_block)
        suffix_texts = [f"{s}\nLabel:" for s in sentences]

        results: List[List[float]] = []
        bs = self.batch_size

        rng = range(0, len(suffix_texts), bs)
        use_tqdm = self.show_tqdm if show_progress is None else bool(show_progress)
        if use_tqdm:
            rng = tqdm(rng, desc=desc, dynamic_ncols=True, leave=False)

        label_ids_list = [self._label_token_ids[lab] for lab in LABELS]
        L = len(LABELS)

        for i in rng:
            batch_suffix = suffix_texts[i:i + bs]

            batch_ids_list: List[List[int]] = []
            suffix_lens: List[int] = []
            for txt in batch_suffix:
                ids = self._suffix_token_cache.get(txt)
                if ids is None:
                    ids = tok(txt, add_special_tokens=False).input_ids
                    self._suffix_token_cache[txt] = ids
                batch_ids_list.append(ids)
                suffix_lens.append(len(ids))

            seqs = [torch.tensor(x, dtype=torch.long) for x in batch_ids_list]
            suffix_ids = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id).to(device)

            B, Tnew = suffix_ids.shape
            suffix_attn = torch.zeros((B, Tnew), device=device, dtype=torch.long)
            for b, ln in enumerate(suffix_lens):
                suffix_attn[b, :ln] = 1

            prefix_attn = torch.ones((B, prefix_len), device=device, dtype=suffix_attn.dtype)
            full_attn_suffix = torch.cat([prefix_attn, suffix_attn], dim=1)

            past_B = self._repeat_past(past_prefix, B)

            if self._single_token_labels:
                out = mdl(
                    input_ids=suffix_ids,
                    attention_mask=full_attn_suffix,
                    past_key_values=past_B,
                    use_cache=False,
                )
                logits = out.logits  # (B,Tnew,V)

                last_idx = (suffix_attn.sum(dim=1) - 1).clamp(min=0)
                b_idx = torch.arange(B, device=device)
                last_logits = logits[b_idx, last_idx, :]
                denom = torch.logsumexp(last_logits, dim=-1)

                lab_token_ids = torch.tensor(
                    [self._label_token_ids[lab][0] for lab in LABELS],
                    device=device,
                    dtype=torch.long,
                )
                lab_logits = last_logits.index_select(dim=1, index=lab_token_ids)
                lab_logp = lab_logits - denom.unsqueeze(1)
                results.extend(lab_logp.tolist())
                continue

            # Multi-token labels path
            full_seqs: List[List[int]] = []
            pref_lens: List[int] = []
            suff_lens: List[int] = []

            for b in range(B):
                Ls = int(suffix_lens[b])
                base = suffix_ids[b, :Ls].tolist()
                for lab_ids in label_ids_list:
                    full_seqs.append(base + lab_ids)
                    pref_lens.append(Ls)
                    suff_lens.append(len(lab_ids))

            seqs2 = [torch.tensor(x, dtype=torch.long) for x in full_seqs]
            inp = torch.nn.utils.rnn.pad_sequence(seqs2, batch_first=True, padding_value=pad_id).to(device)
            attn2 = (inp != pad_id).long()

            BL, T2 = inp.shape
            prefix_attn2 = torch.ones((BL, prefix_len), device=device, dtype=attn2.dtype)
            full_attn2 = torch.cat([prefix_attn2, attn2], dim=1)

            past_BL = self._repeat_past(past_prefix, BL)

            out2 = mdl(
                input_ids=inp,
                attention_mask=full_attn2,
                past_key_values=past_BL,
                use_cache=False,
            )
            logits2 = out2.logits
            denom2 = torch.logsumexp(logits2, dim=-1)

            max_s = max(suff_lens) if suff_lens else 0
            pref_t = torch.tensor(pref_lens, device=device, dtype=torch.long).unsqueeze(1)
            sl_t = torch.tensor(suff_lens, device=device, dtype=torch.long).unsqueeze(1)

            j = torch.arange(max_s, device=device).unsqueeze(0)
            pos = pref_t + j
            pred_pos = (pos - 1).clamp(min=0)
            mask = (j < sl_t).float()

            pos_clamped = pos.clamp(max=T2 - 1)
            tok_ids = inp.gather(1, pos_clamped)

            b_idx2 = torch.arange(BL, device=device).unsqueeze(1).expand(BL, max_s)
            tok_logits = logits2[b_idx2, pred_pos, tok_ids]
            tok_logp = tok_logits - denom2[b_idx2, pred_pos]

            scores_flat = (tok_logp * mask).sum(dim=1).tolist()

            for b in range(B):
                row = scores_flat[b * L:(b + 1) * L]
                results.append([float(x) for x in row])

        return results

    @torch.inference_mode()
    def predict_labels_for_demo_block_cached(
        self,
        demo_block: str,
        sentences: List[str],
        *,
        desc: str = "Cached predict",
        show_progress: Optional[bool] = None,
    ) -> List[str]:
        scores = self.score_labels_for_demo_block_cached(demo_block, sentences, desc=desc, show_progress=show_progress)
        preds: List[str] = []
        for row in scores:
            j = max(range(len(row)), key=lambda t: row[t])
            preds.append(LABELS[j])
        return preds

    @torch.inference_mode()
    def generate_continuations(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        desc: str = "Generating",
    ) -> List[str]:
        tok = self.tokenizer
        mdl = self.model
        device = next(mdl.parameters()).device

        max_new_tokens = max(1, int(max_new_tokens))
        temperature = float(max(1e-6, temperature))
        top_p = float(min(1.0, max(1e-6, top_p)))

        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all()

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        outputs_text: List[str] = []

        batches = range(0, len(prompts), self.batch_size)
        if self.show_tqdm:
            batches = tqdm(batches, desc=desc, dynamic_ncols=True, leave=False)

        for i in batches:
            batch_prompts = prompts[i:i + self.batch_size]
            enc = tok(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            in_lens = attn.sum(dim=1).tolist()

            gen_ids = mdl.generate(
                input_ids=input_ids,
                attention_mask=attn,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
            )

            for b in range(gen_ids.size(0)):
                start = int(in_lens[b])
                new_tokens = gen_ids[b, start:]
                txt = tok.decode(new_tokens, skip_special_tokens=True)
                outputs_text.append(txt)

        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

        return outputs_text


# ============================================================
# Accuracy evaluation (uses cached scoring)
# ============================================================
def eval_accuracy_for_order(
    evaluator: LabelScoringEvaluator,
    demos_ordered: List[SubjItem],
    items: List[SubjItem],
    *,
    desc: str,
) -> float:
    demo_block = format_demos(demos_ordered)
    golds = [it.label for it in items]
    sentences = [it.sentence for it in items]

    preds = evaluator.predict_labels_for_demo_block_cached(
        demo_block, sentences, desc=f"Scoring ({desc})"
    )
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    return correct / max(1, len(golds))


def eval_accuracy_for_many_perms(
    evaluator: LabelScoringEvaluator,
    demos: List[SubjItem],
    perms: List[Tuple[int, ...]],
    items: List[SubjItem],
    *,
    desc: str,
) -> List[float]:
    golds = [it.label for it in items]
    sentences = [it.sentence for it in items]

    accs: List[float] = []
    perm_it = perms
    if evaluator.show_tqdm:
        perm_it = tqdm(perms, desc=desc, dynamic_ncols=True, leave=False)

    for perm in perm_it:
        ordered = [demos[i] for i in perm]
        demo_block = format_demos(ordered)
        preds = evaluator.predict_labels_for_demo_block_cached(
            demo_block,
            sentences,
            desc=f"{desc} (perm scoring)",
            show_progress=False,
        )
        correct = sum(1 for p, g in zip(preds, golds) if p == g)
        accs.append(correct / max(1, len(golds)))

    return accs


def _split_inner_outer_validation(
    val_pool: List[SubjItem],
    *,
    subset: str,
    seed: int,
    inner_frac: float = 0.8,
) -> Tuple[List[SubjItem], List[SubjItem]]:
    """
    Deterministically select a TOTAL labeled validation budget (size determined by `subset`),
    then split it into disjoint inner/outer sets by `inner_frac`.

    - subset == "full": total = full val_pool
    - subset == int:    total = deterministic sample of size subset from val_pool

    Returns:
      inner_set, outer_set
    """
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

    # Deterministic shuffle before splitting
    rng2 = random.Random(int(seed) + 7654321)
    rng2.shuffle(total)

    N = len(total)
    if N <= 0:
        return [], []

    # Edge case: too small to split disjointly
    if N < 2:
        log(
            f"[VAL] Warning: total validation size={N} too small for disjoint inner/outer split; "
            f"using same set for both (cannot avoid overlap)."
        )
        return list(total), list(total)

    inner_frac = float(max(0.0, min(1.0, inner_frac)))
    inner_size = int(round(inner_frac * N))

    # Ensure both non-empty for N>=2
    inner_size = max(1, min(N - 1, inner_size))
    outer_size = N - inner_size
    if outer_size <= 0:
        outer_size = 1
        inner_size = N - 1

    inner = total[:inner_size]
    outer = total[inner_size:]

    return inner, outer


def build_validation_eval_sets(
    val_pool: List[SubjItem],
    *,
    subset: str,
    replay_size: int,
    ce_iters: int,
    seed: int,
    resample: bool,
) -> Tuple[List[List[SubjItem]], List[SubjItem], List[SubjItem]]:
    """
    Validation set builder that returns a disjoint inner/outer split plus per-iteration inner sets.

    Returns:
      - eval_sets_inner: list of per-iteration INNER validation sets (typically used for CE updates)
      - inner_selection_set: fixed INNER set (often useful for reporting/diagnostics)
      - outer_val_set: fixed OUTER set (often useful as a final selection/eval set)

    Notes:
      - This function only constructs the split; how you *use* inner vs outer is up to the experiment.
      - Total labeled validation budget is controlled by `subset` (or full pool).
      - The inner/outer split is 80/20 of that total.
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

    # Fixed selection set (same for all algorithms)
    inner_selection_set = list(inner_set)

    # If resample is OFF: use one fixed inner set each iteration.
    if not resample:
        eval_sets_inner = [list(inner_set) for _ in range(ce_iters)]
        return eval_sets_inner, inner_selection_set, list(outer_set)

    # If resample is ON: distribute the inner set across iterations as "new" samples (without replacement),
    # plus optional replay from previous iterations.
    rng = random.Random(int(seed) + 24681357)

    inner_N = len(inner_set)
    new_per_iter = max(1, int(math.ceil(inner_N / max(1, ce_iters))))

    remaining = list(inner_set)
    per_iter_new: List[List[SubjItem]] = []
    eval_sets_inner: List[List[SubjItem]] = []

    log(f"[VAL] resample=ON: inner_N={inner_N} ce_iters={ce_iters} new_per_iter={new_per_iter}")

    for t in range(ce_iters):
        take = min(new_per_iter, len(remaining))
        new_items = rng.sample(remaining, take) if take > 0 else []
        new_keys = {x.key for x in new_items}
        remaining = [x for x in remaining if x.key not in new_keys]

        replay_items: List[SubjItem] = []
        if t > 0 and replay_size > 0:
            for j in range(t):
                prev = per_iter_new[j]
                if not prev:
                    continue
                m = min(replay_size, len(prev))
                rj = random.Random(int(seed) + 900000 + 10000 * t + j)
                replay_items.extend(rj.sample(prev, m))

        seen = set()
        merged: List[SubjItem] = []
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
    demos: List[SubjItem],
    evaluator: LabelScoringEvaluator,
    eval_sets: List[List[SubjItem]],
    selection_set: List[SubjItem],
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

        perms_t: List[Tuple[int, ...]] = []
        for _ in range(ce_batch):
            perms_t.append(sample_pl_permutation_from_logits(logits, rng))

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

        history.append(
            {
                "iter": t,
                "best_acc": best_acc_t,
                "elite_mean_acc": elite_mean,
                "best_perm": best_perm_t,
                "E": E,
                "val_size": len(items_t),
            }
        )

        if not evaluator.show_tqdm:
            log(f"[CE] iter={t} val_size={len(items_t)} best_acc={best_acc_t:.4f} elite_mean={elite_mean:.4f} E={E}")

    final_rng = random.Random(seed + 99991)
    final_candidate_perms: List[Tuple[int, ...]] = []
    for _ in range(final_draws):
        final_candidate_perms.append(sample_pl_permutation_from_logits(logits, final_rng))

    log(
        f"[CE] final selection: sampling final_draws={final_draws} perms from FINAL learned PL distribution "
        f"and evaluating on selection_set={len(selection_set)}"
    )

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
# TRUE CE / MLE update for PL
# ============================================================
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
            grad[idx] = grad[idx] + w * 1.0
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
    demos: List[SubjItem],
    evaluator: LabelScoringEvaluator,
    eval_sets: List[List[SubjItem]],
    selection_set: List[SubjItem],
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

        perms_t: List[Tuple[int, ...]] = []
        for _ in range(ce_batch):
            perms_t.append(sample_pl_permutation_from_logits(logits, rng))

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

        history.append(
            {
                "iter": t,
                "best_acc": best_acc_t,
                "elite_mean_acc": elite_mean,
                "best_perm": best_perm_t,
                "E": E,
                "val_size": len(items_t),
            }
        )

        if not evaluator.show_tqdm:
            log(f"[PL-CE-MLE] iter={t} val_size={len(items_t)} best_acc={best_acc_t:.4f} elite_mean={elite_mean:.4f} E={E}")

    final_rng = random.Random(seed + 99992)
    final_candidate_perms: List[Tuple[int, ...]] = []
    for _ in range(final_draws):
        final_candidate_perms.append(sample_pl_permutation_from_logits(logits, final_rng))

    log(
        f"[PL-CE-MLE] final selection: sampling final_draws={final_draws} perms from FINAL learned PL distribution "
        f"and evaluating on selection_set={len(selection_set)}"
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
    demos: List[SubjItem],
    evaluator: LabelScoringEvaluator,
    eval_sets: List[List[SubjItem]],
    selection_set: List[SubjItem],
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
        for _ in range(ce_batch):
            perm, _k = sample_mixture_pl_permutation(mixture_logits, mixture_weights, rng)
            perms_t.append(perm)

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
        for m in range(len(elites)):
            for k in range(K):
                w_new[k] += base_w[m] * resp[m][k]
        w_new = _normalize_probs([max(min_comp_weight, wk) for wk in w_new])

        # M-step logits: fit each component via weighted PL MLE
        logits_new: List[List[float]] = []
        for k in range(K):
            wk = [base_w[m] * resp[m][k] for m in range(len(elites))]
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
        f"[MIX-PL] final selection: sampling final_draws={final_draws} perms from FINAL learned MIXTURE distribution "
        f"and evaluating on selection_set={len(selection_set)}"
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