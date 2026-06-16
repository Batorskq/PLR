# utils/sst5_utils.py
"""
SST-5 task adapter built on top of utils/subj_utils.py.

This module retargets the subj infrastructure (evaluator and PLR optimizers)
to SST-5 Sentiment Classification by patching:
  - LABELS
  - MANUAL_PROMPT
  - normalize_label (and leaving msg_user_text as-is)

Ready to run; no placeholders.
"""

from typing import Optional

from utils import subj_utils as _su


# ============================================================
# Task config (SST-5)
# ============================================================
SST5_LABELS = ["terrible", "bad", "okay", "good", "great"]

SST5_MANUAL_PROMPT = (
    "Please perform Sentiment Classification task. Given the sentence, assign a sentiment label from "
    "['terrible', 'bad', 'okay', 'good', 'great']. Return label only without any other text."
)

# Apply task config to underlying implementation module
_su.LABELS = list(SST5_LABELS)
_su.MANUAL_PROMPT = str(SST5_MANUAL_PROMPT)


# ============================================================
# Label normalization (SST-5)
# ============================================================
def normalize_label(obj: dict) -> Optional[str]:
    """
    Canonicalize labels to one of:
      ['terrible','bad','okay','good','great'].

    Supports:
      - exact strings (case-insensitive) + common variants
      - numeric:
          * 0..4  -> terrible..great
          * 1..5  -> terrible..great
          * -2..2 -> terrible..great
    """
    lab = None
    for k in ("solution", "label", "sentiment", "gold", "y", "score", "class"):
        if k in obj and obj[k] is not None:
            lab = obj[k]
            break
    if lab is None:
        return None

    s = str(lab).strip()
    if not s:
        return None

    low = s.lower().strip()
    low = low.replace(".", "").replace("_", " ").replace("-", " ").strip()
    low = " ".join(low.split())

    # numeric labels
    if low.lstrip("+-").isdigit():
        v = int(low)
        if v in (0, 1, 2, 3, 4):
            return SST5_LABELS[v]
        if v in (1, 2, 3, 4, 5):
            return SST5_LABELS[v - 1]
        if v in (-2, -1, 0, 1, 2):
            return SST5_LABELS[v + 2]
        return None

    # direct / common string matches
    if low in ("terrible", "very negative", "veryneg", "verynegative", "strongly negative", "strong negative"):
        return "terrible"
    if low in ("bad", "negative", "neg", "poor"):
        return "bad"
    if low in ("okay", "neutral", "neu", "avg", "average", "mid", "meh", "fine"):
        return "okay"
    if low in ("good", "positive", "pos", "nice"):
        return "good"
    if low in ("great", "very positive", "verypos", "verypositive", "excellent", "amazing", "awesome"):
        return "great"

    # soft keyword fallback
    if "terrible" in low or ("very" in low and "negative" in low):
        return "terrible"
    if "bad" in low or "negative" in low:
        return "bad"
    if "okay" in low or "neutral" in low:
        return "okay"
    if "good" in low or "positive" in low:
        return "good"
    if "great" in low or ("very" in low and "positive" in low):
        return "great"

    return None


# Patch underlying label normalizer (loader uses it)
_su.normalize_label = normalize_label


# ============================================================
# Public API: re-export everything needed from subj_utils
# ============================================================
K_DEFAULT = _su.K_DEFAULT
LABELS = _su.LABELS
log = _su.log
set_all_seeds = _su.set_all_seeds

LabelScoringEvaluator = _su.LabelScoringEvaluator

# Loader for SST-5
def load_sst5_items(path: str, *, show_tqdm: bool = True):
    # Uses subj loader, but now with patched LABELS/MANUAL_PROMPT/normalize_label.
    return _su.load_subj_items(path, show_tqdm=show_tqdm)

# Validation split/builder
build_validation_eval_sets = _su.build_validation_eval_sets

# Evaluation helpers
eval_accuracy_for_order = _su.eval_accuracy_for_order
eval_accuracy_for_many_perms = _su.eval_accuracy_for_many_perms

# CE / PL optimizers
ce_optimize_order_plackett_luce = _su.ce_optimize_order_plackett_luce
ce_optimize_order_plackett_luce_mle = _su.ce_optimize_order_plackett_luce_mle

# PLR optimizers
mixture_pl_best_perm = _su.mixture_pl_best_perm