# utils/mr_utils.py
"""
MR (Movie Review) sentiment task adapter built on top of utils/subj_utils.py.

This module retargets the subj infrastructure (evaluator and PLR optimizers)
to MR Sentiment Classification by patching:
  - LABELS
  - MANUAL_PROMPT
  - normalize_label

Ready to run; no placeholders.
"""

from typing import Optional

from utils import subj_utils as _su


# ============================================================
# Task config (MR)
# ============================================================
MR_LABELS = ["positive", "negative"]

MR_MANUAL_PROMPT = (
    "Please perform Sentiment Classification task. Given the sentence, assign a "
    "label from ['positive', 'negative']. Return label only without any other text."
)

# Apply task config to underlying implementation module
_su.LABELS = list(MR_LABELS)
_su.MANUAL_PROMPT = str(MR_MANUAL_PROMPT)


# ============================================================
# Label normalization (MR)
# ============================================================
def normalize_label(obj: dict) -> Optional[str]:
    """
    Canonicalize labels to one of: ['positive','negative'].

    Supports:
      - exact strings (case-insensitive): positive/negative
      - common variants: pos/neg, +/-, 1/0, 1/-1
      - keyword fallback
    """
    lab = None
    for k in ("solution", "label", "sentiment", "gold", "y", "polarity", "class"):
        if k in obj and obj[k] is not None:
            lab = obj[k]
            break
    if lab is None:
        return None

    s = str(lab).strip()
    if not s:
        return None

    low = s.lower().strip()
    low = low.replace(".", "").replace("_", "").replace("-", "").strip()

    if low in ("positive", "pos", "+", "plus"):
        return "positive"
    if low in ("negative", "neg", "-", "minus"):
        return "negative"

    if low.lstrip("+-").isdigit():
        v = int(low)
        # Common encodings:
        # 1=positive, 0=negative
        if v == 1:
            return "positive"
        if v == 0:
            return "negative"
        # +/-1 scheme
        if v > 0:
            return "positive"
        if v < 0:
            return "negative"

    # keyword fallback
    if "pos" in low or "positive" in low:
        return "positive"
    if "neg" in low or "negative" in low:
        return "negative"

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


def load_mr_items(path: str, *, show_tqdm: bool = True):
    # Uses subj loader, but now with patched LABELS/MANUAL_PROMPT/normalize_label.
    return _su.load_subj_items(path, show_tqdm=show_tqdm)


build_validation_eval_sets = _su.build_validation_eval_sets

eval_accuracy_for_order = _su.eval_accuracy_for_order
eval_accuracy_for_many_perms = _su.eval_accuracy_for_many_perms

ce_optimize_order_plackett_luce = _su.ce_optimize_order_plackett_luce
ce_optimize_order_plackett_luce_mle = _su.ce_optimize_order_plackett_luce_mle

mixture_pl_best_perm = _su.mixture_pl_best_perm