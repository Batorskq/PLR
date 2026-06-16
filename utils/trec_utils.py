# utils/trec_utils.py
"""
TREC (Question Classification) task adapter built on top of utils/subj_utils.py.

This module "retargets" the subj task infrastructure (evaluator and PLR optimizers)
to the TREC coarse Question Classification task by patching:
  - LABELS
  - MANUAL_PROMPT
  - msg_user_text
  - normalize_label
  - format_demos / build_user_content

No placeholders; ready to run.

Coarse TREC labels:
  ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM']
"""

from typing import List, Optional, Dict
import re

from utils import subj_utils as _su


# ============================================================
# Task config (TREC coarse)
# ============================================================
TREC_LABELS = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]

TREC_MANUAL_PROMPT = (
    "Please perform TREC Question Classification. Given the question, assign a label from "
    "['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM']. Return label only without any other text."
)

# Apply task config to the underlying implementation module
_su.LABELS = list(TREC_LABELS)
_su.MANUAL_PROMPT = str(TREC_MANUAL_PROMPT)


# ============================================================
# Data extraction + label normalization (TREC)
# ============================================================
def msg_user_text(obj: dict) -> str:
    """
    Supports chat-style JSONL:
      {"messages":[{"role":"user","content":"..."}, ...], ...}
    and common TREC formats:
      {"question":"..."} {"text":"..."} {"sentence":"..."} {"query":"..."} {"input":"..."}
    """
    for m in obj.get("messages", []):
        if m.get("role") == "user":
            return str(m.get("content", "")).strip()

    for k in ("question", "q", "text", "sentence", "query", "input", "prompt"):
        if k in obj and obj[k] is not None:
            s = str(obj[k]).strip()
            if s:
                return s

    return ""


_ABBR_PAT = re.compile(r"\b(abbrev|abbreviation|acronym|stands\s+for)\b", re.I)
_HUM_PAT = re.compile(r"^\s*(who|whom|whose)\b", re.I)
_LOC_PAT = re.compile(r"^\s*(where)\b", re.I)
_NUM_PAT = re.compile(r"^\s*(when|how\s+(many|much|old|long|far)|what\s+(year|date|time|number))\b", re.I)


def normalize_label(obj: dict) -> Optional[str]:
    """
    Canonicalize labels to one of:
      ['ABBR','ENTY','DESC','HUM','LOC','NUM'].

    Supports:
      - exact coarse strings (case-insensitive)
      - fine-grained TREC strings like "HUM:ind", "ENTY:animal" (takes prefix before ':')
      - numeric: 0..5 (0=ABBR,1=ENTY,2=DESC,3=HUM,4=LOC,5=NUM)
      - numeric: 1..6 (1=ABBR,...,6=NUM)
      - some common synonyms/variants
    """
    lab = None
    for k in ("solution", "label", "coarse_label", "class", "category", "gold", "y", "target"):
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
        if v in (0, 1, 2, 3, 4, 5):
            return TREC_LABELS[v]
        if v in (1, 2, 3, 4, 5, 6):
            return TREC_LABELS[v - 1]
        return None

    # handle fine labels like "HUM:ind"
    if ":" in low:
        low = low.split(":", 1)[0].strip()

    # direct coarse matches
    if low in ("abbr", "abbreviation", "acronym"):
        return "ABBR"
    if low in ("enty", "entity", "entities"):
        return "ENTY"
    if low in ("desc", "description", "definition", "def"):
        return "DESC"
    if low in ("hum", "human", "person", "people"):
        return "HUM"
    if low in ("loc", "location", "place"):
        return "LOC"
    if low in ("num", "number", "numeric", "quantity", "date", "time"):
        return "NUM"

    # fallback keyword heuristics (only if dataset has messy labels)
    q = msg_user_text(obj)
    if q:
        if _ABBR_PAT.search(q):
            return "ABBR"
        if _HUM_PAT.search(q):
            return "HUM"
        if _LOC_PAT.search(q):
            return "LOC"
        if _NUM_PAT.search(q):
            return "NUM"

    return None


# Patch the underlying functions used by loader
_su.msg_user_text = msg_user_text
_su.normalize_label = normalize_label


# ============================================================
# Prompt formatting (TREC)
# ============================================================
def format_demos(examples) -> str:
    lines = [_su.MANUAL_PROMPT, ""]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Question: {ex.sentence}")
        lines.append(f"Label: {ex.label}")
        lines.append("")
    return "\n".join(lines).strip() + "\n\n"


def build_user_content(demo_block: str, question: str) -> str:
    return f"{demo_block}Question: {question}\nLabel:"


_su.format_demos = format_demos
_su.build_user_content = build_user_content


# ============================================================
# Public API: re-export everything needed from subj_utils
# ============================================================
K_DEFAULT = _su.K_DEFAULT
LABELS = _su.LABELS
log = _su.log
set_all_seeds = _su.set_all_seeds

LabelScoringEvaluator = _su.LabelScoringEvaluator

# Loader for TREC
def load_trec_items(path: str, *, show_tqdm: bool = True):
    # This calls the original loader, but it now uses the patched msg_user_text/normalize_label above.
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