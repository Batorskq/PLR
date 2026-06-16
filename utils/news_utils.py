# utils/news_utils.py
"""
News task adapter built on top of utils/subj_utils.py.

This module "retargets" the subj task infrastructure (evaluator and PLR optimizers)
to the News Classification task by patching:
  - LABELS
  - MANUAL_PROMPT
  - msg_user_text
  - normalize_label
  - format_demos / build_user_content

No placeholders; ready to run.
"""

from typing import List, Optional, Dict
import re

from utils import subj_utils as _su


# ============================================================
# Task config (News)
# ============================================================
NEWS_LABELS = ["World", "Sports", "Business", "Tech"]

NEWS_MANUAL_PROMPT = (
    "Please perform News Classification task. Given the news item, assign a label from "
    "['World', 'Sports', 'Business', 'Tech']. Return label only without any other text."
)


# Apply task config to the underlying implementation module
_su.LABELS = list(NEWS_LABELS)
_su.MANUAL_PROMPT = str(NEWS_MANUAL_PROMPT)


# ============================================================
# Data extraction + label normalization (News)
# ============================================================
def msg_user_text(obj: dict) -> str:
    """
    Supports chat-style JSONL:
      {"messages":[{"role":"user","content":"..."}, ...], ...}
    and common news formats:
      {"news":"..."} {"headline":"..."} {"title":"..."} {"text":"..."} {"content":"..."} {"body":"..."} {"article":"..."}
    """
    for m in obj.get("messages", []):
        if m.get("role") == "user":
            return str(m.get("content", "")).strip()

    for k in ("news", "headline", "title", "sentence", "text", "content", "body", "article", "input", "query"):
        if k in obj and obj[k] is not None:
            s = str(obj[k]).strip()
            if s:
                return s
    return ""


_WORLD_PAT = re.compile(r"\b(world|international|nation|politics|government|diplomacy|europe|asia|africa|middle\s*east)\b", re.I)
_SPORTS_PAT = re.compile(r"\b(sports?|sport|match|game|league|tournament|nba|nfl|mlb|nhl|soccer|football|cricket|tennis)\b", re.I)
_BUS_PAT = re.compile(r"\b(business|finance|market|stocks?|economy|economic|trade|company|companies|earnings|revenue)\b", re.I)
_TECH_PAT = re.compile(r"\b(tech|technology|ai|software|hardware|computer|internet|device|smartphone|chip|semiconductor|startup)\b", re.I)


def normalize_label(obj: dict) -> Optional[str]:
    """
    Canonicalize labels to one of: ['World','Sports','Business','Tech'].

    Supports:
      - exact strings: World/Sports/Business/Tech (case-insensitive)
      - common variants: world news, sport, biz, technology
      - numeric: 0/1/2/3 (0=World,1=Sports,2=Business,3=Tech)
      - numeric: 1/2/3/4 (1=World,2=Sports,3=Business,4=Tech)
    """
    lab = None
    for k in ("solution", "label", "category", "topic", "gold", "y"):
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
        if v in (0, 1, 2, 3):
            return NEWS_LABELS[v]
        if v in (1, 2, 3, 4):
            return NEWS_LABELS[v - 1]
        return None

    # direct / common string matches
    if low in ("world", "worldnews", "world news", "international", "international news"):
        return "World"
    if low in ("sports", "sport", "sporting"):
        return "Sports"
    if low in ("business", "biz", "finance", "economy", "economic"):
        return "Business"
    if low in ("tech", "technology", "sci", "science", "it"):
        return "Tech"

    # keyword fallback
    if _WORLD_PAT.search(low):
        return "World"
    if _SPORTS_PAT.search(low):
        return "Sports"
    if _BUS_PAT.search(low):
        return "Business"
    if _TECH_PAT.search(low):
        return "Tech"

    return None


# Patch the underlying functions used by loader
_su.msg_user_text = msg_user_text
_su.normalize_label = normalize_label


# ============================================================
# Prompt formatting (News)
# ============================================================
def format_demos(examples) -> str:
    lines = [_su.MANUAL_PROMPT, ""]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"News item: {ex.sentence}")
        lines.append(f"Label: {ex.label}")
        lines.append("")
    return "\n".join(lines).strip() + "\n\n"


def build_user_content(demo_block: str, news_item: str) -> str:
    return f"{demo_block}News item: {news_item}\nLabel:"


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

# Loader for news
def load_news_items(path: str, *, show_tqdm: bool = True):
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