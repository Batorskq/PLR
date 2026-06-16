# subj.py
import argparse
import random
from typing import List, Tuple, Optional, Dict
import os
from collections import OrderedDict
import re

from utils.subj_utils import (
    K_DEFAULT,
    LABELS,
    log,
    set_all_seeds,
    load_subj_items,
    LabelScoringEvaluator,
    build_validation_eval_sets,
    eval_accuracy_for_order,
    ce_optimize_order_plackett_luce,
    ce_optimize_order_plackett_luce_mle,
    mixture_pl_best_perm,
)


def _sanitize_tag(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(" ", "")
    s = s.replace("/", "-")
    s = s.replace("\\", "-")
    s = s.replace(":", "-")
    return s or "none"


def results_dir_from_args(args) -> str:
    subset_tag = _sanitize_tag(getattr(args, "subset", "unknown"))
    k_tag = int(getattr(args, "k", 0))
    return f"results_elite_05_subset{subset_tag}_k{k_tag}"


def _fmt_float_for_key(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace("E", "e")


def make_pl_key(
    base: str,
    *,
    ce_iters: int,
    final_draws: int,
    lr: Optional[float] = None,
) -> str:
    base = str(base).strip().lower()
    key = f"{base}__ce{int(ce_iters)}__final{int(final_draws)}"
    if lr is not None:
        key += f"__lr{_fmt_float_for_key(lr)}"
    return key



def parse_mixture_pl_method(method_name: str) -> Optional[int]:
    return 4 if str(method_name).strip().lower() == "mixture_pl_4" else None


def method_to_result_key(method_name: str, args, final_draws: int) -> str:
    b = str(method_name).strip().lower()

    if b == "method_pl_ce":
        return make_pl_key("method_pl_ce", ce_iters=int(args.ce_iters), final_draws=int(final_draws))

    if b == "pl_ce_mle":
        return make_pl_key(
            "pl_ce_mle",
            ce_iters=int(args.ce_iters),
            final_draws=int(final_draws),
            lr=float(args.ce_mle_lr),
        )

    mk = parse_mixture_pl_method(b)
    if mk is not None:
        return make_pl_key(
            f"mixture_pl_{mk}",
            ce_iters=int(args.ce_iters),
            final_draws=int(final_draws),
            lr=float(args.ce_mle_lr),
        )

    return b


_TEST_ACC_RE = re.compile(r"test_acc=([0-9]+(?:\.[0-9]+)?)")
_VAL_ACC_RE = re.compile(r"val_acc=([0-9]+(?:\.[0-9]+)?)")


def _parse_accs_from_result_line(line: str) -> Tuple[float, float]:
    try:
        mt = _TEST_ACC_RE.search(line)
        mv = _VAL_ACC_RE.search(line)
        test_acc = float(mt.group(1)) if mt else float("-inf")
        val_acc = float(mv.group(1)) if mv else float("-inf")
        return test_acc, val_acc
    except Exception:
        return float("-inf"), float("-inf")


def parse_seed_list(seed_arg) -> List[int]:
    if seed_arg is None:
        return [123]
    s = str(seed_arg).strip()
    if not s:
        return [123]

    tokens: List[str] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        for t in chunk.split():
            if t:
                tokens.append(t)

    seeds: List[int] = []
    for t in tokens:
        tt = t.strip()
        if not tt:
            continue
        try:
            seeds.append(int(tt))
        except Exception as e:
            raise ValueError(
                f"--seed must be an int or a comma/space-separated list of ints; got '{seed_arg}'"
            ) from e

    if not seeds:
        return [123]

    seen = set()
    out: List[int] = []
    for x in seeds:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def read_results_file(path: str) -> Tuple["OrderedDict[int, OrderedDict[str, str]]", List[int]]:
    seed_data: "OrderedDict[int, OrderedDict[str, str]]" = OrderedDict()
    seed_order: List[int] = []

    if not os.path.exists(path):
        return seed_data, seed_order

    current_seed: Optional[int] = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            stripped = raw.strip()
            if not stripped:
                continue

            low = stripped.lower()
            if low.startswith("seed"):
                parts = stripped.replace(":", " ").split()
                seed_val: Optional[int] = None
                for p in parts[1:]:
                    pp = p.strip()
                    if pp.lstrip("+-").isdigit():
                        seed_val = int(pp)
                        break
                current_seed = seed_val
                if current_seed is None:
                    continue
                if current_seed not in seed_data:
                    seed_data[current_seed] = OrderedDict()
                    seed_order.append(current_seed)
                continue

            if current_seed is None:
                continue

            if ":" in stripped:
                name = stripped.split(":", 1)[0].strip()
                if name and name not in seed_data[current_seed]:
                    seed_data[current_seed][name] = stripped

    return seed_data, seed_order


def _dedup_by_key(items: List) -> List:
    seen = set()
    out = []
    for x in items:
        k = x.key
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out



def write_results_file(
    path: str,
    seed_data: "OrderedDict[int, OrderedDict[str, str]]",
    seed_order: List[int],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for si, seed in enumerate(seed_order):
            f.write(f"Seed {seed}\n\n")
            block = seed_data.get(seed, OrderedDict())

            names = list(block.keys())

            def sort_key(name: str):
                line = block.get(name, "")
                test_acc, val_acc = _parse_accs_from_result_line(line)
                return (-test_acc, -val_acc, str(name).lower())

            names.sort(key=sort_key)

            for name in names:
                f.write(block[name] + "\n")

            if si != len(seed_order) - 1:
                f.write("\n")


def format_result_line(
    name: str,
    test_acc: float,
    val_acc: float,
    perm: Tuple[int, ...],
    extra: Optional[str] = None,
) -> str:
    perm_str = "[" + ",".join(map(str, perm)) + "]"
    base = f"{name}: test_acc={test_acc:.4f} val_acc={val_acc:.4f} perm={perm_str}"
    if extra:
        base += f" | {extra}"
    return base


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        type=str,
        default="/gpfs/project/lox20cow/ms-swift/qwen7binstruct",
        help="HF model id or local path.",
    )

    ap.add_argument(
        "--seed",
        type=str,
        default="1,2,3,4,5",
        help="Seed(s). Examples: --seed 123  OR  --seed 1,2,3  OR  --seed '1 2 3'.",
    )

    ap.add_argument("--k", type=int, default=K_DEFAULT, help="Number of in-context examples.")
    ap.add_argument("--infer-path", type=str, default="dataset/subj_infer.jsonl", help="Path to subj_infer.jsonl.")
    ap.add_argument("--test-path", type=str, default="dataset/subj_test.jsonl", help="Path to subj_test.jsonl.")

    ap.add_argument(
        "--subset",
        type=str,
        default="1000",
        help="Validation subset size per CE iteration: integer (e.g. 64) or 'full'.",
    )
    ap.add_argument(
        "--replay-size",
        type=int,
        default=10,
        help="Replay buffer size per previous iteration (only used when --subset is an integer AND --resample is set).",
    )
    ap.add_argument(
        "--resample",
        action="store_true",
        help=(
            "If set, resample a NEW validation subset each CE iteration (enables replay when --subset is integer). "
            "Default: OFF (use one fixed subset; replay ignored)."
        ),
    )

    ap.add_argument("--ce-iters", type=int, default=15, help="Number of CE iterations.")
    ap.add_argument("--ce-batch", type=int, default=15, help="Permutations sampled per CE iteration.")
    ap.add_argument("--ce-elite-frac", type=float, default=0.2, help="Elite fraction for CE updates.")
    ap.add_argument("--ce-alpha", type=float, default=0.7, help="Smoothing for CE updates (0..1).")
    ap.add_argument("--ce-rank-temp", type=float, default=1.0, help="Rank-to-logit temperature (smaller => more aggressive).")

    ap.add_argument(
        "--final-draws",
        type=int,
        default=10,
        help=(
            "After CE finishes optimizing the permutation distribution, sample this many perms FROM THE FINAL distribution, "
            "evaluate on the final selection set, and pick best. If <=0, defaults to ce_iters*ce_batch."
        ),
    )

    ap.add_argument("--ce-mle-steps", type=int, default=60, help="Adam steps for PL MLE fit on elites (pl_ce_mle).")
    ap.add_argument("--ce-mle-lr", type=float, default=0.1, help="Adam learning rate for PL MLE fit on elites (pl_ce_mle).")
    ap.add_argument("--ce-mle-l2", type=float, default=0.0, help="L2 regularization strength for PL MLE fit (pl_ce_mle).")
    ap.add_argument("--ce-mle-clip", type=float, default=20.0, help="Clamp logits to [-clip, clip] for stability (pl_ce_mle).")
    ap.add_argument(
        "--ce-mle-weighted",
        action="store_true",
        help="If set, weight elite permutations by their validation accuracy during PL MLE fit (pl_ce_mle).",
    )

    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for scoring.")

    ap.add_argument(
        "--methods",
        nargs="*",
        default=["method_pl_ce", "pl_ce_mle", "mixture_pl_4"],
        help=(
            "PLR methods to run: method_pl_ce (EMA update), pl_ce_mle (MLE update), "
            "and mixture_pl_4. If you pass --methods with no values, no methods are computed."
        ),
    )

    ap.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars (keeps logging).")
    ap.add_argument("--print-demos", action="store_true", help="Print the sampled demo sentences and labels (can be long).")
    return ap.parse_args()


def main():
    args = parse_args()

    seeds = parse_seed_list(args.seed)
    methods = [b.lower() for b in (args.methods or [])]

    allowed_methods = {"method_pl_ce", "pl_ce_mle", "mixture_pl_4"}
    for b in methods:
        if b in allowed_methods:
            continue
        raise ValueError(
            f"Unknown method '{b}'. Supported: method_pl_ce, pl_ce_mle, mixture_pl_4"
        )

    show_tqdm = not args.no_tqdm

    final_draws = int(args.final_draws)
    if final_draws <= 0:
        final_draws = max(1, int(args.ce_iters) * int(args.ce_batch))

    results_dir = results_dir_from_args(args)
    results_path = os.path.join(results_dir, "subj.txt")

    seed_data, seed_order = read_results_file(results_path)

    method_key = make_pl_key("method_pl_ce", ce_iters=int(args.ce_iters), final_draws=int(final_draws))
    requested_keys = [method_to_result_key(b, args, final_draws) for b in methods]

    missing_by_seed: Dict[int, List[str]] = {}
    for sd in seeds:
        existing = seed_data.get(sd, OrderedDict())
        missing = [name for name in requested_keys if name not in existing]
        if missing:
            missing_by_seed[sd] = missing
            if sd not in seed_data:
                seed_data[sd] = OrderedDict()
                seed_order.append(sd)

    if not missing_by_seed:
        log(f"All requested results already present in {results_path}. Nothing to run.")
        return

    log("Loading datasets...")
    infer_items = load_subj_items(args.infer_path, show_tqdm=show_tqdm)
    test_items = load_subj_items(args.test_path, show_tqdm=show_tqdm)
    log(f"Loaded infer_items={len(infer_items)} (deduped), test_items={len(test_items)} (deduped)")

    log("Loading model/evaluator...")
    evaluator = LabelScoringEvaluator(args.model_path, batch_size=args.batch_size, show_tqdm=show_tqdm)
    log("Model loaded.")

    def _val_acc_on_outer(demos_list: List, perm: Tuple[int, ...], outer_items: List) -> float:
        if not outer_items:
            ordered = [demos_list[i] for i in perm]
            return eval_accuracy_for_order(evaluator, ordered, outer_items, desc="outer val (empty)")
        ordered = [demos_list[i] for i in perm]
        return eval_accuracy_for_order(evaluator, ordered, outer_items, desc="outer val")

    def _val_acc_on_inner(demos_list: List, perm: Tuple[int, ...], inner_items: List) -> float:
        ordered = [demos_list[i] for i in perm]
        return eval_accuracy_for_order(evaluator, ordered, inner_items, desc="inner val")

    def _val_acc_on_full(demos_list: List, perm: Tuple[int, ...], full_items: List) -> float:
        ordered = [demos_list[i] for i in perm]
        return eval_accuracy_for_order(evaluator, ordered, full_items, desc="full val (inner+outer)")

    for sd in seeds:
        if sd not in missing_by_seed:
            continue

        missing_set = set(missing_by_seed[sd])

        log("=======================================")
        log(f"RUN seed={sd} | requested_methods={methods} | missing={sorted(missing_set)}")
        log("=======================================")

        set_all_seeds(sd)

        if len(infer_items) < args.k:
            raise RuntimeError(f"Not enough infer items ({len(infer_items)}) to sample k={args.k} demos.")

        demo_rng = random.Random(sd + 999)
        demos = demo_rng.sample(infer_items, args.k)
        demo_keys = {d.key for d in demos}

        if args.print_demos:
            log("Demos (in sampled order):")
            for i, d in enumerate(demos, 1):
                print(f"  Demo{i}: label={d.label} | sentence={d.sentence}")

        val_pool = [x for x in infer_items if x.key not in demo_keys]
        if not val_pool:
            raise RuntimeError("Validation pool is empty after excluding demos.")
        log(f"Validation pool size after excluding demos: {len(val_pool)}")

        # Validation split: INNER is used for fitting/updates; OUTER is used for final selection (PL family).
        eval_sets_inner, inner_selection_set, outer_val_set = build_validation_eval_sets(
            val_pool,
            subset=args.subset,
            replay_size=args.replay_size,
            ce_iters=args.ce_iters,
            seed=sd,
            resample=bool(args.resample),
        )
        full_val_set = _dedup_by_key(inner_selection_set + outer_val_set)

        log(
            f"[VAL] inner_selection_set={len(inner_selection_set)} "
            f"outer_val_set={len(outer_val_set)} full_val_set={len(full_val_set)}"
        )

        # -------------------------
        # METHOD (CE + PL heuristic)
        # Fit/update on INNER; final selection on OUTER.
        # -------------------------
        if method_key in missing_set:
            log("Running METHOD: CE + Plackett–Luce (heuristic update) FIT on INNER; SELECT on OUTER...")
            method_info = ce_optimize_order_plackett_luce(
                demos,
                evaluator,
                eval_sets_inner,      # fit on INNER
                outer_val_set,        # select on OUTER
                ce_iters=args.ce_iters,
                ce_batch=args.ce_batch,
                elite_frac=args.ce_elite_frac,
                alpha=args.ce_alpha,
                rank_temp=args.ce_rank_temp,
                seed=sd + 2024000,
                final_draws=final_draws,
            )
            method_perm = method_info["best_perm"]
            method_outer_best = method_info["best_val_acc"]

            log("Evaluating METHOD permutation on INNER, FULL, and test set...")
            method_val_inner = _val_acc_on_inner(demos, method_perm, inner_selection_set)
            method_val_full = _val_acc_on_full(demos, method_perm, full_val_set)

            ordered_method = [demos[i] for i in method_perm]
            method_test_acc = eval_accuracy_for_order(evaluator, ordered_method, test_items, desc="method_pl_ce test")

            seed_data[sd][method_key] = format_result_line(
                method_key, method_test_acc, method_outer_best, method_perm,
                extra=(
                    f"inner_acc={method_val_inner:.4f} full_acc={method_val_full:.4f} "
                    f"inner_n={len(inner_selection_set)} outer_n={len(outer_val_set)} full_n={len(full_val_set)} "
                    f"ce_iters={int(args.ce_iters)} final_draws={int(final_draws)}"
                )
            )

        pl_mle_key = make_pl_key(
            "pl_ce_mle",
            ce_iters=int(args.ce_iters),
            final_draws=int(final_draws),
            lr=float(args.ce_mle_lr),
        )
        if pl_mle_key in missing_set:
            log("Running PL-CE-MLE: FIT on INNER; SELECT on OUTER...")
            info = ce_optimize_order_plackett_luce_mle(
                demos,
                evaluator,
                eval_sets_inner,      # fit on INNER
                outer_val_set,        # select on OUTER
                ce_iters=args.ce_iters,
                ce_batch=args.ce_batch,
                elite_frac=args.ce_elite_frac,
                alpha=args.ce_alpha,
                mle_steps=args.ce_mle_steps,
                mle_lr=args.ce_mle_lr,
                mle_l2=args.ce_mle_l2,
                mle_clip=args.ce_mle_clip,
                mle_weighted=bool(args.ce_mle_weighted),
                seed=sd + 3033000,
                final_draws=final_draws,
            )
            perm = info["best_perm"]
            best_outer = info["best_val_acc"]

            log("Evaluating PL-CE-MLE-selected permutation on INNER, FULL, and test...")
            best_inner = _val_acc_on_inner(demos, perm, inner_selection_set)
            best_full = _val_acc_on_full(demos, perm, full_val_set)
            ordered = [demos[i] for i in perm]
            test_acc = eval_accuracy_for_order(evaluator, ordered, test_items, desc="pl_ce_mle test")

            seed_data[sd][pl_mle_key] = format_result_line(
                pl_mle_key, test_acc, best_outer, perm,
                extra=(
                    f"inner_acc={best_inner:.4f} full_acc={best_full:.4f} "
                    f"inner_n={len(inner_selection_set)} outer_n={len(outer_val_set)} full_n={len(full_val_set)} "
                    f"ce_iters={int(args.ce_iters)} lr={float(args.ce_mle_lr)} final_draws={int(final_draws)}"
                )
            )

        for b in methods:
            mix_k = 4 if b == "mixture_pl_4" else None
            if mix_k is None:
                continue

            mix_key = make_pl_key(
                f"mixture_pl_{mix_k}",
                ce_iters=int(args.ce_iters),
                final_draws=int(final_draws),
                lr=float(args.ce_mle_lr),
            )
            if mix_key not in missing_set:
                continue

            log(f"Running MIXTURE-PL ({b}): FIT on INNER; SELECT on OUTER (K={mix_k})...")
            info = mixture_pl_best_perm(
                demos,
                evaluator,
                eval_sets_inner,      # fit on INNER
                outer_val_set,        # select on OUTER
                ce_iters=args.ce_iters,
                ce_batch=args.ce_batch,
                elite_frac=args.ce_elite_frac,
                alpha=args.ce_alpha,
                mix_components=int(mix_k),
                mle_steps=args.ce_mle_steps,
                mle_lr=args.ce_mle_lr,
                mle_l2=args.ce_mle_l2,
                mle_clip=args.ce_mle_clip,
                mle_weighted=bool(args.ce_mle_weighted),
                seed=sd + 4044000 + int(mix_k),
                final_draws=final_draws,
            )
            perm = info["best_perm"]
            best_outer = info["best_val_acc"]

            log(f"Evaluating {b}-selected permutation on INNER, FULL, and test...")
            best_inner = _val_acc_on_inner(demos, perm, inner_selection_set)
            best_full = _val_acc_on_full(demos, perm, full_val_set)
            ordered = [demos[i] for i in perm]
            test_acc = eval_accuracy_for_order(evaluator, ordered, test_items, desc=f"{b} test")

            seed_data[sd][mix_key] = format_result_line(
                mix_key, test_acc, best_outer, perm,
                extra=(
                    f"inner_acc={best_inner:.4f} full_acc={best_full:.4f} "
                    f"inner_n={len(inner_selection_set)} outer_n={len(outer_val_set)} full_n={len(full_val_set)} "
                    f"ce_iters={int(args.ce_iters)} lr={float(args.ce_mle_lr)} final_draws={int(final_draws)} K={int(mix_k)}"
                )
            )

        write_results_file(results_path, seed_data, seed_order)
        log(f"Wrote/updated {results_path} (seed={sd}).")


if __name__ == "__main__":
    main()