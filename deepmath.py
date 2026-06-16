# deepmath.py
import argparse
import json
import os
import random
from typing import List, Dict, Any
from collections import defaultdict

from utils.deepmath_utils import (
    log,
    set_all_seeds,
    load_deepmath_items,
    DeepMathGenerationEvaluator,
    build_validation_eval_sets,
    eval_accuracy_for_order,
    ce_optimize_order_plackett_luce,
    ce_optimize_order_plackett_luce_mle,
    mixture_pl_best_perm,
    K_DEFAULT,
)


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _append_jsonl(path: str, obj: dict) -> None:
    _ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _run_key(args: argparse.Namespace, seed: int) -> str:
    return (
        f"model={args.model_path}__k={args.k}__subset={args.subset}__seed={seed}"
        f"__ce={args.ce_iters}__final={args.final_draws}__bs={args.batch_size}__new={args.max_new_tokens}"
        f"__attn={args.attn_impl or 'auto'}__resample={int(args.resample)}__replay={args.replay_size}"
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--model-path", type=str, default="/gpfs/project/lox20cow/ms-swift/llama8binstruct")
    p.add_argument("--infer-path", type=str, default="dataset/deepmath_infer.jsonl")
    p.add_argument("--test-path", type=str, default="dataset/deepmath_test.jsonl")
    p.add_argument("--results-path", type=str, default="results/deepmath_runs.jsonl")

    p.add_argument("--k", type=int, default=K_DEFAULT)
    p.add_argument("--subset", type=str, default="2000")
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--attn-impl", type=str, default=None, choices=[None, "sdpa", "eager", "flash_attention_2"])

    p.add_argument(
        "--methods",
        nargs="*",
        default=["method_pl_ce", "pl_ce_mle", "mixture_pl_4"],
        help="PLR methods: method_pl_ce, pl_ce_mle, mixture_pl_4.",
    )

    p.add_argument("--ce-iters", type=int, default=10)
    p.add_argument("--ce-batch", type=int, default=10)
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--rank-temp", type=float, default=1.0)
    p.add_argument("--final-draws", type=int, default=10)

    p.add_argument("--replay-size", type=int, default=10)
    p.add_argument("--resample", action="store_true")

    # MLE / mixture settings
    p.add_argument("--mle-steps", type=int, default=50)
    p.add_argument("--mle-lr", type=float, default=0.5)
    p.add_argument("--mle-l2", type=float, default=0.0)
    p.add_argument("--mle-clip", type=float, default=20.0)
    p.add_argument("--mle-weighted", action="store_true")

    args = p.parse_args()

    allowed_methods = {"method_pl_ce", "pl_ce_mle", "mixture_pl_4"}
    args.methods = [b.lower() for b in (args.methods or [])]
    unknown_methods = [b for b in args.methods if b not in allowed_methods]
    if unknown_methods:
        raise ValueError(
            f"Unknown method(s): {unknown_methods}. "
            f"Supported: {sorted(allowed_methods)}"
        )

    seed = int(args.seed)
    set_all_seeds(seed)

    log("Loading datasets...")
    infer_items = load_deepmath_items(args.infer_path, show_tqdm=True)
    test_items = load_deepmath_items(args.test_path, show_tqdm=True)
    log(f"Loaded infer_items={len(infer_items)} (deduped), test_items={len(test_items)} (deduped)")

    if args.k <= 0 or args.k > len(infer_items):
        raise ValueError(f"--k must be in [1, {len(infer_items)}], got {args.k}")

    rng = random.Random(seed + 1337)
    demos = rng.sample(infer_items, args.k)
    demo_keys = {d.key for d in demos}
    val_pool = [x for x in infer_items if x.key not in demo_keys]
    log(f"Validation pool size after excluding demos: {len(val_pool)}")

    eval_sets_inner, inner_selection_set, outer_val_set = build_validation_eval_sets(
        val_pool,
        subset=args.subset,
        replay_size=args.replay_size,
        ce_iters=args.ce_iters,
        seed=seed,
        resample=bool(args.resample),
    )
    full_val_set = inner_selection_set + outer_val_set

    log(f"[VAL] inner_selection_set={len(inner_selection_set)} outer_val_set={len(outer_val_set)} full_val_set={len(full_val_set)}")

    log("Loading model/evaluator...")
    evaluator = DeepMathGenerationEvaluator(
        args.model_path,
        batch_size=int(args.batch_size),
        max_new_tokens=int(args.max_new_tokens),
        show_tqdm=True,
        attn_implementation=args.attn_impl,
    )
    log("Model loaded.")

    run_key = _run_key(args, seed)
    requested = list(args.methods)

    existing = _read_jsonl(args.results_path)
    done = defaultdict(set)
    for row in existing:
        rk = row.get("run_key")
        b = row.get("method")
        if rk and b:
            done[rk].add(b)

    missing = [b for b in requested if b not in done.get(run_key, set())]

    log("=======================================")
    log(f"RUN seed={seed} | requested_methods={requested} | missing={missing}")
    log("=======================================")

    def save_result(method: str, payload: Dict[str, Any]):
        row = {"run_key": run_key, "method": method, **payload}
        _append_jsonl(args.results_path, row)

    # ------------------------------------------------------------
    # METHOD: CE + PL heuristic update
    # ------------------------------------------------------------
    if "method_pl_ce" in missing:
        log("Running METHOD: CE + Plackett–Luce (heuristic update) FIT on INNER; SELECT on OUTER...")
        info = ce_optimize_order_plackett_luce(
            demos,
            evaluator,
            eval_sets_inner,
            outer_val_set,
            ce_iters=int(args.ce_iters),
            ce_batch=int(args.ce_batch),
            elite_frac=float(args.elite_frac),
            alpha=float(args.alpha),
            rank_temp=float(args.rank_temp),
            seed=seed,
            final_draws=int(args.final_draws),
        )
        best_perm = info["best_perm"]
        best_demos = [demos[i] for i in best_perm]
        test_acc = eval_accuracy_for_order(evaluator, best_demos, test_items, desc="method_pl_ce best -> test")

        payload = {
            "k": args.k,
            "subset": args.subset,
            "seed": seed,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "ce_iters": args.ce_iters,
            "ce_batch": args.ce_batch,
            "elite_frac": args.elite_frac,
            "alpha": args.alpha,
            "rank_temp": args.rank_temp,
            "final_draws": args.final_draws,
            "val_acc_outer": info["best_val_acc"],
            "test_acc": test_acc,
            "best_perm": list(best_perm),
            "history": info.get("history", []),
        }
        save_result("method_pl_ce", payload)
        log(f"METHOD done. val_outer={info['best_val_acc']:.4f} test={test_acc:.4f}")

    # ------------------------------------------------------------
    # PL-CE-MLE method
    # ------------------------------------------------------------
    if "pl_ce_mle" in missing:
        log("Running PL-CE-MLE FIT on INNER; SELECT on OUTER...")
        info = ce_optimize_order_plackett_luce_mle(
            demos,
            evaluator,
            eval_sets_inner,
            outer_val_set,
            ce_iters=int(args.ce_iters),
            ce_batch=int(args.ce_batch),
            elite_frac=float(args.elite_frac),
            alpha=float(args.alpha),
            mle_steps=int(args.mle_steps),
            mle_lr=float(args.mle_lr),
            mle_l2=float(args.mle_l2),
            mle_clip=float(args.mle_clip),
            mle_weighted=bool(args.mle_weighted),
            seed=seed,
            final_draws=int(args.final_draws),
        )
        best_perm = info["best_perm"]
        best_demos = [demos[i] for i in best_perm]
        test_acc = eval_accuracy_for_order(evaluator, best_demos, test_items, desc="pl_ce_mle best -> test")

        payload = {
            "k": args.k,
            "subset": args.subset,
            "seed": seed,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "ce_iters": args.ce_iters,
            "ce_batch": args.ce_batch,
            "elite_frac": args.elite_frac,
            "alpha": args.alpha,
            "mle_steps": args.mle_steps,
            "mle_lr": args.mle_lr,
            "mle_l2": args.mle_l2,
            "mle_clip": args.mle_clip,
            "mle_weighted": bool(args.mle_weighted),
            "final_draws": args.final_draws,
            "val_acc_outer": info["best_val_acc"],
            "test_acc": test_acc,
            "best_perm": list(best_perm),
            "history": info.get("history", []),
        }
        save_result("pl_ce_mle", payload)
        log(f"PL-CE-MLE done. val_outer={info['best_val_acc']:.4f} test={test_acc:.4f}")

    # ------------------------------------------------------------
    # Mixture-PL (K=4) method
    # ------------------------------------------------------------
    if "mixture_pl_4" in missing:
        log("Running Mixture-PL (K=4) FIT on INNER; SELECT on OUTER...")
        info = mixture_pl_best_perm(
            demos,
            evaluator,
            eval_sets_inner,
            outer_val_set,
            ce_iters=int(args.ce_iters),
            ce_batch=int(args.ce_batch),
            elite_frac=float(args.elite_frac),
            alpha=float(args.alpha),
            mix_components=4,
            mle_steps=int(args.mle_steps),
            mle_lr=float(args.mle_lr),
            mle_l2=float(args.mle_l2),
            mle_clip=float(args.mle_clip),
            mle_weighted=bool(args.mle_weighted),
            seed=seed,
            final_draws=int(args.final_draws),
        )
        best_perm = info["best_perm"]
        best_demos = [demos[i] for i in best_perm]
        test_acc = eval_accuracy_for_order(evaluator, best_demos, test_items, desc="mixture_pl_4 best -> test")

        payload = {
            "k": args.k,
            "subset": args.subset,
            "seed": seed,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "ce_iters": args.ce_iters,
            "ce_batch": args.ce_batch,
            "elite_frac": args.elite_frac,
            "alpha": args.alpha,
            "mix_components": 4,
            "mle_steps": args.mle_steps,
            "mle_lr": args.mle_lr,
            "mle_l2": args.mle_l2,
            "mle_clip": args.mle_clip,
            "mle_weighted": bool(args.mle_weighted),
            "final_draws": args.final_draws,
            "val_acc_outer": info["best_val_acc"],
            "test_acc": test_acc,
            "best_perm": list(best_perm),
            "history": info.get("history", []),
            "final_mix_weights": info.get("final_mix_weights"),
        }
        save_result("mixture_pl_4", payload)
        log(f"MIXTURE-PL-4 done. val_outer={info['best_val_acc']:.4f} test={test_acc:.4f}")

    log("All done.")


if __name__ == "__main__":
    main()
