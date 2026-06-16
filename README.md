# PLR

**Official implementation of [_PLR: Plackett-Luce for Reordering In-Context Learning Examples_](https://arxiv.org/abs/2603.21373).**

PLR learns a distribution over in-context example orderings, samples candidate permutations, and selects the ordering that performs best on a held-out validation split.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export MODEL=/path/to/instruct-model
./scripts/run_plr.sh mr method_pl_ce "$MODEL" --k 16 --subset 1000 --seed 1,2,3,4,5
```

All required JSONL files for MR, NEWS, SST5, TREC, SUBJ, GSM8K, DeepMath, and Math500 are already under `dataset/`; there is no download step.

## Methods

Only the PLR methods below are exposed:

| Method | Update | Use |
| --- | --- | --- |
| `method_pl_ce` | EMA-style cross-entropy update | Fast PL update from elite permutations |
| `pl_ce_mle` | Maximum-likelihood update | Direct PL fit on elite permutations |
| `mixture_pl_4` | Mixture of 4 PL distributions | Multi-component order distribution |

Run one method:

```bash
./scripts/run_plr.sh subj pl_ce_mle "$MODEL" --k 16 --subset 1000 --seed 1,2,3,4,5
```

Run all three methods for one task:

```bash
./scripts/run_all_methods.sh trec "$MODEL" --k 16 --subset 1000 --seed 1,2,3,4,5
```

## Task Scripts

| Task | Entry point |
| --- | --- |
| MR | `mr.py` |
| NEWS | `news.py` |
| SST5 | `sst5.py` |
| TREC | `trec.py` |
| SUBJ | `subj.py` |
| GSM8K | `gsm8k.py` |
| DeepMath | `deepmath.py` |
| Math500 | `math500.py` |

The helper scripts are thin wrappers around these entry points. They validate the task and method names, then pass any remaining flags through to the Python script.

## Direct Commands

Classification tasks share the same CLI. Replace `mr.py` with `news.py`, `sst5.py`, `subj.py`, or `trec.py`.

EMA update:

```bash
python mr.py \
  --model-path "$MODEL" \
  --methods method_pl_ce \
  --k 16 \
  --subset 1000 \
  --seed 1,2,3,4,5 \
  --ce-iters 15 \
  --ce-batch 15 \
  --ce-alpha 0.7 \
  --final-draws 10
```

MLE update:

```bash
python mr.py \
  --model-path "$MODEL" \
  --methods pl_ce_mle \
  --k 16 \
  --subset 1000 \
  --seed 1,2,3,4,5 \
  --ce-iters 15 \
  --ce-batch 15 \
  --ce-mle-steps 60 \
  --ce-mle-lr 0.1 \
  --final-draws 10
```

Mixture of 4 Plackett-Luce distributions:

```bash
python mr.py \
  --model-path "$MODEL" \
  --methods mixture_pl_4 \
  --k 16 \
  --subset 1000 \
  --seed 1,2,3,4,5 \
  --ce-iters 15 \
  --ce-batch 15 \
  --ce-mle-steps 60 \
  --ce-mle-lr 0.1 \
  --final-draws 10
```

Generation tasks use the same method names with shorter optimizer flags:

```bash
python gsm8k.py --model-path "$MODEL" --methods method_pl_ce --k 16 --subset 1000
python deepmath.py --model-path "$MODEL" --methods pl_ce_mle --k 16 --subset 1000 --mle-steps 50 --mle-lr 0.5
python math500.py --model-path "$MODEL" --methods mixture_pl_4 --k 16 --subset 1000 --mle-steps 50 --mle-lr 0.5
```

## Generated Files

Result files are created locally and ignored by git:

| Script family | Output path |
| --- | --- |
| MR, NEWS, SST5, TREC, SUBJ | `results_*` |
| GSM8K, DeepMath, Math500 | `results/` |

Use `make clean` to remove generated outputs and Python caches.

## Checks

```bash
make check
```

This compiles all Python entry points and utility modules.
