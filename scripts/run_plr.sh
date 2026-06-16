#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_plr.sh TASK METHOD MODEL [extra python args...]

Tasks:
  mr news sst5 trec subj gsm8k deepmath math500

Methods:
  method_pl_ce pl_ce_mle mixture_pl_4

Example:
  scripts/run_plr.sh mr method_pl_ce /path/to/model --k 16 --subset 1000 --seed 1,2,3,4,5
USAGE
}

if [[ $# -lt 3 ]]; then
  usage
  exit 1
fi

task="$1"
method="$2"
model="$3"
shift 3

case "$task" in
  mr|news|sst5|trec|subj|gsm8k|deepmath|math500)
    script="${task}.py"
    ;;
  *)
    echo "Unknown task: $task" >&2
    usage >&2
    exit 1
    ;;
esac

case "$method" in
  method_pl_ce|pl_ce_mle|mixture_pl_4)
    ;;
  *)
    echo "Unknown method: $method" >&2
    usage >&2
    exit 1
    ;;
esac

cd "$ROOT_DIR"
exec python "$script" --model-path "$model" --methods "$method" "$@"
