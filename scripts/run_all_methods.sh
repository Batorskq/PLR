#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_all_methods.sh TASK MODEL [extra python args...]

Runs:
  method_pl_ce
  pl_ce_mle
  mixture_pl_4

Example:
  scripts/run_all_methods.sh subj /path/to/model --k 16 --subset 1000 --seed 1,2,3,4,5
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

task="$1"
model="$2"
shift 2

for method in method_pl_ce pl_ce_mle mixture_pl_4; do
  echo "==> $task :: $method"
  "$ROOT_DIR/scripts/run_plr.sh" "$task" "$method" "$model" "$@"
done
