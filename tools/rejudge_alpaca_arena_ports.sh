#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# if ! command -v eval-framework >/dev/null 2>&1; then
#   echo "ERROR: 'eval-framework' not found. Install first: pip install -e ."
#   exit 1
# fi

MODEL_NAME="${MODEL_NAME:-qwen3-4B}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/qwen3-4B}"
PORTS="${PORTS:-30001 30002 30003 30004}"
JUDGE_MODELS="${JUDGE_MODELS:-qwen-flash qwen-plus}"
NUM_THREADS="${NUM_THREADS:-32}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8000/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"

if [[ -z "${JUDGE_API_KEY}" ]]; then
  echo "ERROR: set JUDGE_API_KEY first."
  exit 1
fi

MODEL_FILE="${MODEL_NAME//\//_}.jsonl"
TS="$(date +%Y%m%d_%H%M%S)"

for judge_model in ${JUDGE_MODELS}; do
  OUT_ROOT="outputs/${MODEL_NAME}-judge-${judge_model}-${TS}"
  for port in ${PORTS}; do
    echo "===== ${judge_model} | port-${port} | alpaca-eval ====="
    ALPACA_ANS_DIR="${SOURCE_ROOT}/port-${port}/alpaca-eval/model_answer"
    ALPACA_JDG_DIR="${OUT_ROOT}/port-${port}/alpaca-eval/model_judgment"
    if [[ -f "${ALPACA_ANS_DIR}/${MODEL_FILE}" ]]; then
      eval-framework \
        --task alpaca-eval \
        --model "${MODEL_NAME}" \
        --judge-model "${judge_model}" \
        --judge-base-url "${JUDGE_BASE_URL}" \
        --judge-api-key "${JUDGE_API_KEY}" \
        --judge-only \
        --num-threads "${NUM_THREADS}" \
        --output-dir "${OUT_ROOT}/port-${port}/alpaca-eval" \
        --alpaca-eval-answers-dir "${ALPACA_ANS_DIR}" \
        --alpaca-eval-judgments-dir "${ALPACA_JDG_DIR}"
    else
      echo "SKIP alpaca-eval: missing ${ALPACA_ANS_DIR}/${MODEL_FILE}"
    fi

    echo "===== ${judge_model} | port-${port} | arena-hard ====="
    ARENA_ANS_DIR="${SOURCE_ROOT}/port-${port}/arena-hard/model_answer"
    ARENA_JDG_DIR="${OUT_ROOT}/port-${port}/arena-hard/model_judgment/${judge_model}"
    if [[ -f "${ARENA_ANS_DIR}/${MODEL_FILE}" ]]; then
      eval-framework \
        --task arena-hard \
        --model "${MODEL_NAME}" \
        --judge-model "${judge_model}" \
        --arena-hard-judge-name "${judge_model}" \
        --judge-base-url "${JUDGE_BASE_URL}" \
        --judge-api-key "${JUDGE_API_KEY}" \
        --judge-only \
        --num-threads "${NUM_THREADS}" \
        --output-dir "${OUT_ROOT}/port-${port}/arena-hard" \
        --arena-hard-answers-dir "${ARENA_ANS_DIR}" \
        --arena-hard-judgments-dir "${ARENA_JDG_DIR}"
    else
      echo "SKIP arena-hard: missing ${ARENA_ANS_DIR}/${MODEL_FILE}"
    fi
  done
  echo "DONE ${judge_model}: ${OUT_ROOT}"
done
