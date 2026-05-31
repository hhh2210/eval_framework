#!/usr/bin/env bash
# =============================================================================
# compare_two_runs.sh — Evaluate and compare two training runs (e.g. biased vs unbiased)
#
# This script evaluates checkpoints from two separate runs on the same set of
# benchmarks, then plots them side-by-side for visual comparison.
#
# Usage:
#   1. Edit the CONFIG section below.
#   2. bash examples/compare_two_runs.sh
# =============================================================================
set -euo pipefail

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

EVAL_FRAMEWORK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${EVAL_FRAMEWORK_ROOT}/.venv"
REPO="$(cd "${EVAL_FRAMEWORK_ROOT}/../.." && pwd)"

# -- Run A --
RUN_A_LABEL="${RUN_A_LABEL:-run_a}"
RUN_A_CKPT="${RUN_A_CKPT:-${EVAL_FRAMEWORK_ROOT}/checkpoints/run_a}"
RUN_A_OUT="${RUN_A_OUT:-${EVAL_FRAMEWORK_ROOT}/outputs/run_a}"
RUN_A_STEPS_RAW="${RUN_A_STEPS:-120,240,360}"
IFS=', ' read -r -a RUN_A_STEPS <<< "${RUN_A_STEPS_RAW}"

# -- Run B --
RUN_B_LABEL="${RUN_B_LABEL:-run_b}"
RUN_B_CKPT="${RUN_B_CKPT:-${EVAL_FRAMEWORK_ROOT}/checkpoints/run_b}"
RUN_B_OUT="${RUN_B_OUT:-${EVAL_FRAMEWORK_ROOT}/outputs/run_b}"
RUN_B_STEPS_RAW="${RUN_B_STEPS:-120,240,360}"
IFS=', ' read -r -a RUN_B_STEPS <<< "${RUN_B_STEPS_RAW}"

PLOT_DIR="${PLOT_DIR:-${EVAL_FRAMEWORK_ROOT}/outputs/comparison_plots}"

# -- Tasks --
INFERENCE_TASKS="ifeval,ifbench,writingbench,healthbench,arena-hard,alpaca-eval"
JUDGE_TASKS="writingbench,healthbench,arena-hard,alpaca-eval"

# -- GPU --
NUM_GPUS="${NUM_GPUS:-1}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
BASE_PORT="${BASE_PORT:-30001}"
INFERENCE_THREADS="${INFERENCE_THREADS:-512}"

# -- Judge --
if [ -f "${EVAL_FRAMEWORK_ROOT}/.env" ]; then
  source "${EVAL_FRAMEWORK_ROOT}/.env"
fi
JUDGE_MODEL="${AGENT_MODEL:-qwen-plus}"
JUDGE_BASE_URL="${AGENT_API_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
JUDGE_API_KEY="${AGENT_API_KEY:-}"
JUDGE_THREADS="${JUDGE_THREADS:-32}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-5}"

LOG_DIR="${LOG_DIR:-/tmp/eval_logs/compare}"

# -- Phases --
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
RUN_PLOT="${RUN_PLOT:-1}"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  END CONFIG                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

source "${VENV}/bin/activate"
cd "${REPO}"
mkdir -p "${LOG_DIR}" "${PLOT_DIR}"

# Merge both runs into a single job list: (label step ckpt_dir out_dir)
declare -a JOBS=()
for step in "${RUN_A_STEPS[@]}"; do
  JOBS+=("${RUN_A_LABEL} ${step} ${RUN_A_CKPT} ${RUN_A_OUT}")
done
for step in "${RUN_B_STEPS[@]}"; do
  JOBS+=("${RUN_B_LABEL} ${step} ${RUN_B_CKPT} ${RUN_B_OUT}")
done

serve_and_eval() {
  local gpu=$1 port=$2 model_path=$3 name=$4 out_dir=$5 tasks=$6

  echo "[GPU ${gpu}] Serve ${name} → :${port}"
  CUDA_VISIBLE_DEVICES=${gpu} vllm serve "${model_path}" \
    --served-model-name "${name}" \
    --host 0.0.0.0 --port "${port}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    > "${LOG_DIR}/vllm_${name}.log" 2>&1 &
  local pid=$!

  local waited=0
  while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
    if ! kill -0 ${pid} 2>/dev/null; then
      echo "[GPU ${gpu}] FAIL: vllm died — see ${LOG_DIR}/vllm_${name}.log"; return 1
    fi
    if (( waited >= 300 )); then
      echo "[GPU ${gpu}] FAIL: timeout"; kill ${pid} 2>/dev/null || true; return 1
    fi
    sleep 3; waited=$((waited + 3))
  done
  echo "[GPU ${gpu}] Ready (${waited}s). Evaluating..."

  eval-framework \
    --tasks "${tasks}" \
    --model "${name}" \
    --base-url "http://localhost:${port}/v1" \
    --inference-only \
    --output-dir "${out_dir}" \
    --num-threads "${INFERENCE_THREADS}" \
    2>&1 | tee "${LOG_DIR}/eval_${name}.log"

  echo "[GPU ${gpu}] Done: ${name}"
  kill -- -${pid} 2>/dev/null || kill ${pid} 2>/dev/null || true
  wait ${pid} 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Phase 1: Inference
# ---------------------------------------------------------------------------
if [ "${RUN_INFERENCE}" -eq 1 ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  Phase 1: Inference (${#JOBS[@]} checkpoints, ${NUM_GPUS} GPUs)                   "
  echo "╚══════════════════════════════════════════════════════════════════╝"

  slots=$((NUM_GPUS / TP_SIZE))
  total=${#JOBS[@]}
  rounds=$(( (total + slots - 1) / slots ))

  for (( round=0; round<rounds; round++ )); do
    start=$((round * slots))
    end=$((start + slots))
    (( end > total )) && end=${total}

    echo ""
    echo "── Round $((round+1))/${rounds} ──"

    for (( j=0; j<end-start; j++ )); do
      idx=$((start + j))
      read -r label step ckpt_dir out_base <<< "${JOBS[$idx]}"
      gpu=$((j * TP_SIZE))
      port=$((BASE_PORT + j))
      name="${label:0:2}_s${step}"
      model_path="${ckpt_dir}/global_step_${step}/actor_hf"
      out_dir="${out_base}/step_${step}"

      serve_and_eval ${gpu} ${port} "${model_path}" "${name}" "${out_dir}" "${INFERENCE_TASKS}" &
    done

    wait
    echo "── Round $((round+1)) cleanup ──"
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 3
  done

  echo "Phase 1 complete."
fi

# ---------------------------------------------------------------------------
# Phase 2: Judge
# ---------------------------------------------------------------------------
if [ "${RUN_JUDGE}" -eq 1 ] && [ -n "${JUDGE_API_KEY}" ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  Phase 2: Judge-only (judge=${JUDGE_MODEL})                     "
  echo "╚══════════════════════════════════════════════════════════════════╝"

  judge_one() {
    local label=$1 step=$2 out_base=$3
    local name="${label:0:2}_s${step}"
    local dir="${out_base}/step_${step}"
    echo "[${label}/step_${step}] Judging..."
    eval-framework \
      --tasks "${JUDGE_TASKS}" \
      --model "${name}" \
      --judge-model "${JUDGE_MODEL}" \
      --judge-base-url "${JUDGE_BASE_URL}" \
      --judge-api-key "${JUDGE_API_KEY}" \
      --output-dir "${dir}" \
      --judge-only \
      --num-threads "${JUDGE_THREADS}" \
      > "${LOG_DIR}/judge_${name}.log" 2> >(tee -a "${LOG_DIR}/judge_${name}.log" >&2)
    echo "[${label}/step_${step}] Done."
  }

  job_count=0
  batch_num=0
  total_batches=$(( (${#JOBS[@]} + JUDGE_BATCH_SIZE - 1) / JUDGE_BATCH_SIZE ))

  for job in "${JOBS[@]}"; do
    read -r label step ckpt_dir out_base <<< "${job}"
    if (( job_count % JUDGE_BATCH_SIZE == 0 )); then
      (( job_count > 0 )) && wait
      batch_num=$((batch_num + 1))
      echo "── Judge batch ${batch_num}/${total_batches} ──"
    fi
    judge_one "${label}" "${step}" "${out_base}" &
    job_count=$((job_count + 1))
  done
  wait

  echo "Phase 2 complete."
fi

# ---------------------------------------------------------------------------
# Phase 3: Plot comparison
# ---------------------------------------------------------------------------
if [ "${RUN_PLOT}" -eq 1 ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  Phase 3: Plotting comparison                                   "
  echo "╚══════════════════════════════════════════════════════════════════╝"

  all_steps=()
  for s in "${RUN_A_STEPS[@]}" "${RUN_B_STEPS[@]}"; do all_steps+=("$s"); done
  IFS=$'\n' all_steps=($(printf '%s\n' "${all_steps[@]}" | sort -un)); unset IFS
  steps_csv=$(IFS=,; echo "${all_steps[*]}")

  python "${EVAL_FRAMEWORK_ROOT}/tools/plot_training_curves.py" \
    --runs "${RUN_A_LABEL}=${RUN_A_OUT}" \
    --runs "${RUN_B_LABEL}=${RUN_B_OUT}" \
    --name-pattern "${RUN_A_LABEL}=step_{step}" \
    --name-pattern "${RUN_B_LABEL}=step_{step}" \
    --steps "${steps_csv}" \
    --tasks "ifeval,ifbench,healthbench,writingbench,arena-hard,alpaca-eval" \
    --plot-dir "${PLOT_DIR}"

  echo "Plots saved to: ${PLOT_DIR}/"
fi

echo ""
echo "All done!"
