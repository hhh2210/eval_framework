#!/usr/bin/env bash
# =============================================================================
# shard_parallel_eval.sh — Evaluate ONE model across all benchmarks using N GPUs
#
# Serves the same model on N GPUs, shards each task's input data across them,
# runs inference in parallel, and merges results. Much faster than single-GPU
# for tasks with long outputs (writingbench: 8x speedup).
#
# Usage:
#   1. Edit the CONFIG section below.
#   2. bash examples/shard_parallel_eval.sh
#
# Pipeline:
#   Phase 1: Start N vLLM servers (same model on N GPUs)
#   Phase 2: For each task → shard data → N-way parallel inference → merge
#   Phase 3: Kill vLLM servers
#   Phase 4: Judge (API, no GPU needed)
# =============================================================================
set -euo pipefail

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

EVAL_FRAMEWORK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${EVAL_FRAMEWORK_ROOT}/.venv"
REPO="$(cd "${EVAL_FRAMEWORK_ROOT}/../.." && pwd)"

# -- Model --
MODEL_PATH="${MODEL_PATH:-${EVAL_FRAMEWORK_ROOT}/models/Qwen3-4B}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"

# -- Output --
OUT_DIR="${OUT_DIR:-${EVAL_FRAMEWORK_ROOT}/outputs/${MODEL_NAME}}"
PLOT_DIR="${PLOT_DIR:-${OUT_DIR}/plots}"

# -- Tasks --
# All 6 tasks, sharded across GPUs for parallel inference:
SHARD_TASKS="ifeval ifbench writingbench healthbench arena-hard alpaca-eval"
# Judge tasks (API-based, run after inference):
JUDGE_TASKS="writingbench,healthbench,arena-hard,alpaca-eval"

# -- GPU & vLLM --
GPUS_RAW="${GPUS:-0}"
IFS=', ' read -r -a GPUS <<< "${GPUS_RAW}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
BASE_PORT="${BASE_PORT:-30001}"
INFERENCE_THREADS="${INFERENCE_THREADS:-512}"

# -- Judge --
if [ -f "${EVAL_FRAMEWORK_ROOT}/.env" ]; then
  source "${EVAL_FRAMEWORK_ROOT}/.env"
fi
JUDGE_MODEL="${JUDGE_MODEL:-${AGENT_MODEL:-qwen-plus}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-${AGENT_API_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${AGENT_API_KEY:-}}"
JUDGE_THREADS="${JUDGE_THREADS:-32}"

# -- Temp & Logging --
TMP="/tmp/shard_eval_${MODEL_NAME}"
LOG_DIR="/tmp/eval_logs/${MODEL_NAME}"

# -- Phases to run --
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  END CONFIG                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

source "${VENV}/bin/activate"
cd "${REPO}"
mkdir -p "${TMP}" "${LOG_DIR}" "${OUT_DIR}" "${PLOT_DIR}"

NUM_GPUS=${#GPUS[@]}
SLOTS=$((NUM_GPUS / TP_SIZE))

# -- Task input file mapping --
# Arena-Hard data is bundled under tasks/arena_hard/data/; override ARENA_HARD_DIR to use a custom repo (e.g. newer bench version).
ARENA_HARD_DIR="${ARENA_HARD_DIR:-${EVAL_FRAMEWORK_ROOT}/tasks/arena_hard}"
ARENA_HARD_BENCH="${ARENA_HARD_BENCH:-arena-hard-v2.0}"

task_input_file() {
  local task=$1
  case "${task}" in
    ifeval)       echo "${EVAL_FRAMEWORK_ROOT}/tasks/ifeval/data/input_data.jsonl" ;;
    ifbench)      echo "${EVAL_FRAMEWORK_ROOT}/tasks/ifbench/data/IFBench_test.jsonl" ;;
    writingbench) echo "${EVAL_FRAMEWORK_ROOT}/tasks/writingbench/benchmark_query/benchmark_all.jsonl" ;;
    healthbench)  echo "${EVAL_FRAMEWORK_ROOT}/tasks/healthbench/data/healthbench_eval.jsonl" ;;
    arena-hard)   echo "${ARENA_HARD_DIR}/data/${ARENA_HARD_BENCH}/question.jsonl" ;;
    alpaca-eval)  echo "" ;;  # bundled JSON (not JSONL); handled below
  esac
}

# -- Task-specific input flag for eval-framework --
task_input_flag() {
  local task=$1
  case "${task}" in
    ifeval)       echo "--ifeval-input" ;;
    ifbench)      echo "--ifbench-input" ;;
    writingbench) echo "--writingbench-query" ;;
    healthbench)  echo "--healthbench-data" ;;
    arena-hard)   echo "--ifeval-input" ;;  # arena-hard ignores this flag; question sharding via --arena-hard-dir
    alpaca-eval)  echo "--alpaca-eval-data" ;;
  esac
}

# -- Locate the responses file produced by a single-task run --
# arena-hard and alpaca-eval write to model_answer/<model>.jsonl
task_responses_pattern() {
  local task=$1
  case "${task}" in
    writingbench)           echo "responses/responses.jsonl" ;;
    arena-hard|alpaca-eval) echo "model_answer/*.jsonl" ;;
    *)                      echo "responses.jsonl" ;;
  esac
}

# -- Merge destination (multi-task layout: out_dir/<task>/...) --
task_merge_dir() {
  local task=$1 out=$2
  echo "${out}/${task}"
}

rebuild_rule_task_outputs() {
  local task=$1
  local input_file=$2
  local output_dir=$3
  local log_file="${LOG_DIR}/rebuild_${task}.log"

  case "${task}" in
    ifeval)
      eval-framework \
        --tasks "${task}" \
        --model "${MODEL_NAME}_0" \
        --base-url "http://localhost:${BASE_PORT}/v1" \
        --output-dir "${output_dir}" \
        --ifeval-input "${input_file}" \
        --num-threads "${INFERENCE_THREADS}" \
        > "${log_file}" 2>&1
      ;;
    ifbench)
      eval-framework \
        --tasks "${task}" \
        --model "${MODEL_NAME}_0" \
        --base-url "http://localhost:${BASE_PORT}/v1" \
        --output-dir "${output_dir}" \
        --ifbench-dir "${EVAL_FRAMEWORK_ROOT}/.external/IFBench" \
        --ifbench-input "${input_file}" \
        --num-threads "${INFERENCE_THREADS}" \
        > "${log_file}" 2>&1
      ;;
  esac
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Start N vLLM servers
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Starting ${SLOTS} vLLM servers for ${MODEL_NAME}               "
echo "╚══════════════════════════════════════════════════════════════════╝"

VLLM_PIDS=()
for (( i=0; i<SLOTS; i++ )); do
  port=$((BASE_PORT + i))

  # Build CUDA_VISIBLE_DEVICES from GPUS array (supports TP_SIZE > 1)
  gpu_list=""
  for (( t=0; t<TP_SIZE; t++ )); do
    idx=$((i * TP_SIZE + t))
    [ -n "${gpu_list}" ] && gpu_list+=","
    gpu_list+="${GPUS[$idx]}"
  done

  CUDA_VISIBLE_DEVICES=${gpu_list} vllm serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}_${i}" \
    --host 0.0.0.0 --port "${port}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    > "${LOG_DIR}/vllm_gpu${i}.log" 2>&1 &
  VLLM_PIDS+=($!)
done

for (( i=0; i<SLOTS; i++ )); do
  port=$((BASE_PORT + i))
  waited=0
  while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
    if ! kill -0 ${VLLM_PIDS[$i]} 2>/dev/null; then
      echo "[GPU ${i}] FAIL: vllm died — see ${LOG_DIR}/vllm_gpu${i}.log"; exit 1
    fi
    if (( waited >= 300 )); then
      echo "[GPU ${i}] FAIL: timeout"; exit 1
    fi
    sleep 3; waited=$((waited + 3))
  done
  echo "[GPU ${i}] Ready (${waited}s) → :${port}"
done
echo "All ${SLOTS} servers ready."

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Inference
# ═══════════════════════════════════════════════════════════════════════════
if [ "${RUN_INFERENCE:-0}" -eq 1 ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  Phase 2: Sharded parallel inference                            "
  echo "╚══════════════════════════════════════════════════════════════════╝"

  for task in ${SHARD_TASKS}; do
    input_file=$(task_input_file "${task}")
    input_flag=$(task_input_flag "${task}")
    merge_dir=$(task_merge_dir "${task}" "${OUT_DIR}")
    mkdir -p "${merge_dir}"
    shard_dir="${TMP}/${task}"
    rm -rf "${shard_dir}"
    mkdir -p "${shard_dir}"

    # ── alpaca-eval: bundled data is JSON array; convert to JSONL for sharding ──
    if [ "${task}" = "alpaca-eval" ] && [ -z "${input_file}" ]; then
      bundled="${EVAL_FRAMEWORK_ROOT}/tasks/alpaca_eval/data/alpaca_eval_gpt4_baseline.json"
      if [ ! -f "${bundled}" ]; then
        echo "ERROR: alpaca-eval bundled data not found at ${bundled}" >&2; exit 1
      fi
      python3 -c "
import json
with open('${bundled}') as f: data = json.load(f)
with open('${shard_dir}/all_data.jsonl','w') as f:
  for r in data: f.write(json.dumps(r)+'\n')
print(f'  Converted {len(data)} prompts to JSONL')
"
      input_file="${shard_dir}/all_data.jsonl"
    fi

    # ── arena-hard: shard questions via per-shard arena dirs ──
    if [ "${task}" = "arena-hard" ]; then
      total=$(wc -l < "${input_file}")
      chunk=$(( (total + SLOTS - 1) / SLOTS ))
      echo ""
      echo "── ${task}: ${total} items → ${SLOTS} shards of ~${chunk} ──"
      split -l "${chunk}" -d -a 1 "${input_file}" "${shard_dir}/shard_"

      for (( i=0; i<SLOTS; i++ )); do
        shard="${shard_dir}/shard_${i}"
        [ -f "${shard}" ] || continue
        arena_tmp="${shard_dir}/arena_${i}/data/${ARENA_HARD_BENCH}"
        mkdir -p "${arena_tmp}"
        cp "${shard}" "${arena_tmp}/question.jsonl"
      done
    else
      total=$(wc -l < "${input_file}")
      chunk=$(( (total + SLOTS - 1) / SLOTS ))
      echo ""
      echo "── ${task}: ${total} items → ${SLOTS} shards of ~${chunk} ──"
      split -l "${chunk}" -d -a 1 "${input_file}" "${shard_dir}/shard_"
    fi

    EVAL_PIDS=()
    for (( i=0; i<SLOTS; i++ )); do
      port=$((BASE_PORT + i))
      shard="${shard_dir}/shard_${i}"
      [ -f "${shard}" ] || continue
      out="${shard_dir}/out_${i}"
      mkdir -p "${out}"

      extra_flags=""
      if [ "${task}" = "arena-hard" ]; then
        extra_flags="--arena-hard-dir ${shard_dir}/arena_${i}"
      else
        extra_flags="${input_flag} ${shard}"
      fi

      eval-framework \
        --tasks "${task}" \
        --model "${MODEL_NAME}_${i}" \
        --base-url "http://localhost:${port}/v1" \
        --inference-only \
        --output-dir "${out}" \
        ${extra_flags} \
        --num-threads "${INFERENCE_THREADS}" \
        > "${LOG_DIR}/eval_${task}_gpu${i}.log" 2>&1 &
      EVAL_PIDS+=($!)
    done
    for pid in "${EVAL_PIDS[@]}"; do
      if ! wait "${pid}"; then
        echo "ERROR: ${task} shard worker failed. See ${LOG_DIR}/eval_${task}_gpu*.log" >&2
        exit 1
      fi
    done

    # ── Merge results ──
    if [ "${task}" = "arena-hard" ] || [ "${task}" = "alpaca-eval" ]; then
      # Pairwise tasks: merge model_answer/*.jsonl into one file
      mkdir -p "${merge_dir}/model_answer"
      rm -f "${merge_dir}/model_answer/${MODEL_NAME}_0.jsonl"
      merged_count=0
      for (( i=0; i<SLOTS; i++ )); do
        src=$(find "${shard_dir}/out_${i}/model_answer" -maxdepth 1 -name "*.jsonl" 2>/dev/null | head -1 || true)
        if [ -n "${src}" ] && [ -f "${src}" ]; then
          n=$(wc -l < "${src}")
          cat "${src}" >> "${merge_dir}/model_answer/${MODEL_NAME}_0.jsonl"
          merged_count=$((merged_count + n))
        fi
      done
      echo "  ✓ Merged: ${merged_count}/${total} answers → ${merge_dir}/model_answer/"
    else
      # Standard tasks: merge responses.jsonl
      resp_pattern=$(task_responses_pattern "${task}")
      merged_file="${merge_dir}/${resp_pattern}"
      mkdir -p "$(dirname "${merged_file}")"
      rm -f "${merged_file}"
      merged_count=0
      for (( i=0; i<SLOTS; i++ )); do
        src="${shard_dir}/out_${i}/${resp_pattern}"
        if [ -f "${src}" ]; then
          n=$(wc -l < "${src}")
          cat "${src}" >> "${merged_file}"
          merged_count=$((merged_count + n))
        fi
      done
      echo "  ✓ Merged: ${merged_count}/${total} responses → ${merge_dir}/"

      if [ "${task}" = "ifeval" ] || [ "${task}" = "ifbench" ]; then
        rebuild_rule_task_outputs "${task}" "${input_file}" "${merge_dir}"
        echo "  ✓ Rebuilt derived metrics → ${merge_dir}/summary.json"
      fi
    fi
  done

  echo ""
  echo "Phase 2 complete."
fi

# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Kill vLLM servers
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "Shutting down vLLM servers..."
for pid in "${VLLM_PIDS[@]}"; do
  kill -- -${pid} 2>/dev/null || kill ${pid} 2>/dev/null || true
done
pkill -f "vllm serve" 2>/dev/null || true
sleep 3

# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Judge (API-based, no GPU)
# ═══════════════════════════════════════════════════════════════════════════
if [ "${RUN_JUDGE:-0}" -eq 1 ]; then
  if [ -z "${JUDGE_API_KEY:-}" ]; then
    echo "WARNING: RUN_JUDGE=1 but JUDGE_API_KEY/AGENT_API_KEY is empty."
    echo "Skipping judge phase."
  else
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  Phase 4: Judge-only scoring (${JUDGE_MODEL})                   "
    echo "╚══════════════════════════════════════════════════════════════════╝"

    eval-framework \
      --tasks "${JUDGE_TASKS}" \
      --model "${MODEL_NAME}_0" \
      --judge-model "${JUDGE_MODEL}" \
      --judge-base-url "${JUDGE_BASE_URL}" \
      --judge-api-key "${JUDGE_API_KEY}" \
      --output-dir "${OUT_DIR}" \
      --judge-only \
      --num-threads "${JUDGE_THREADS}" \
      2>&1 | tee "${LOG_DIR}/judge.log"

    echo "Judge complete."
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════════
rm -rf "${TMP}"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  All done!                                                      "
echo "║  Results : ${OUT_DIR}/                                "
echo "║  Logs    : ${LOG_DIR}/                                "
echo "╚══════════════════════════════════════════════════════════════════╝"
