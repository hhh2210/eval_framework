#!/usr/bin/env bash
# =============================================================================
# batch_eval.sh — Multi-GPU parallel evaluation for RL training checkpoints
#
# Full pipeline:  Inference (vLLM) → Judge (API) → Plot training curves
#
# Usage:
#   1. Copy this file and edit the CONFIG section below.
#   2. bash examples/batch_eval.sh
#
# The script auto-schedules checkpoints across available GPUs in rounds,
# so you can evaluate more checkpoints than GPUs.
# =============================================================================
set -euo pipefail

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG — Edit defaults here, or override any field at invocation time: ║
# ║                                                                         ║
# ║  Common recipes:                                                        ║
# ║    PHASE=inference STEPS=600 bash examples/batch_eval.sh                ║
# ║    PHASE=judge     STEPS="120,240,360,480,600" bash ...                 ║
# ║    PHASE=plot      STEPS="120,240,360,480,600" bash ...                 ║
# ║    PHASE=all       SKIP_COMPLETE=1 bash ...  # resume where you left off║
# ║    GPU_IDS="2 3 5 7" bash ...                                           ║
# ║    DRY_RUN=1 bash ...                          # preview then exit      ║
# ║                                                                         ║
# ║  PHASE accepts: all | inference | judge | plot | ij | jp  (default: all)║
# ║  STEPS / GPU_IDS accept comma or space separated strings when passed    ║
# ║  via env, e.g. STEPS="120,240,360" or STEPS="600".                      ║
# ║  SKIP_COMPLETE=1 auto-skips steps whose artifacts already look complete ║
# ║  for the phase(s) about to run (see _step_is_complete).                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# -- Paths --
EVAL_FRAMEWORK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${EVAL_FRAMEWORK_ROOT}/.venv"
REPO="$(cd "${EVAL_FRAMEWORK_ROOT}/../.." && pwd)"

# Source .env early so credentials are available before overrides below.
if [ -f "${EVAL_FRAMEWORK_ROOT}/.env" ]; then
  source "${EVAL_FRAMEWORK_ROOT}/.env"
fi

# `: "${VAR:=default}"` only assigns when VAR is unset/empty, so anything
# exported in the environment (or .env above) takes precedence.

# Checkpoint directory: expects subdirs like global_step_120/actor_hf.
: "${CKPT_DIR:=${EVAL_FRAMEWORK_ROOT}/checkpoints}"

# Experiment label (used in plot legends & log dir)
: "${EXP_LABEL:=example_run}"

# Output directory: results go to ${OUT_DIR}/step_${STEP}/${TASK}/
: "${OUT_DIR:=${EVAL_FRAMEWORK_ROOT}/outputs/${EXP_LABEL}}"

# Plot output directory
: "${PLOT_DIR:=${OUT_DIR}/plots}"

# Steps to evaluate. Accepts env-var overrides as comma or space separated
# strings, e.g. STEPS="600" or STEPS="120,240,360". Leave empty to
# auto-detect all global_step_* directories under CKPT_DIR.
: "${STEPS:=}"

# Tasks — all supported: ifeval,ifbench,writingbench,healthbench,arena-hard,alpaca-eval
: "${INFERENCE_TASKS:=ifeval,ifbench,writingbench,healthbench,arena-hard,alpaca-eval}"
: "${JUDGE_TASKS:=writingbench,healthbench,arena-hard,alpaca-eval}"

# Number of samples per prompt for mean@N + error bar, PER TASK. Results go to
# ${OUT_DIR}/step_${STEP}/run_${k}/${task}/... and are later aggregated into
# ${OUT_DIR}/step_${STEP}/${task}/summary_agg.json.
# - Rule-based tasks (ifeval/ifbench): free to crank up, only costs GPU decode.
# - Rubric tasks (healthbench/writingbench): costs judge API N×, so pick carefully.
# - Pairwise tasks (arena-hard/alpaca-eval): they already report bootstrap CI
#   internally, mean@N adds little; keep at 1 unless you really need it.
# Set all to 1 to reproduce the old single-run behavior.
: "${N_SAMPLES_IFEVAL:=8}"
: "${N_SAMPLES_IFBENCH:=8}"
: "${N_SAMPLES_HEALTHBENCH:=8}"
: "${N_SAMPLES_WRITINGBENCH:=4}"
: "${N_SAMPLES_ARENA_HARD:=1}"
: "${N_SAMPLES_ALPACA_EVAL:=1}"

# Error bar kind passed to plot_training_curves.py: ci95 | sem | std | none
: "${ERRORBAR:=ci95}"

# GPU & vLLM settings. GPU_IDS is the exact list of devices to use, in
# scheduling order. When TP_SIZE>1, GPUs are consumed in chunks of TP_SIZE,
# so GPU_IDS="2 3 5 7" with TP_SIZE=2 yields slots {2,3} and {5,7}.
: "${GPU_IDS:=0}"
: "${TP_SIZE:=1}"
: "${GPU_MEM_UTIL:=0.95}"
: "${BASE_PORT:=30001}"
: "${INFERENCE_THREADS:=512}"

# Judge settings (Phase 2)
: "${JUDGE_MODEL:=${AGENT_MODEL:-qwen-plus}}"
: "${JUDGE_BASE_URL:=${AGENT_API_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
: "${JUDGE_API_KEY:=${AGENT_API_KEY:-}}"
: "${JUDGE_THREADS:=32}"
: "${JUDGE_BATCH_SIZE:=5}"

# Logging
: "${LOG_DIR:=/tmp/eval_logs/${EXP_LABEL}}"

# Phases to run (set to 0 to skip). If PHASE is set, it overrides these.
: "${RUN_INFERENCE:=1}"
: "${RUN_JUDGE:=1}"
: "${RUN_PLOT:=1}"

# PHASE is a convenience knob that rewrites RUN_INFERENCE/JUDGE/PLOT in one go.
# Valid values: all | inference | judge | plot | ij | jp
: "${PHASE:=all}"

# SKIP_COMPLETE=1 drops steps that already have all phase-relevant artifacts
# on disk (see _step_is_complete below). Great for resuming an interrupted run.
: "${SKIP_COMPLETE:=0}"

# Set DRY_RUN=1 to print the effective config and exit without running.
: "${DRY_RUN:=0}"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  END CONFIG — You should not need to edit below this line               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ---------------------------------------------------------------------------
# Normalize list-valued config (STEPS, GPU_IDS) into bash arrays.
# Accept either array literals (from in-file edits) or comma/space-separated
# strings (from env-var overrides).
# ---------------------------------------------------------------------------
_to_array() {
  # Usage: _to_array VAR_NAME  — parses $VAR_NAME as whitespace/comma list
  # and rebinds it to an array with the same name.
  local name=$1 raw
  raw="${!name}"
  # Normalize commas to spaces, then split on whitespace.
  raw="${raw//,/ }"
  read -ra _tmp <<< "$raw"
  eval "${name}=(\"\${_tmp[@]}\")"
}

# STEPS / GPU_IDS may arrive either as a bash array literal (when users edit
# this file directly, e.g. STEPS=(120 240)) or as a scalar string from the
# environment. Only normalize when the variable is NOT already an array, so
# array literals keep working.
_is_array() {
  [[ "$(declare -p "$1" 2>/dev/null)" == "declare -a"* ]]
}
_is_array STEPS   || _to_array STEPS
_is_array GPU_IDS || _to_array GPU_IDS

# ---------------------------------------------------------------------------
# Resolve PHASE → RUN_INFERENCE/RUN_JUDGE/RUN_PLOT
# PHASE takes precedence when explicitly set to anything other than "all".
# ---------------------------------------------------------------------------
case "${PHASE}" in
  all)                              ;;  # keep whatever RUN_* the user set
  inference|infer|i)                RUN_INFERENCE=1; RUN_JUDGE=0; RUN_PLOT=0 ;;
  judge|j)                          RUN_INFERENCE=0; RUN_JUDGE=1; RUN_PLOT=0 ;;
  plot|p)                           RUN_INFERENCE=0; RUN_JUDGE=0; RUN_PLOT=1 ;;
  ij|inference+judge|infer+judge)   RUN_INFERENCE=1; RUN_JUDGE=1; RUN_PLOT=0 ;;
  jp|judge+plot)                    RUN_INFERENCE=0; RUN_JUDGE=1; RUN_PLOT=1 ;;
  *)
    echo "ERROR: invalid PHASE='${PHASE}'. Allowed: all | inference | judge | plot | ij | jp" >&2
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
source "${VENV}/bin/activate"
cd "${REPO}"
mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${PLOT_DIR}"

# Auto-detect steps from checkpoint directory when STEPS is empty
if [ ${#STEPS[@]} -eq 0 ]; then
  echo "Auto-detecting steps from ${CKPT_DIR}..."
  for d in "${CKPT_DIR}"/global_step_*/; do
    [ -d "$d" ] || continue
    step=$(basename "$d" | sed 's/global_step_//')
    STEPS+=("$step")
  done
  IFS=$'\n' STEPS=($(sort -n <<<"${STEPS[*]}")); unset IFS
  echo "  Found ${#STEPS[@]} steps: ${STEPS[*]}"
fi

if [ ${#STEPS[@]} -eq 0 ]; then
  echo "ERROR: No steps found in ${CKPT_DIR}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Per-task N lookup + grouping. We accept per-task N so cheap benchmarks
# (ifeval/ifbench) can crank to 8 while judge-expensive ones stay at 1.
# ---------------------------------------------------------------------------
_n_for_task() {
  case "$1" in
    ifeval)       echo "${N_SAMPLES_IFEVAL}" ;;
    ifbench)      echo "${N_SAMPLES_IFBENCH}" ;;
    healthbench)  echo "${N_SAMPLES_HEALTHBENCH}" ;;
    writingbench) echo "${N_SAMPLES_WRITINGBENCH}" ;;
    arena-hard)   echo "${N_SAMPLES_ARENA_HARD}" ;;
    alpaca-eval)  echo "${N_SAMPLES_ALPACA_EVAL}" ;;
    *)            echo 1 ;;
  esac
}

# Groups tasks by their N value so we make as few eval-framework calls as
# possible while still running each task the right number of times.
# Input : "ifeval,ifbench,healthbench,arena-hard"
# Output: lines of the form "N:task1,task2", one per unique N.
_group_tasks_by_n() {
  local tasks_csv=$1 task n
  declare -A _g=()
  local IFS=','
  for task in ${tasks_csv}; do
    n=$(_n_for_task "$task")
    if [ -n "${_g[$n]:-}" ]; then
      _g[$n]+=","
    fi
    _g[$n]+="${task}"
  done
  for n in "${!_g[@]}"; do
    echo "${n}:${_g[$n]}"
  done
}

# ---------------------------------------------------------------------------
# Completion detectors. Markers match what each task writes on disk — keep in
# sync with tasks/*_task.py. Used by SKIP_COMPLETE to prune resumable runs.
# New layout: artifacts live under run_${k}/${task}/... for k in 0..N-1.
# ---------------------------------------------------------------------------
_response_artifact() {
  local task=$1 dir=$2
  case "${task}" in
    writingbench)           echo "${dir}/${task}/responses/responses.jsonl" ;;
    arena-hard|alpaca-eval) echo "${dir}/${task}/model_answer" ;;  # directory
    *)                      echo "${dir}/${task}/responses.jsonl" ;;
  esac
}

_inference_complete() {
  local step=$1 dir="${OUT_DIR}/step_${step}" t marker n k
  local IFS=','
  for t in ${INFERENCE_TASKS}; do
    n=$(_n_for_task "$t")
    for (( k=0; k<n; k++ )); do
      marker=$(_response_artifact "$t" "${dir}/run_${k}")
      [ -e "$marker" ] || return 1
    done
  done
  return 0
}

_judge_complete() {
  local step=$1 dir="${OUT_DIR}/step_${step}" t n k
  local IFS=','
  for t in ${JUDGE_TASKS}; do
    n=$(_n_for_task "$t")
    for (( k=0; k<n; k++ )); do
      [ -f "${dir}/run_${k}/${t}/summary.json" ] || return 1
    done
  done
  return 0
}

# Usage: _filter_steps predicate_fn label out_array_name step1 step2 ...
_filter_steps() {
  local pred=$1 label=$2 out_name=$3; shift 3
  local step keep=()
  for step in "$@"; do
    if "$pred" "$step"; then
      echo "  [skip ${label}] step_${step} (artifacts already present)"
    else
      keep+=("$step")
    fi
  done
  eval "${out_name}=(\"\${keep[@]}\")"
}

# ---------------------------------------------------------------------------
# Resolve per-phase step lists now so DRY_RUN can show exactly what will run.
# Defaults: each phase operates on the full STEPS list. With SKIP_COMPLETE=1
# we prune steps whose artifacts already exist for that specific phase.
# ---------------------------------------------------------------------------
declare -a PHASE1_STEPS=("${STEPS[@]}")
declare -a PHASE2_STEPS=("${STEPS[@]}")
if [ "${SKIP_COMPLETE}" = "1" ]; then
  [ "${RUN_INFERENCE}" -eq 1 ] && _filter_steps _inference_complete "infer" PHASE1_STEPS "${STEPS[@]}"
  [ "${RUN_JUDGE}"     -eq 1 ] && _filter_steps _judge_complete     "judge" PHASE2_STEPS "${STEPS[@]}"
fi

# ---------------------------------------------------------------------------
# Print effective config so the user sees exactly what will run.
# ---------------------------------------------------------------------------
_mask() { [ -n "$1" ] && echo "<set, ${#1} chars>" || echo "<empty>"; }
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Effective config                                               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
printf "  %-20s %s\n" "EXP_LABEL"        "${EXP_LABEL}"
printf "  %-20s %s\n" "CKPT_DIR"         "${CKPT_DIR}"
printf "  %-20s %s\n" "OUT_DIR"          "${OUT_DIR}"
printf "  %-20s %s\n" "PLOT_DIR"         "${PLOT_DIR}"
printf "  %-20s %s\n" "LOG_DIR"          "${LOG_DIR}"
printf "  %-20s %s (%d step(s))\n" "STEPS"  "${STEPS[*]}" "${#STEPS[@]}"
if [ "${SKIP_COMPLETE}" = "1" ]; then
  [ "${RUN_INFERENCE}" -eq 1 ] && \
    printf "  %-20s %s (%d step(s))\n" "  → phase1 inference" "${PHASE1_STEPS[*]:-<none>}" "${#PHASE1_STEPS[@]}"
  [ "${RUN_JUDGE}" -eq 1 ] && \
    printf "  %-20s %s (%d step(s))\n" "  → phase2 judge"     "${PHASE2_STEPS[*]:-<none>}" "${#PHASE2_STEPS[@]}"
fi
printf "  %-20s %s (%d gpu(s), TP=%s)\n" "GPU_IDS" "${GPU_IDS[*]}" "${#GPU_IDS[@]}" "${TP_SIZE}"
printf "  %-20s %s\n" "INFERENCE_TASKS"  "${INFERENCE_TASKS}"
printf "  %-20s %s\n" "JUDGE_TASKS"      "${JUDGE_TASKS}"
_nsamp_str=""
_IFS_BAK="${IFS-}"; IFS=','
for _t in ${INFERENCE_TASKS}; do _nsamp_str+="${_t}=$(_n_for_task "${_t}") "; done
IFS="${_IFS_BAK}"
printf "  %-20s %s\n" "N_SAMPLES"        "${_nsamp_str}"
printf "  %-20s %s\n" "ERRORBAR"         "${ERRORBAR}"
printf "  %-20s %s\n" "JUDGE_MODEL"      "${JUDGE_MODEL}"
printf "  %-20s %s\n" "JUDGE_BASE_URL"   "${JUDGE_BASE_URL}"
printf "  %-20s %s\n" "JUDGE_API_KEY"    "$(_mask "${JUDGE_API_KEY}")"
printf "  %-20s %s  (inference=%s judge=%s plot=%s)\n" "PHASE" "${PHASE}" "${RUN_INFERENCE}" "${RUN_JUDGE}" "${RUN_PLOT}"
printf "  %-20s %s\n" "SKIP_COMPLETE"    "${SKIP_COMPLETE}"
echo ""

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1 — exiting without executing any phase."
  exit 0
fi

# ---------------------------------------------------------------------------
# Helper: serve one checkpoint on one GPU, run eval, then kill vLLM
# ---------------------------------------------------------------------------
serve_and_eval() {
  local gpu_list=$1 port=$2 model_path=$3 name=$4 out_dir=$5 tasks=$6

  echo "[GPU ${gpu_list}] Serve ${name} → :${port} | tasks: ${tasks}"

  CUDA_VISIBLE_DEVICES=${gpu_list} vllm serve "${model_path}" \
    --served-model-name "${name}" \
    --host 0.0.0.0 --port "${port}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    > "${LOG_DIR}/vllm_${name}.log" 2>&1 &
  local pid=$!

  # Health check with timeout (5 min)
  local waited=0
  while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
    if ! kill -0 ${pid} 2>/dev/null; then
      echo "[GPU ${gpu_list}] FAIL: vllm died — see ${LOG_DIR}/vllm_${name}.log"
      return 1
    fi
    if (( waited >= 300 )); then
      echo "[GPU ${gpu_list}] FAIL: vllm startup timeout (${waited}s)"
      kill ${pid} 2>/dev/null || true
      return 1
    fi
    sleep 3; waited=$((waited + 3))
  done
  echo "[GPU ${gpu_list}] Ready (${waited}s). Evaluating..."

  # Group tasks by their configured N and loop each group N times against the
  # SAME live vllm server. Keeping the server alive across samples lets vLLM's
  # prefix cache fully amortize prefill, so wall clock ≈ decode(N)×, not N×
  # fresh runs.
  local group n grp_tasks k log_file
  while IFS= read -r group; do
    [ -z "${group}" ] && continue
    n="${group%%:*}"
    grp_tasks="${group#*:}"
    for (( k=0; k<n; k++ )); do
      log_file="${LOG_DIR}/eval_${name}_run${k}.log"
      echo "[GPU ${gpu_list}] ${name} run ${k}/${n} | tasks: ${grp_tasks}"
      eval-framework \
        --tasks "${grp_tasks}" \
        --model "${name}" \
        --base-url "http://localhost:${port}/v1" \
        --inference-only \
        --output-dir "${out_dir}/run_${k}" \
        --num-threads "${INFERENCE_THREADS}" \
        2>&1 | tee -a "${log_file}"
    done
  done < <(_group_tasks_by_n "${tasks}")

  echo "[GPU ${gpu_list}] Done: ${name}"

  # Kill the entire process group, then the pid as fallback
  kill -- -${pid} 2>/dev/null || kill ${pid} 2>/dev/null || true
  wait ${pid} 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Phase 1: Parallel inference (multi-round scheduling)
# ---------------------------------------------------------------------------
if [ "${RUN_INFERENCE}" -eq 1 ]; then
  if [ ${#PHASE1_STEPS[@]} -eq 0 ]; then
    echo "Phase 1: nothing to do (all ${#STEPS[@]} step(s) already have inference artifacts)."
  else

  num_gpus=${#GPU_IDS[@]}
  if (( num_gpus % TP_SIZE != 0 )); then
    echo "ERROR: ${num_gpus} GPUs in GPU_IDS not divisible by TP_SIZE=${TP_SIZE}" >&2
    exit 1
  fi
  slots=$((num_gpus / TP_SIZE))

  # Pre-compute slot -> comma-separated GPU id list
  declare -a SLOT_GPUS=()
  for (( s=0; s<slots; s++ )); do
    base=$((s * TP_SIZE))
    list=""
    for (( k=0; k<TP_SIZE; k++ )); do
      [ -n "${list}" ] && list+=","
      list+="${GPU_IDS[$((base + k))]}"
    done
    SLOT_GPUS+=("${list}")
  done

  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  Phase 1: Inference (${#PHASE1_STEPS[@]} checkpoints, ${num_gpus} GPUs, TP=${TP_SIZE}, slots=${slots})"
  echo "║  GPU_IDS: ${GPU_IDS[*]}  →  slots: ${SLOT_GPUS[*]}"
  echo "╚══════════════════════════════════════════════════════════════════╝"

  total=${#PHASE1_STEPS[@]}
  rounds=$(( (total + slots - 1) / slots ))

  for (( round=0; round<rounds; round++ )); do
    start_idx=$((round * slots))
    end_idx=$((start_idx + slots))
    (( end_idx > total )) && end_idx=${total}
    count=$((end_idx - start_idx))

    echo ""
    echo "── Round $((round+1))/${rounds}: steps ${PHASE1_STEPS[$start_idx]}..${PHASE1_STEPS[$((end_idx-1))]} (${count} jobs) ──"

    for (( j=0; j<count; j++ )); do
      idx=$((start_idx + j))
      step=${PHASE1_STEPS[$idx]}
      gpu_list=${SLOT_GPUS[$j]}
      port=$((BASE_PORT + j))
      name="s${step}"
      model_path="${CKPT_DIR}/global_step_${step}/actor_hf"
      out_dir="${OUT_DIR}/step_${step}"

      serve_and_eval "${gpu_list}" ${port} "${model_path}" "${name}" "${out_dir}" "${INFERENCE_TASKS}" &
    done

    wait
    echo "── Round $((round+1)) done. Cleaning up... ──"
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 3
  done

  echo ""
  echo "Phase 1 complete. Inference outputs: ${OUT_DIR}/"
  fi   # end: PHASE1_STEPS non-empty guard
fi

# ---------------------------------------------------------------------------
# Phase 2: Judge-only scoring (batched to respect API rate limits)
# ---------------------------------------------------------------------------
if [ "${RUN_JUDGE}" -eq 1 ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  Phase 2: Judge-only scoring (judge=${JUDGE_MODEL})             "
  echo "╚══════════════════════════════════════════════════════════════════╝"

  if [ -z "${JUDGE_API_KEY}" ]; then
    echo "WARNING: JUDGE_API_KEY is empty. Set it in .env or in the CONFIG section."
    echo "Skipping judge phase."
  else
    if [ ${#PHASE2_STEPS[@]} -eq 0 ]; then
      echo "Phase 2: nothing to do (all ${#STEPS[@]} step(s) already judged)."
    else
      judge_one() {
        local step=$1 name="s${1}"
        local dir="${OUT_DIR}/step_${step}"
        local group n grp_tasks k log_file
        while IFS= read -r group; do
          [ -z "${group}" ] && continue
          n="${group%%:*}"
          grp_tasks="${group#*:}"
          for (( k=0; k<n; k++ )); do
            log_file="${LOG_DIR}/judge_s${step}_run${k}.log"
            echo "[step_${step}] Judging run ${k}/${n} (${grp_tasks})..."
            eval-framework \
              --tasks "${grp_tasks}" \
              --model "${name}" \
              --judge-model "${JUDGE_MODEL}" \
              --judge-base-url "${JUDGE_BASE_URL}" \
              --judge-api-key "${JUDGE_API_KEY}" \
              --output-dir "${dir}/run_${k}" \
              --judge-only \
              --num-threads "${JUDGE_THREADS}" \
              > "${log_file}" 2> >(tee -a "${log_file}" >&2)
          done
        done < <(_group_tasks_by_n "${JUDGE_TASKS}")
        echo "[step_${step}] Judge done."
      }

      # Split into batches to avoid API rate limits
      batch_num=0
      job_count=0
      total_batches=$(( (${#PHASE2_STEPS[@]} + JUDGE_BATCH_SIZE - 1) / JUDGE_BATCH_SIZE ))

      for step in "${PHASE2_STEPS[@]}"; do
        if (( job_count % JUDGE_BATCH_SIZE == 0 )); then
          (( job_count > 0 )) && wait
          batch_num=$((batch_num + 1))
          echo ""
          echo "── Judge batch ${batch_num}/${total_batches} ──"
        fi
        judge_one "${step}" &
        job_count=$((job_count + 1))
      done
      wait

      echo ""
      echo "Phase 2 complete."
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Phase 3: Aggregate N runs → summary_agg.json, then plot training curves
# Aggregation is idempotent and cheap; we always run it before plotting so
# stale per-run directories don't silently show up without error bars.
# ---------------------------------------------------------------------------
if [ "${RUN_PLOT}" -eq 1 ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  Phase 3: Aggregation + Plotting                               "
  echo "╚══════════════════════════════════════════════════════════════════╝"

  steps_csv=$(IFS=,; echo "${STEPS[*]}")

  # Build per-task N spec for aggregate_runs.py
  all_tasks="ifeval,ifbench,healthbench,writingbench,arena-hard,alpaca-eval"
  n_spec=""
  _IFS_BAK="${IFS-}"; IFS=','
  for _t in ${all_tasks}; do
    [ -n "${n_spec}" ] && n_spec+=","
    n_spec+="${_t}=$(_n_for_task "${_t}")"
  done
  IFS="${_IFS_BAK}"

  echo "── Aggregating per-step runs (${n_spec}) ──"
  python "${EVAL_FRAMEWORK_ROOT}/tools/aggregate_runs.py" \
    --out-dir "${OUT_DIR}" \
    --steps "${steps_csv}" \
    --tasks "${all_tasks}" \
    --n-samples "${n_spec}"

  echo ""
  echo "── Plotting (errorbar=${ERRORBAR}) ──"
  python "${EVAL_FRAMEWORK_ROOT}/tools/plot_training_curves.py" \
    --runs "${EXP_LABEL}=${OUT_DIR}" \
    --name-pattern "${EXP_LABEL}=step_{step}" \
    --steps "${steps_csv}" \
    --tasks "${all_tasks}" \
    --plot-dir "${PLOT_DIR}" \
    --show-errorbar "${ERRORBAR}"

  echo ""
  echo "Plots saved to: ${PLOT_DIR}/"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  All done!                                                      "
echo "║  Outputs : ${OUT_DIR}/                               "
echo "║  Plots   : ${PLOT_DIR}/                              "
echo "║  Logs    : ${LOG_DIR}/                               "
echo "╚══════════════════════════════════════════════════════════════════╝"
