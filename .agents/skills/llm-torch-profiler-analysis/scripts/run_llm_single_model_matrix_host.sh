#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_llm_single_model_matrix_host.sh \
    --model-id gpt_oss_20b \
    --model openai/gpt-oss-20b \
    --root /data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix \
    --gpus 2,3,4,5 \
    --sglang-port 30098 \
    --vllm-formal-port 31098 \
    --vllm-mapping-port 31099 \
    --trt-formal-prefill-port 32098 \
    --trt-formal-decode-port 32099 \
    --trt-mapping-prefill-port 32198 \
    --trt-mapping-decode-port 32199

This script is intended to run on the H100 host. It:
1. captures SGLang live profiling and writes `analysis_sglang.txt`
2. captures vLLM formal + eager mapping traces and writes `analysis_vllm.txt`
3. captures TensorRT-LLM formal + graph-off mapping traces and writes `analysis_trtllm.txt`
4. stores one benchmark JSON per framework under the model run directory

Default profiler workloads are stage-separated:
  prefill: input 4090, output 1
  decode:  input 1, output 2048

Environment:
  Export `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` before running.
EOF
}

MODEL_ID=""
MODEL=""
ROOT=""
GPUS=""
TP_SIZE=""
SGLANG_PORT=""
VLLM_FORMAL_PORT=""
VLLM_MAPPING_PORT=""
TRT_FORMAL_PREFILL_PORT=""
TRT_FORMAL_DECODE_PORT=""
TRT_MAPPING_PREFILL_PORT=""
TRT_MAPPING_DECODE_PORT=""
SGLANG_MEM_FRACTION="0.85"
MAX_MODEL_LEN="4096"
KV_FRACTION="0.85"
SGLANG_SERVER_EXTRA=""
PROFILE_WORKLOAD="both"
PREFILL_INPUT_LEN=4090
PREFILL_OUTPUT_LEN=1
DECODE_INPUT_LEN=1
DECODE_OUTPUT_LEN=2048
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRT_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:latest"
TRT_OVERRIDE_ROOT="/data/bbuf/validate/unified_llm_profiler_skill/overrides/trtllm"
TRT_OVERRIDE_SOURCE="$TRT_OVERRIDE_ROOT/py_executor.original.py"
TRT_OVERRIDE_PATH="$TRT_OVERRIDE_ROOT/py_executor_with_stack.py"

require_option_value() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "Missing value for $1." >&2
    usage >&2
    exit 2
  fi
}

validate_port() {
  if [[ ! "$2" =~ ^[0-9]+$ ]] || (( 10#$2 < 1 || 10#$2 > 65535 )); then
    echo "$1 must be an integer between 1 and 65535: $2" >&2
    exit 2
  fi
}

validate_fraction() {
  if [[ ! "$2" =~ ^(0([.][0-9]+)?|1([.][0-9]+)?)$ ]]; then
    echo "$1 must be a number between 0 and 1: $2" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id) require_option_value "$@"; MODEL_ID="$2"; shift 2 ;;
    --model) require_option_value "$@"; MODEL="$2"; shift 2 ;;
    --root) require_option_value "$@"; ROOT="$2"; shift 2 ;;
    --gpus) require_option_value "$@"; GPUS="$2"; shift 2 ;;
    --tp-size) require_option_value "$@"; TP_SIZE="$2"; shift 2 ;;
    --sglang-port) require_option_value "$@"; SGLANG_PORT="$2"; shift 2 ;;
    --vllm-formal-port) require_option_value "$@"; VLLM_FORMAL_PORT="$2"; shift 2 ;;
    --vllm-mapping-port) require_option_value "$@"; VLLM_MAPPING_PORT="$2"; shift 2 ;;
    --trt-formal-prefill-port) require_option_value "$@"; TRT_FORMAL_PREFILL_PORT="$2"; shift 2 ;;
    --trt-formal-decode-port) require_option_value "$@"; TRT_FORMAL_DECODE_PORT="$2"; shift 2 ;;
    --trt-mapping-prefill-port) require_option_value "$@"; TRT_MAPPING_PREFILL_PORT="$2"; shift 2 ;;
    --trt-mapping-decode-port) require_option_value "$@"; TRT_MAPPING_DECODE_PORT="$2"; shift 2 ;;
    --sglang-mem-fraction) require_option_value "$@"; SGLANG_MEM_FRACTION="$2"; shift 2 ;;
    --sglang-server-extra) require_option_value "$@"; SGLANG_SERVER_EXTRA="$2"; shift 2 ;;
    --max-model-len) require_option_value "$@"; MAX_MODEL_LEN="$2"; shift 2 ;;
    --kv-fraction) require_option_value "$@"; KV_FRACTION="$2"; shift 2 ;;
    --profile-workload) require_option_value "$@"; PROFILE_WORKLOAD="$2"; shift 2 ;;
    --prefill-input-len) require_option_value "$@"; PREFILL_INPUT_LEN="$2"; shift 2 ;;
    --prefill-output-len) require_option_value "$@"; PREFILL_OUTPUT_LEN="$2"; shift 2 ;;
    --decode-input-len) require_option_value "$@"; DECODE_INPUT_LEN="$2"; shift 2 ;;
    --decode-output-len) require_option_value "$@"; DECODE_OUTPUT_LEN="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before running." >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
export HF_TOKEN HUGGINGFACE_HUB_TOKEN

for value in \
  MODEL_ID MODEL ROOT GPUS \
  SGLANG_PORT VLLM_FORMAL_PORT VLLM_MAPPING_PORT \
  TRT_FORMAL_PREFILL_PORT TRT_FORMAL_DECODE_PORT \
  TRT_MAPPING_PREFILL_PORT TRT_MAPPING_DECODE_PORT; do
  if [[ -z "${!value}" ]]; then
    echo "Missing required argument: $value" >&2
    usage >&2
    exit 2
  fi
done

validate_port --sglang-port "$SGLANG_PORT"
validate_port --vllm-formal-port "$VLLM_FORMAL_PORT"
validate_port --vllm-mapping-port "$VLLM_MAPPING_PORT"
validate_port --trt-formal-prefill-port "$TRT_FORMAL_PREFILL_PORT"
validate_port --trt-formal-decode-port "$TRT_FORMAL_DECODE_PORT"
validate_port --trt-mapping-prefill-port "$TRT_MAPPING_PREFILL_PORT"
validate_port --trt-mapping-decode-port "$TRT_MAPPING_DECODE_PORT"
validate_fraction --sglang-mem-fraction "$SGLANG_MEM_FRACTION"
validate_fraction --kv-fraction "$KV_FRACTION"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
GPU_COUNT="${#GPU_LIST[@]}"
if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "Could not parse --gpus: $GPUS" >&2
  exit 2
fi
if [[ -z "$TP_SIZE" ]]; then
  TP_SIZE="$GPU_COUNT"
fi
if [[ ! "$TP_SIZE" =~ ^[0-9]+$ ]] || (( TP_SIZE < 1 || TP_SIZE > GPU_COUNT )); then
  echo "--tp-size must be between 1 and the visible GPU count ($GPU_COUNT)." >&2
  exit 2
fi

MODEL_ROOT="$ROOT/$MODEL_ID"
SGLANG_ANALYSIS="$MODEL_ROOT/analysis_sglang.txt"
VLLM_FORMAL_DIR="$MODEL_ROOT/vllm_formal"
VLLM_MAPPING_DIR="$MODEL_ROOT/vllm_mapping"
VLLM_ANALYSIS="$MODEL_ROOT/analysis_vllm.txt"
TRT_FORMAL_DIR="$MODEL_ROOT/trtllm_formal"
TRT_MAPPING_DIR="$MODEL_ROOT/trtllm_mapping"
TRT_ANALYSIS="$MODEL_ROOT/analysis_trtllm.txt"

docker exec -e MODEL_ROOT="$MODEL_ROOT" sglang_bbuf \
  bash -lc 'mkdir -p "$MODEL_ROOT"'

if [[ ! -s "$TRT_OVERRIDE_SOURCE" ]]; then
  echo "[bootstrap] TensorRT-LLM py_executor source snapshot"
  docker exec -e TRT_OVERRIDE_ROOT="$TRT_OVERRIDE_ROOT" sglang_bbuf \
    bash -lc 'mkdir -p "$TRT_OVERRIDE_ROOT"'
  docker run --rm --entrypoint cat "$TRT_IMAGE" \
    /usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/pyexecutor/py_executor.py \
    | docker exec -i -e TRT_OVERRIDE_SOURCE="$TRT_OVERRIDE_SOURCE" sglang_bbuf \
        bash -lc 'cat > "$TRT_OVERRIDE_SOURCE"'
fi
echo "[bootstrap] TensorRT-LLM py_executor override with with_stack=True and rank0-only trace export"
docker exec \
  -e SCRIPT_DIR="$SCRIPT_DIR" \
  -e TRT_OVERRIDE_SOURCE="$TRT_OVERRIDE_SOURCE" \
  -e TRT_OVERRIDE_PATH="$TRT_OVERRIDE_PATH" \
  sglang_bbuf bash -lc '
    cd "$SCRIPT_DIR"
    python3 make_trtllm_py_executor_override.py \
      --source "$TRT_OVERRIDE_SOURCE" \
      --output "$TRT_OVERRIDE_PATH"
  '

sglang_args=(
  --model "$MODEL"
  --run-dir "$MODEL_ROOT"
  --port "$SGLANG_PORT"
  --gpus "$GPUS"
  --tp-size "$TP_SIZE"
  --mem-fraction "$SGLANG_MEM_FRACTION"
  --profile-workload "$PROFILE_WORKLOAD"
  --prefill-input-len "$PREFILL_INPUT_LEN"
  --prefill-output-len "$PREFILL_OUTPUT_LEN"
  --decode-input-len "$DECODE_INPUT_LEN"
  --decode-output-len "$DECODE_OUTPUT_LEN"
  --trust-remote-code
)
if [[ -n "$SGLANG_SERVER_EXTRA" ]]; then
  sglang_args+=(--server-extra "$SGLANG_SERVER_EXTRA")
fi

echo "[1/6] SGLang server + live triage"
HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  "$SCRIPT_DIR/run_sglang_torch_profile_host.sh" \
  "${sglang_args[@]}"

echo "[2/6] vLLM formal"
HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  "$SCRIPT_DIR/run_vllm_torch_profile_host.sh" \
  --model "$MODEL" \
  --run-dir "$VLLM_FORMAL_DIR" \
  --port "$VLLM_FORMAL_PORT" \
  --gpus "$GPUS" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --profile-workload "$PROFILE_WORKLOAD" \
  --prefill-input-len "$PREFILL_INPUT_LEN" \
  --prefill-output-len "$PREFILL_OUTPUT_LEN" \
  --decode-input-len "$DECODE_INPUT_LEN" \
  --decode-output-len "$DECODE_OUTPUT_LEN" \
  --trust-remote-code

echo "[3/6] vLLM mapping"
HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  "$SCRIPT_DIR/run_vllm_torch_profile_host.sh" \
  --model "$MODEL" \
  --run-dir "$VLLM_MAPPING_DIR" \
  --port "$VLLM_MAPPING_PORT" \
  --gpus "$GPUS" \
  --tensor-parallel-size "$TP_SIZE" \
  --profiler-active-iterations 2 \
  --max-model-len "$MAX_MODEL_LEN" \
  --profile-workload "$PROFILE_WORKLOAD" \
  --prefill-input-len "$PREFILL_INPUT_LEN" \
  --prefill-output-len "$PREFILL_OUTPUT_LEN" \
  --decode-input-len "$DECODE_INPUT_LEN" \
  --decode-output-len "$DECODE_OUTPUT_LEN" \
  --trust-remote-code \
  --enforce-eager

echo "[4/6] vLLM mapping-formal analysis"
docker exec \
  -e SCRIPT_DIR="$SCRIPT_DIR" \
  -e VLLM_MAPPING_DIR="$VLLM_MAPPING_DIR" \
  -e VLLM_FORMAL_DIR="$VLLM_FORMAL_DIR" \
  -e VLLM_ANALYSIS="$VLLM_ANALYSIS" \
  sglang_bbuf bash -lc '
    cd "$SCRIPT_DIR"
    python3 analyze_llm_torch_profile.py \
      --framework vllm \
      --mapping-input "$VLLM_MAPPING_DIR" \
      --formal-input "$VLLM_FORMAL_DIR" \
      >"$VLLM_ANALYSIS"
  '

echo "[5/6] TensorRT-LLM formal + mapping captures"
HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  "$SCRIPT_DIR/run_trtllm_pytorch_profile_host.sh" \
  --model "$MODEL" \
  --run-dir "$TRT_FORMAL_DIR" \
  --stage prefill \
  --port "$TRT_FORMAL_PREFILL_PORT" \
  --gpus "$GPUS" \
  --tp-size "$TP_SIZE" \
  --kv-fraction "$KV_FRACTION" \
  --input-len "$PREFILL_INPUT_LEN" \
  --output-len "$PREFILL_OUTPUT_LEN" \
  --override-py-executor "$TRT_OVERRIDE_PATH" \
  --trust-remote-code
HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  "$SCRIPT_DIR/run_trtllm_pytorch_profile_host.sh" \
  --model "$MODEL" \
  --run-dir "$TRT_FORMAL_DIR" \
  --stage decode \
  --port "$TRT_FORMAL_DECODE_PORT" \
  --gpus "$GPUS" \
  --tp-size "$TP_SIZE" \
  --kv-fraction "$KV_FRACTION" \
  --input-len "$DECODE_INPUT_LEN" \
  --output-len "$DECODE_OUTPUT_LEN" \
  --override-py-executor "$TRT_OVERRIDE_PATH" \
  --trust-remote-code
HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  "$SCRIPT_DIR/run_trtllm_pytorch_profile_host.sh" \
  --model "$MODEL" \
  --run-dir "$TRT_MAPPING_DIR" \
  --stage prefill \
  --port "$TRT_MAPPING_PREFILL_PORT" \
  --gpus "$GPUS" \
  --tp-size "$TP_SIZE" \
  --kv-fraction "$KV_FRACTION" \
  --input-len "$PREFILL_INPUT_LEN" \
  --output-len "$PREFILL_OUTPUT_LEN" \
  --override-py-executor "$TRT_OVERRIDE_PATH" \
  --disable-cudagraph \
  --trust-remote-code
HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  "$SCRIPT_DIR/run_trtllm_pytorch_profile_host.sh" \
  --model "$MODEL" \
  --run-dir "$TRT_MAPPING_DIR" \
  --stage decode \
  --port "$TRT_MAPPING_DECODE_PORT" \
  --gpus "$GPUS" \
  --tp-size "$TP_SIZE" \
  --kv-fraction "$KV_FRACTION" \
  --input-len "$DECODE_INPUT_LEN" \
  --output-len "$DECODE_OUTPUT_LEN" \
  --override-py-executor "$TRT_OVERRIDE_PATH" \
  --disable-cudagraph \
  --trust-remote-code

echo "[6/6] TensorRT-LLM mapping-formal analysis"
docker exec \
  -e SCRIPT_DIR="$SCRIPT_DIR" \
  -e TRT_MAPPING_DIR="$TRT_MAPPING_DIR" \
  -e TRT_FORMAL_DIR="$TRT_FORMAL_DIR" \
  -e TRT_ANALYSIS="$TRT_ANALYSIS" \
  sglang_bbuf bash -lc '
    cd "$SCRIPT_DIR"
    python3 analyze_llm_torch_profile.py \
      --framework trtllm \
      --mapping-input "$TRT_MAPPING_DIR" \
      --formal-input "$TRT_FORMAL_DIR" \
      >"$TRT_ANALYSIS"
  '

echo "MODEL_ROOT=$MODEL_ROOT"
echo "ANALYSIS_SGLANG=$SGLANG_ANALYSIS"
echo "ANALYSIS_VLLM=$VLLM_ANALYSIS"
echo "ANALYSIS_TRTLLM=$TRT_ANALYSIS"
