#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Edit the initialization below; trailing CLI arguments override Python options.
DATASET="bird_dev"
RUN_ID="run_2"

# export OPENAI_API_BASE="https://llm-80oiqtb4br9yy56x.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
# export OPENAI_API_KEY="EMPTY"
# MODEL_NAME="qwen3.6-35b-a3b"

export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="EMPTY"
# MODEL_NAME="Ministral-3-14B-Reasoning-2512"
MODEL_NAME="Qwen3.6-35B-A3B"
TEMPERATURE=0
MAX_TOKENS=16384
WORKERS=8
EMBEDDING_BASE_URL="http://127.0.0.1:9000/v1"
EMBEDDING_MODEL="bge-m3"
export CHESS_EMBEDDING_API_KEY="EMPTY"
DATA_PATH="../SAGE-SQL/datasets/bird/dev/dev.json"
DB_ROOT_DIRECTORY="../SAGE-SQL/datasets/bird/dev/dev_databases"
CONFIG="run/configs/CHESS_IR_SS_CG.yaml"

OUTPUT_DIR="outputs/chess_${DATASET}_${MODEL_NAME}_${RUN_ID}"

exec python3 -u src/main.py \
    --data_mode dev --data_path "$DATA_PATH" \
    --db_root_directory "$DB_ROOT_DIRECTORY" \
    --config "$CONFIG" --num_workers "$WORKERS" \
    --local_model "$MODEL_NAME" --base_url "$OPENAI_API_BASE" \
    --temperature "$TEMPERATURE" --max_tokens "$MAX_TOKENS" \
    --embedding_base_url "$EMBEDDING_BASE_URL" --embedding_model "$EMBEDDING_MODEL" \
    --output_dir "$OUTPUT_DIR" "$@"
