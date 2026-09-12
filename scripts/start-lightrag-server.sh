#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# start-lightrag-server.sh — semdoc 自带的 lightrag-server 启动脚本。
#
# 自足：不依赖 semrag 仓库的 .env / scripts。所有运行配置从 lightrag.toml
# 读取（与 Config.toml / schema.toml 解耦），密钥通过 *_env 间接引用环境变量，
# 本文件里永远不出现密钥明文。
#
# 用法：
#   ./scripts/start-lightrag-server.sh                 # 前台运行
#   ./scripts/start-lightrag-server.sh --bg            # 后台，日志 /tmp/lightrag-server.log
#   ./scripts/start-lightrag-server.sh --stop          # 停止
#   ./scripts/start-lightrag-server.sh --health        # 探活（0=健康）
#   ./scripts/start-lightrag-server.sh --config PATH   # 指定配置文件（默认 ./lightrag.toml）
#   ./scripts/start-lightrag-server.sh --key XYZ       # 临时启用 API Key（覆盖配置）
#
# lightrag-server 的每个 CLI 参数都有同名环境变量回退（--workspace→WORKSPACE、
# --key→LIGHTRAG_API_KEY、--llm-binding→LLM_BINDING …），因此本脚本只做
# "toml → env" 翻译，再以 python -m lightrag.api.lightrag_server 启动。
# 用模块方式启动是为了绕开 venv 入口脚本的 shebang——venv 可能是从旧路径
# 复制/迁移来的（shebang 指向不存在的解释器）。
#
# venv 探测顺序（第一个含 python 的生效）：
#   $LIGHTRAG_VENV > /workspace/LightRAG/.venv > /workspace/web/LightRAG/.venv > ./lightrag/.venv
# lightrag 若是源码树安装（venv site-packages 里没有 lightrag 包），脚本会
# 自动把源码目录加进 PYTHONPATH。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="lightrag.toml"
BG=0
STOP=0
HEALTH=0
KEY_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bg)     BG=1; shift ;;
    --stop)   STOP=1; shift ;;
    --health) HEALTH=1; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    --key)    KEY_OVERRIDE="$2"; shift 2 ;;
    --port)   PORT_OVERRIDE="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# --health / --stop：不依赖配置文件，直接操作进程/端口
# ---------------------------------------------------------------------------
PORT_NUM=9621

if [[ "$STOP" -eq 1 ]]; then
  pids=$(pgrep -f "lightrag.api.lightrag_server" || true)
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill $pids
    echo "[lightrag] stopped: $pids"
  else
    echo "[lightrag] no running lightrag-server found"
  fi
  exit 0
fi

if [[ "$HEALTH" -eq 1 ]]; then
  curl -sf -m 5 "http://127.0.0.1:${PORT_NUM}/health" > /dev/null
  echo "[lightrag] healthy"
  exit 0
fi

# ---------------------------------------------------------------------------
# TOML 扁平键值解析（bash 实现，配合 lightrag.toml 的语法约束）
#   输出：每行 "key=value"；无值行跳过。值支持双引号字符串/整数/布尔。
# ---------------------------------------------------------------------------
parse_toml_flat() {
  local file="$1"
  awk '
    /^[[:space:]]*#/ { next }                      # 注释行
    /^[[:space:]]*$/ { next }                      # 空行
    /^\[/            { next }                      # 不支持嵌套表（容忍并跳过）
    /^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=/ {
      line=$0
      sub(/^[^=]*=[[:space:]]*/, "", line)         # 取等号右侧
      sub(/#.*/, "", line)                         # 去尾注释
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
      gsub(/^"|"$/, "", line)                      # 去引号
      key=$0; sub(/[[:space:]]*=.*/, "", key)
      if (line != "") print key "=" line
    }
  ' "$file"
}

if [[ ! -f "$CONFIG" ]]; then
  echo "[lightrag] config not found: $CONFIG (see lightrag.toml in the project root)" >&2
  exit 1
fi

# 配置 → 关联数组
declare -A CFG
while IFS='=' read -r k v; do
  CFG["$k"]="$v"
done < <(parse_toml_flat "$CONFIG")

cfg() { echo "${CFG[$1]:-}"; }

# *_env 间接引用：读环境变量拿密钥。密钥不存在时该绑定留空
# （lightrag 对多数绑定允许匿名），并对必填项给出明确报错。
secret() {
  local env_name; env_name="$(cfg "$1")"
  if [[ -n "$env_name" ]]; then
    printenv "$env_name" || true
  fi
}

# ---------------------------------------------------------------------------
# venv 探测 + PYTHONPATH 修正
# ---------------------------------------------------------------------------
VENV=""
for cand in "${LIGHTRAG_VENV:-}" "/workspace/LightRAG/.venv" "/workspace/web/LightRAG/.venv" "$PROJECT_ROOT/lightrag/.venv"; do
  [[ -n "$cand" && -x "$cand/bin/python" ]] && { VENV="$cand"; break; }
done
if [[ -z "$VENV" ]]; then
  echo "[lightrag] no venv found — set LIGHTRAG_VENV or install one under /workspace/LightRAG/.venv" >&2
  exit 1
fi
PYTHON="$VENV/bin/python"

# lightrag 以源码树安装时（venv 里 import 不到），把源码目录挂进 PYTHONPATH
if ! "$PYTHON" -c "import lightrag" 2>/dev/null; then
  SRC=""
  for cand in "${LIGHTRAG_SRC:-}" "/workspace/LightRAG" "$(dirname "$VENV")"; do
    [[ -n "$cand" && -f "$cand/lightrag/__init__.py" ]] && { SRC="$cand"; break; }
  done
  if [[ -z "$SRC" ]]; then
    echo "[lightrag] cannot import lightrag and no source tree found (tried \$LIGHTRAG_SRC, /workspace/LightRAG)" >&2
    exit 1
  fi
  export PYTHONPATH="${SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi

# venv 附带的 server-only 依赖检查（lightrag-hku 不声明它们）
if ! "$PYTHON" -c "import jwt, aiofiles, python_multipart" 2>/dev/null; then
  echo "[lightrag] missing server deps (pyjwt/aiofiles/python-multipart) in $VENV" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 组装环境变量（lightrag-server 的 env 回退机制会拾取它们）
# ---------------------------------------------------------------------------
export LIGHTRAG_KV_STORAGE="$(cfg kv_storage)"
export LIGHTRAG_VECTOR_STORAGE="$(cfg vector_storage)"
export LIGHTRAG_GRAPH_STORAGE="$(cfg graph_storage)"
export LIGHTRAG_DOC_STATUS_STORAGE="$(cfg doc_status_storage)"

export WORKSPACE="$(cfg workspace)"

export POSTGRES_HOST="$(cfg postgres_host)"
export POSTGRES_PORT="$(cfg postgres_port)"
export POSTGRES_USER="$(cfg postgres_user)"
export POSTGRES_DATABASE="$(cfg postgres_database)"
PG_PW="$(secret postgres_password_env)"
if [[ -z "$PG_PW" ]]; then
  echo "[lightrag] postgres password: env \$(cfg postgres_password_env) not set" >&2
  exit 1
fi
export POSTGRES_PASSWORD="$PG_PW"
export POSTGRES_VECTOR_INDEX_TYPE="$(cfg postgres_vector_index_type)"
export POSTGRES_HNSW_M="$(cfg postgres_hnsw_m)"
export POSTGRES_HNSW_EF="$(cfg postgres_hnsw_ef)"

export NEO4J_URI="$(cfg neo4j_uri)"
export NEO4J_USERNAME="$(cfg neo4j_username)"
NEO_PW="$(secret neo4j_password_env)"
if [[ -z "$NEO_PW" ]]; then
  echo "[lightrag] neo4j password: env \$(cfg neo4j_password_env) not set" >&2
  exit 1
fi
export NEO4J_PASSWORD="$NEO_PW"
export NEO4J_DATABASE="$(cfg neo4j_database)"

export LLM_BINDING="$(cfg llm_binding)"
export LLM_BINDING_HOST="$(cfg llm_binding_host)"
export LLM_MODEL="$(cfg llm_model)"
LLM_KEY="$(secret llm_binding_api_key_env)"
[[ -n "$LLM_KEY" ]] && export LLM_BINDING_API_KEY="$LLM_KEY"

export EMBEDDING_BINDING="$(cfg embedding_binding)"
export EMBEDDING_BINDING_HOST="$(cfg embedding_binding_host)"
export EMBEDDING_MODEL="$(cfg embedding_model)"
export EMBEDDING_DIM="$(cfg embedding_dim)"
EMB_KEY="$(secret embedding_binding_api_key_env)"
[[ -n "$EMB_KEY" ]] && export EMBEDDING_BINDING_API_KEY="$EMB_KEY"

export RERANK_BINDING="$(cfg rerank_binding)"
export RERANK_MODEL="$(cfg rerank_model)"
export RERANK_BINDING_HOST="$(cfg rerank_binding_host)"
RERANK_KEY="$(secret rerank_binding_api_key_env)"
[[ -n "$RERANK_KEY" ]] && export RERANK_BINDING_API_KEY="$RERANK_KEY"

export MAX_PARALLEL_INSERT="$(cfg max_parallel_insert)"
export MAX_ASYNC="$(cfg max_async)"

HOST="$(cfg host)"
PORT="$(cfg port)"
[[ -n "${PORT_OVERRIDE:-}" ]] && PORT="$PORT_OVERRIDE"

# API Key：--key 覆盖 > toml 值 > 留空（不启用认证）
if [[ -n "$KEY_OVERRIDE" ]]; then
  export LIGHTRAG_API_KEY="$KEY_OVERRIDE"
else
  TOML_KEY="$(cfg lightrag_api_key)"
  [[ -n "$TOML_KEY" ]] && export LIGHTRAG_API_KEY="$TOML_KEY"
fi

# ---------------------------------------------------------------------------
# 端口占用预检 + 启动
# ---------------------------------------------------------------------------
if python3 - "$PORT" <<'PYEOF' 2>/dev/null; then
import socket, sys
s = socket.socket()
s.settimeout(1)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PYEOF
  echo "[lightrag] port $PORT already in use — run with --stop first" >&2
  exit 1
fi

LOG_LEVEL="$(cfg log_level)"
CMD=("$PYTHON" -m lightrag.api.lightrag_server --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL")

if [[ "$BG" -eq 1 ]]; then
  LOG=/tmp/lightrag-server.log
  echo "[lightrag] launching in background, log: $LOG"
  # shellcheck disable=SC2086
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo $! > /tmp/lightrag-server.pid
  echo "[lightrag] pid: $(cat /tmp/lightrag-server.pid)  webui: http://$HOST:$PORT/webui/"
  # 等待就绪（最多 30s；PG/Neo4j 连接 + 表检查需要几秒）
  for _ in $(seq 1 30); do
    if curl -sf -m 2 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
      echo "[lightrag] healthy"
      exit 0
    fi
    sleep 1
  done
  echo "[lightrag] WARNING: not healthy after 30s — check $LOG" >&2
  tail -10 "$LOG" >&2 || true
  exit 1
else
  exec "${CMD[@]}"
fi
