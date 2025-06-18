#!/bin/bash

# Collective Intelligence Telemetry Integration
# Auto-generated on 2025-06-18 01:24:45 UTC

# Source the enhanced telemetry collector
TELEMETRY_COLLECTOR_PATH="$(dirname "${BASH_SOURCE[0]}")/collective-intelligence/enhanced-telemetry-collector.sh"
if [[ -f "$TELEMETRY_COLLECTOR_PATH" ]]; then
    source "$TELEMETRY_COLLECTOR_PATH"
else
    # Fallback to find collector in parent directories
    for i in {1..5}; do
        TELEMETRY_COLLECTOR_PATH="$(dirname "${BASH_SOURCE[0]}")$(printf '/..'%.0s {1..$i})/collective-intelligence/enhanced-telemetry-collector.sh"
        if [[ -f "$TELEMETRY_COLLECTOR_PATH" ]]; then
            source "$TELEMETRY_COLLECTOR_PATH"
            break
        fi
    done
fi

# Set script name for telemetry
export COLLECTIVE_SCRIPT_NAME="launch_mcps_as_sse.sh"

# Original script content below
# ============================================


# 检查是否提供了配置文件路径参数
if [ -z "$1" ]; then
  echo "Usage: $0 <config_file_path>"
  exit 1
fi

# 构造完整路径
CONFIG_FILE="$1"
if [[ ! "$CONFIG_FILE" == /* ]]; then
  CONFIG_FILE="configs/$CONFIG_FILE"
fi

# 检查配置文件是否存在
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "配置文件 '$CONFIG_FILE' 不存在。"
  exit 1
fi

# 读取 mcp_pool 数组的长度
SERVER_COUNT=$(jq '.mcp_pool | length' "$CONFIG_FILE")

if [[ "$SERVER_COUNT" -eq 0 ]]; then
  echo "mcp_pool 中未定义服务器。"
  exit 1
fi

# 遍历 mcp_pool 数组，启动每个服务器
for (( i=0; i<SERVER_COUNT; i++ ))
do
  # 使用 jq 提取每个服务器的配置信息
  SERVER=$(jq ".mcp_pool[$i]" "$CONFIG_FILE")

  NAME=$(echo "$SERVER" | jq -r '.name')

  # 检查是否存在 url 字段
  URL=$(echo "$SERVER" | jq -r '.url // empty')

  if [[ -n "$URL" ]]; then
    # 如果存在 url，则不运行 run_config，直接输出相关信息
    echo "服务器 '$NAME' 已配置 URL: $URL，跳过运行命令。"
  else
    # 从 run_config 数组中提取 command、args 和 port
    # 修改后的 args 提取逻辑：支持字符串或列表
    COMMAND=$(echo "$SERVER" | jq -r '.run_config[] | select(.command) | .command')
    ARGS=$(echo "$SERVER" | jq -r '.run_config[] | select(.args) | .args | if type == "array" then join(" ") else . end')
    PORT=$(echo "$SERVER" | jq -r '.run_config[] | select(.port) | .port')

    # 从 tools 数组中提取 tool_name（假设第一个工具）
    TOOL_NAME=$(echo "$SERVER" | jq -r '.tools[0].tool_name')

    # 由于配置中没有 tool_keyword，可以设置为空字符串或根据需要定义
    TOOL_KEYWORD=""

    echo "启动服务器: $NAME on port $PORT"

    # 启动服务器并将其置于后台
    npx -y supergateway \
      --stdio "$ARGS $COMMAND" \
      --port "$PORT" \
      --baseUrl "http://localhost:$PORT" \
      --ssePath /sse \
      --messagePath /message \
      --name "$TOOL_NAME" \
      --keyword "$TOOL_KEYWORD" &

    PID=$!
    echo "服务器 '$NAME' 已启动，PID: $PID"
  fi
done

wait
