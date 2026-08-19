#!/bin/bash
#
# Starts the VASPilot MCP server and the Quart crew server together for this demo,
# so you don't have to run `vaspilot_mcp` and `vaspilot_quart` in two separate terminals.
#
# Usage: ./run_demo.sh
# Stop both servers with Ctrl+C.
#
# All settings below can be overridden by exporting the corresponding env var
# before running this script, e.g.:
#   CREW_CONFIG=/path/to/other_config.yaml ./run_demo.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MCP_CONFIG="${MCP_CONFIG:-$SCRIPT_DIR/configs/mcp_config.yaml}"
MCP_PORT="${MCP_PORT:-8933}"

# Prefer a local (untracked, personal) crew config if present, else the tracked default.
if [ -f "$SCRIPT_DIR/configs/crew_config_local.yaml" ]; then
    CREW_CONFIG_DEFAULT="$SCRIPT_DIR/configs/crew_config_local.yaml"
else
    CREW_CONFIG_DEFAULT="$SCRIPT_DIR/configs/crew_config.yaml"
fi
CREW_CONFIG="${CREW_CONFIG:-$CREW_CONFIG_DEFAULT}"
CREW_PORT="${CREW_PORT:-51293}"
WORK_DIR="${WORK_DIR:-$SCRIPT_DIR/crew_server/work}"
ALLOW_PATH="${ALLOW_PATH:-$REPO_ROOT}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-2}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-10}"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR" "$WORK_DIR"

if [ -z "${PMG_VASP_PSP_DIR:-}" ]; then
    echo "Warning: PMG_VASP_PSP_DIR is not set. VASP POTCAR generation will fail unless it's exported (e.g. in your shell profile)." >&2
fi

echo "Starting vaspilot_mcp on port $MCP_PORT ..."
vaspilot_mcp --config "$MCP_CONFIG" --port "$MCP_PORT" > "$LOG_DIR/mcp.log" 2>&1 &
MCP_PID=$!

echo "Starting vaspilot_quart on port $CREW_PORT ..."
vaspilot_quart --config "$CREW_CONFIG" --port "$CREW_PORT" \
    --work-dir "$WORK_DIR" --allow-path "$ALLOW_PATH" \
    --max-concurrent-tasks "$MAX_CONCURRENT_TASKS" --max-queue-size "$MAX_QUEUE_SIZE" \
    > "$LOG_DIR/quart.log" 2>&1 &
QUART_PID=$!

cleanup() {
    echo ""
    echo "Stopping servers..."
    kill "$MCP_PID" "$QUART_PID" 2>/dev/null
    wait "$MCP_PID" "$QUART_PID" 2>/dev/null
    echo "Stopped."
}
trap cleanup EXIT INT TERM

echo ""
echo "vaspilot_mcp   PID $MCP_PID  -> $LOG_DIR/mcp.log"
echo "vaspilot_quart PID $QUART_PID  -> $LOG_DIR/quart.log  (http://0.0.0.0:$CREW_PORT)"
echo "Press Ctrl+C to stop both servers."
echo ""

wait -n "$MCP_PID" "$QUART_PID"
echo "One of the servers exited; check the logs above for details."
