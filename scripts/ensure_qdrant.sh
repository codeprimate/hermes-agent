#!/bin/bash
# ensure_qdrant.sh — Qdrant server health protocol
# Called by the Mem0 plugin (and potentially any other service) to verify
# Qdrant is reachable, and restart it if not.
#
# Exit codes:
#   0 — Qdrant is healthy
#   1 — Qdrant was restarted and is now healthy
#   2 — Qdrant is unreachable and could not be restored
#
# Usage: ensure_qdrant.sh [--wait SECONDS]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
CONTAINER_NAME="${QDRANT_CONTAINER:-qdrant}"
WAIT="${1:-}"
MAX_WAIT="${WAIT:-30}"  # default: wait up to 30s for recovery

log() { echo "[ensure-qdrant] $*" >&2; }

check_health() {
    curl -sf -o /dev/null "${QDRANT_URL}/healthz" 2>/dev/null
}

# Fast path: already healthy
if check_health; then
    exit 0
fi

log "Qdrant not reachable at ${QDRANT_URL}. Attempting recovery..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    log "Docker daemon is not running. Attempting to start Docker..."
    # On macOS, Docker Desktop is typically launched via open
    open -a Docker 2>/dev/null || true
    # Wait for Docker to become available
    for i in $(seq 1 15); do
        sleep 2
        if docker info >/dev/null 2>&1; then
            log "Docker daemon is now available."
            break
        fi
        if [ "$i" -eq 15 ]; then
            log "Docker daemon failed to start after 30s."
            exit 2
        fi
    done
fi

# Check if Qdrant container exists
if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    STATUS="$(docker inspect --format='{{.State.Status}}' "${CONTAINER_NAME}")"
    case "$STATUS" in
        running)
            log "Container '${CONTAINER_NAME}' is running but health check failed. Restarting..."
            docker restart "${CONTAINER_NAME}" >/dev/null 2>&1
            ;;
        paused|exited|dead)
            log "Container '${CONTAINER_NAME}' is ${STATUS}. Starting..."
            docker start "${CONTAINER_NAME}" >/dev/null 2>&1
            ;;
        restarting|created)
            log "Container '${CONTAINER_NAME}' is ${STATUS}. Waiting..."
            ;;
    esac
else
    log "Container '${CONTAINER_NAME}' does not exist. Creating..."
    docker run -d \
        --name "${CONTAINER_NAME}" \
        --restart unless-stopped \
        -p 6333:6333 \
        -p 6334:6334 \
        -v "/Users/codeprimate/services/hermes-shard/data/mem0/qdrant_storage:/qdrant/storage" \
        qdrant/qdrant:latest >/dev/null 2>&1
fi

# Wait for health
log "Waiting for Qdrant to become healthy (up to ${MAX_WAIT}s)..."
for i in $(seq 1 "${MAX_WAIT}"); do
    sleep 1
    if check_health; then
        log "Qdrant is now healthy after ${i}s."
        exit 1  # recovered
    fi
done

log "Qdrant failed to become healthy within ${MAX_WAIT}s."
exit 2