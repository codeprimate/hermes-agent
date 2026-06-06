"""Mem0 memory plugin — MemoryProvider interface.

Server-side LLM fact extraction, semantic search with reranking, and
automatic deduplication via Mem0.

Two modes:
  cloud  (default) — uses MemoryClient to talk to api.mem0.ai (or custom host).
                      Requires MEM0_API_KEY.
  local             — uses the Memory() library in-process with local Qdrant/SQLite.
                      Configured to avoid OpenAI models (uses OpenRouter instead).
                      Requires OPENROUTER_API_KEY.

Config via environment variables:
  MEM0_API_KEY       — Mem0 Platform API key (cloud mode)
  MEM0_USER_ID       — User identifier (default: hermes-user)
  MEM0_AGENT_ID      — Agent identifier (default: hermes)
  MEM0_MODE          — "cloud" (default) or "local"
  OPENROUTER_API_KEY — LLM provider for local mode

Note: OpenAI models should be avoided. If this restriction ever blocks a
task, ASK the user — they will help reason through alternatives.

Or via $HERMES_HOME/mem0.json.

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import threading
import time
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# SQLite error messages that indicate a broken connection requiring client reset.
# Once a SQLite connection sees these errors (often from dual-process contention
# on the shared history.db), every subsequent operation fails. The only recovery
# is to close the connection and create a fresh Memory instance.
_SQLITE_FATAL_ERRORS = (
    "attempt to write a readonly database",
    "disk i/o error",
    "cannot rollback",
    "database is locked",
    "database disk image is malformed",
    "sqlite_busy",
    "sqlite_ioerr",
    "sqlite_readonly",
)


def _is_sqlite_fatal(exc: Exception) -> bool:
    """Return True if *exc* (or its cause) is a fatal SQLite error.

    The Hermes CLI and gateway share the same history.db.  When two processes
    write concurrently under WAL mode, SQLite can return ``SQLITE_READONLY``,
    ``SQLITE_IOERR`` or similar transient-but-actually-fatal-per-connection
    errors.  Once a connection enters this state the only recovery is to
    discard it and open a fresh one.
    """
    msg = str(exc).lower()
    return any(pattern in msg for pattern in _SQLITE_FATAL_ERRORS) or isinstance(
        exc, sqlite3.OperationalError
    )

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# HTTP proxy client — routes memory ops through the gateway API server
# so the SQLite history.db has a single writer (the gateway process).
# ---------------------------------------------------------------------------

class _HttpMemoryProxy:
    """Mem0-compatible proxy that forwards calls to the gateway's /api/memory/*.

    Used by the CLI when ``mem0.json`` has a ``gateway_url`` setting.
    All SQLite writes happen in the gateway process — no cross-process
    contention on history.db.

    Implements the subset of mem0 ``Memory`` API used by Mem0MemoryProvider.
    """

    def __init__(
        self,
        gateway_url: str,
        api_key: str,
        user_id: str = "hermes-user",
        agent_id: str = "hermes",
    ):
        self._base_url = gateway_url.rstrip("/")
        self._api_key = api_key
        self._user_id = user_id
        self._agent_id = agent_id
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._session = None  # lazily created httpx client
        self._closed = False

    def _get_session(self):
        if self._session is None:
            import httpx
            self._session = httpx.Client(timeout=30.0)
        return self._session

    def add(self, messages, *, user_id=None, agent_id=None, infer=False, **kwargs):
        if self._closed:
            raise RuntimeError("Memory proxy is closed")
        if infer:
            # Inference-mode adds go through the normal sync pipeline;
            # we only proxy non-inference (direct fact store) calls.
            return []
        session = self._get_session()
        flats = "; ".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in (messages if isinstance(messages, list) else [messages])
        )
        resp = session.post(
            f"{self._base_url}/api/memory/conclude",
            json={"conclusion": flats},
            headers=self._headers,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Memory proxy error ({resp.status_code}): {resp.text}")
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        return data.get("results", [])

    def search(self, query, *, user_id=None, agent_id=None,
               rerank=False, top_k=10, filters=None, **kwargs):
        if self._closed:
            raise RuntimeError("Memory proxy is closed")
        session = self._get_session()
        resp = session.post(
            f"{self._base_url}/api/memory/search",
            json={"query": str(query), "rerank": rerank, "top_k": min(top_k, 50)},
            headers=self._headers,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Memory proxy error ({resp.status_code}): {resp.text}")
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        results = data.get("results", data.get("result", []))
        if isinstance(results, str):
            return []
        return [type("Hit", (), {"id": "", "payload": {"data": r["memory"]}, "score": r.get("score", 0)})()  # noqa
                if isinstance(r, dict) else r for r in results]

    def get_all(self, *, filters=None, **kwargs):
        if self._closed:
            raise RuntimeError("Memory proxy is closed")
        session = self._get_session()
        resp = session.post(
            f"{self._base_url}/api/memory/profile",
            json={},
            headers=self._headers,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Memory proxy error ({resp.status_code}): {resp.text}")
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        result = data.get("result", "")
        if isinstance(result, str) and result:
            lines = result.split("\n")
            return [{"memory": line, "id": "", "hash": "", "score": 0} for line in lines]
        return data.get("results", [])

    @property
    def db(self):
        """Minimal db shim so _reset_client().db.close() doesn't crash."""
        return self

    def close(self):
        if self._session:
            self._session.close()
            self._session = None
        self._closed = True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides.

    Environment variables provide defaults; mem0.json (if present) overrides
    individual keys.  This avoids a silent failure when the JSON file exists
    but is missing fields like ``api_key`` that the user set in ``.env``.
    """
    from hermes_constants import get_hermes_home

    config = {
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "mode": os.environ.get("MEM0_MODE", "cloud"),
        "rerank": True,
        "keyword_search": False,
        # Local-mode options
        "llm_model": os.environ.get("MEM0_LLM_MODEL", "deepseek/deepseek-v4-flash"),
        "embedding_model": os.environ.get("MEM0_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b"),
        "embedding_dims": int(os.environ.get("MEM0_EMBEDDING_DIMS", "1024")),
        "vector_store_path": os.environ.get("MEM0_VECTOR_STORE_PATH", ""),
        "history_db_path": os.environ.get("MEM0_HISTORY_DB_PATH", os.path.expanduser("~/.hermes/data/mem0/history.db")),
        # Qdrant server mode (when vector_store_path is empty)
        "qdrant_host": os.environ.get("MEM0_QDRANT_HOST", "127.0.0.1"),
        "qdrant_port": int(os.environ.get("MEM0_QDRANT_PORT", "6333")),
        "qdrant_health_url": os.environ.get("MEM0_QDRANT_HEALTH_URL", "http://127.0.0.1:6333/healthz"),
        # Gateway proxy mode — forward memory ops to the API server
        # so the SQLite history.db has a single writer (the gateway process).
        "gateway_url": os.environ.get("MEM0_GATEWAY_URL", ""),
    }

    config_path = get_hermes_home() / "mem0.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity. "
        "Set rerank=true for higher accuracy on important queries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {"type": "boolean", "description": "Enable reranking for precision (default: false)."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Store a durable fact about the user. Stored verbatim (no LLM extraction). "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 Platform memory with server-side extraction and semantic search."""

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._api_key = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._mode = "cloud"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        mode = cfg.get("mode", "cloud")
        if mode == "local":
            return bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        return bool(cfg.get("api_key"))

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self):
        return [
            {"key": "api_key", "description": "Mem0 Platform API key (cloud mode only)", "secret": True, "required": False, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "mode", "description": "Operation mode", "default": "cloud", "choices": ["cloud", "local"]},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "true", "choices": ["true", "false"]},
        ]

    def _get_client(self):
        """Thread-safe client accessor with lazy initialization.

        Cloud mode returns a MemoryClient connected to the Mem0 Platform API.
        Local mode returns a Memory() library instance with local Qdrant + SQLite,
        configured to use OpenRouter (or another OpenAI-compatible endpoint) for
        LLM fact extraction and embeddings.
        """
        with self._client_lock:
            if self._client is not None:
                return self._client
            if self._mode == "local":
                return self._get_local_client()
            try:
                from mem0 import MemoryClient
                self._client = MemoryClient(api_key=self._api_key)
                return self._client
            except ImportError:
                raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")

    def _get_local_client(self):
        """Create a local Memory() library instance with OpenRouter via env var.

        Returns Memory instance (cached).  The Mem0 LLM layer auto-detects
        OPENROUTER_API_KEY and routes to OpenRouter; the embedder is configured
        with the same key + base URL so both extraction and vectorisation use
        the same provider.

        In Qdrant server mode (vector_store_path empty), also runs the
        health-check protocol to ensure Qdrant is reachable before creating
        the client.

        If ``gateway_url`` is set in config, returns an ``_HttpMemoryProxy``
        that forwards all operations to the gateway's /api/memory/* endpoints
        instead of creating a local Memory() instance.  This avoids dual-process
        SQLite contention on history.db.
        """
        cfg = self._config or {}

        # Gateway proxy mode — forward all memory ops to the API server
        gateway_url = cfg.get("gateway_url", "").strip()
        if gateway_url:
            api_key = cfg.get("api_key", os.environ.get("API_SERVER_KEY", ""))
            if not api_key:
                logger.warning(
                    "Mem0 gateway proxy configured (%s) but no API key found. "
                    "Set api_key in mem0.json or API_SERVER_KEY in .env.",
                    gateway_url,
                )
                # Fall through — create local client as fallback
            else:
                return _HttpMemoryProxy(
                    gateway_url=gateway_url,
                    api_key=api_key,
                    user_id=cfg.get("user_id", "hermes-user"),
                    agent_id=cfg.get("agent_id", "hermes"),
                )

        try:
            from mem0 import Memory

            cfg = self._config or {}
            embed_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
            embed_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
            emb_model = cfg.get("embedding_model", "qwen/qwen3-embedding-8b")
            emb_dims = int(cfg.get("embedding_dims", 1024))

            # Build vector store config: server mode (host:port) when
            # vector_store_path is empty/absent, else embedded (path).
            vs_path = cfg.get("vector_store_path", "")
            if vs_path:
                vs_config = {
                    "on_disk": True,
                    "path": vs_path,
                    "embedding_model_dims": emb_dims,
                }
            else:
                # Server mode — ensure Qdrant is healthy before creating client
                self._ensure_qdrant_healthy()
                vs_config = {
                    "host": cfg.get("qdrant_host", "127.0.0.1"),
                    "port": int(cfg.get("qdrant_port", 6333)),
                    "embedding_model_dims": emb_dims,
                }

            mem_config = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": cfg.get("llm_model", "deepseek/deepseek-v4-flash"),
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": emb_model,
                        "openai_base_url": embed_url,
                        "api_key": embed_key,
                        "embedding_dims": emb_dims,
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": vs_config,
                },
                "history_db_path": cfg.get("history_db_path", os.path.expanduser("~/.hermes/data/mem0/history.db")),
                "reranker": None,
                "version": "v1.1",
            }

            memory = Memory.from_config(mem_config)
            self._client = memory
            return self._client
        except ImportError:
            raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")

    def _reset_client(self) -> None:
        """Discard the cached Memory client so the next call to _get_client()
        creates a fresh one with a new SQLite connection.

        This is the only way to recover from fatal SQLite errors (readonly
        database, I/O error, broken rollback state) that arise from dual-process
        contention on the shared history.db between the CLI and gateway.
        """
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.db.close()
                except Exception:
                    pass
                self._client = None

    def _ensure_qdrant_healthy(self) -> bool:
        """Verify Qdrant server is reachable, attempting recovery if not.

        In server mode (vector_store_path empty), the mem0 library's
        Memory.from_config() will try to connect to the Qdrant server.
        If the server is down, the Memory init will fail with a connection
        error. This method proactively checks health and restarts the
        Qdrant container via the ensure_qdrant.sh script before we
        attempt to create the client.

        Returns True if Qdrant is healthy, False if unreachable.
        """
        cfg = self._config or {}
        vs_path = cfg.get("vector_store_path", "")

        # Only applies in server mode
        if vs_path:
            return True

        health_url = cfg.get("qdrant_health_url", "http://127.0.0.1:6333/healthz")

        # Fast check — is it already healthy?
        try:
            import urllib.request
            resp = urllib.request.urlopen(health_url, timeout=2)
            if resp.status == 200:
                return True
        except Exception:
            pass

        # Not healthy — run recovery script
        logger.warning("Qdrant health check failed. Running recovery protocol...")
        try:
            script = os.path.expanduser("~/services/hermes-shard/scripts/ensure_qdrant.sh")
            result = subprocess.run(
                [script, "--wait", "30"],
                capture_output=True, text=True, timeout=45,
            )
            if result.returncode in (0, 1):
                logger.info("Qdrant recovery succeeded (script exit %d).", result.returncode)
                return True
            logger.error("Qdrant recovery failed (script exit %d): %s",
                         result.returncode, result.stderr.strip())
        except Exception as e:
            logger.error("Qdrant recovery script failed: %s", e)

        return False

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures).

        Local mode has no network calls so the breaker is always open (never tripped).
        """
        if self._mode == "local":
            return False
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            # Cooldown expired — reset and allow a retry
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_key = self._config.get("api_key", "")
        # Prefer gateway-provided user_id for per-user memory scoping;
        # fall back to config/env default for CLI (single-user) sessions.
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._config.get("rerank", True)
        self._mode = self._config.get("mode", "cloud")

    def _read_filters(self) -> Dict[str, Any]:
        """Filters for search/get_all — scoped to user only for cross-session recall."""
        return {"user_id": self._user_id}

    def _write_filters(self) -> Dict[str, Any]:
        """Filters for add — scoped to user + agent for attribution."""
        return {"user_id": self._user_id, "agent_id": self._agent_id}

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        """Normalize Mem0 API response — v2 wraps results in {"results": [...]}."""
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 Memory\n"
            f"Active. User: {self._user_id}.\n"
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_profile for a full overview."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Mem0 Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                client = self._get_client()
                results = self._unwrap_results(client.search(
                    query=query,
                    filters=self._read_filters(),
                    rerank=self._rerank,
                    top_k=5,
                ))
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                if _is_sqlite_fatal(e):
                    self._reset_client()
                logger.debug("Mem0 prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for server-side fact extraction (non-blocking)."""
        if self._is_breaker_open():
            return

        def _sync():
            try:
                client = self._get_client()
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                client.add(messages, **self._write_filters())
                self._record_success()
            except Exception as e:
                self._record_failure()
                if _is_sqlite_fatal(e):
                    self._reset_client()
                logger.warning("Mem0 sync failed: %s", e)

        # Wait for any previous sync before starting a new one
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 API temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        if tool_name in ("mem0_profile", "mem0_search", "mem0_conclude"):
            # For Qdrant server mode, verify health before every operation
            cfg = self._config or {}
            if not cfg.get("vector_store_path", ""):
                if not self._ensure_qdrant_healthy():
                    return tool_error(
                        "Qdrant vector store is not available. "
                        "Memory operations are temporarily disabled. "
                        "The recovery protocol has been attempted — check "
                        "~/.hermes/logs/errors.log for details."
                    )

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "mem0_profile":
            try:
                memories = self._unwrap_results(client.get_all(filters=self._read_filters()))
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                if _is_sqlite_fatal(e):
                    self._reset_client()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            rerank = args.get("rerank", False)
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                results = self._unwrap_results(client.search(
                    query=query,
                    filters=self._read_filters(),
                    rerank=rerank,
                    top_k=top_k,
                ))
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                if _is_sqlite_fatal(e):
                    self._reset_client()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                client.add(
                    [{"role": "user", "content": conclusion}],
                    **self._write_filters(),
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                if _is_sqlite_fatal(e):
                    self._reset_client()
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            self._client = None


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
