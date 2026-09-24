import json
import os
import shlex
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel

def mcp_stdio_enabled() -> bool:
    """Whether per-user stdio MCP servers (local command spawning) are allowed.

    OFF by default. A stdio config persists ``{command, args, env}`` that the
    agent later spawns as a subprocess — arbitrary command execution on the
    host. HTTP/SSE (remote URL) MCP servers stay allowed; that is the intended
    consumption model (e.g. MANNA). Set ``QUASAR_ENABLE_MCP_STDIO=1`` only in a
    trusted local/dev environment (see docs/v2 S2). (S2)
    """
    return os.getenv("QUASAR_ENABLE_MCP_STDIO", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


class MCPServerConfig(BaseModel):
    name: str              # e.g., "manna" or "github"
    # "stdio" | "http" (legacy SSE) | "streamable_http" (modern MCP HTTP)
    transport: str = "stdio"
    # Stdio args
    command: Optional[str] = None  # e.g., "npx" or "uvx"
    args: List[str] = []   # e.g., ["-y", "@modelcontextprotocol/server-sqlite", "--db", "test.db"]
    # SSE args
    url: Optional[str] = None # e.g. "https://huggingface.co/mcp"

    env: Dict[str, str] = {}    # e.g., {"GITHUB_TOKEN": "ghp_..."}


# The complete transport roster. Anything else is rejected up front: the mount
# path treats every non-HTTP transport as stdio (command execution), so an
# unrecognized string must never reach it (CX-01).
KNOWN_TRANSPORTS = ("stdio", "http", "streamable_http")


def normalize_transport(value: Optional[str]) -> str:
    return (value or "stdio").strip().lower()


def validate_server_config(cfg: MCPServerConfig) -> MCPServerConfig:
    """Shared structural validation for user-saved AND platform configs.

    Normalizes ``transport`` in place and raises ``ValueError`` on a config
    the mount path could not start (or would start as something the author
    did not intend).
    """
    # Normalize, don't just check: a validated config must be canonical, or
    # " manna " becomes a noncanonical tool namespace and " uvx " an
    # unlaunchable executable downstream (CX-25).
    cfg.name = (cfg.name or "").strip()
    cfg.command = (cfg.command or "").strip() or None
    cfg.url = (cfg.url or "").strip() or None
    if not cfg.name:
        raise ValueError("Server name is required.")
    cfg.transport = normalize_transport(cfg.transport)
    if cfg.transport not in KNOWN_TRANSPORTS:
        raise ValueError(
            f"Unknown MCP transport {cfg.transport!r}; expected one of "
            f"{', '.join(KNOWN_TRANSPORTS)}."
        )
    if cfg.transport == "stdio" and not cfg.command:
        raise ValueError("Server command is required for stdio transport.")
    if cfg.transport in ("http", "streamable_http") and not cfg.url:
        raise ValueError("Server URL is required for http transports.")
    return cfg


def manna_enabled() -> bool:
    """Whether the MANNA MCP server (NSF-Simons CosmicAI's IVOA archive server)
    is mounted process-wide at agent startup.

    OFF by default: MANNA's tools overlap Quasar's native ALMA/Data Lab
    families, and the stdio default spawns a child process (``uvx``) whose
    dependency tree (astropy/pyvo) roughly doubles the Python footprint — do
    not enable stdio mode on a small host (e.g. the 2 GB Render box); point
    ``MANNA_MCP_URL`` at a remote server instead.
    """
    return os.getenv("QUASAR_ENABLE_MANNA", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


# Spawns MANNA from PyPI on demand; --stdio keeps it off Quasar's HTTP port
# (MANNA's HTTP default is :8000, which collides with the backend). Pinned to
# the release this integration was verified against — bump deliberately, not
# whenever upstream publishes.
_MANNA_DEFAULT_COMMAND = "uvx --from manna-mcp==0.9.0 manna --stdio"


def _split_command(command_line: str) -> List[str]:
    """Split a command line into argv, keeping Windows paths intact.

    POSIX shlex eats backslashes (``C:\\uv\\uvx.exe`` → ``C:uvuvx.exe``), so on
    Windows split in non-POSIX mode and strip the quote characters it leaves
    on quoted tokens.
    """
    if os.name == "nt":
        return [tok.strip('"') for tok in shlex.split(command_line, posix=False)]
    return shlex.split(command_line)


def platform_mcp_servers() -> List[dict]:
    """Operator-level (process-wide) MCP server configs, from environment.

    Unlike the per-user configs persisted by :class:`MCPServerService` (which
    come from the web UI and are RCE-gated by ``mcp_stdio_enabled``), these are
    set by whoever controls the deployment's environment — someone who can
    already run arbitrary commands — so stdio entries are honored as-is.

    Sources, in order:
    - ``QUASAR_ENABLE_MANNA``: mounts MANNA under the ``manna`` namespace.
      ``MANNA_MCP_URL`` selects a remote streamable-HTTP server; otherwise
      ``MANNA_MCP_COMMAND`` (default: pinned ``uvx --from manna-mcp==...``)
      is spawned over stdio.
    - ``QUASAR_PLATFORM_MCP_SERVERS``: JSON list of MCPServerConfig objects
      for any additional servers.

    Robustness contract: one bad entry (or a bad MANNA command) is logged and
    skipped without suppressing the other servers, and duplicate names keep
    the first occurrence — concurrent bridges must never race to register the
    same ``{name}__*`` tools.
    """
    servers: List[dict] = []

    def _add(cfg: MCPServerConfig, source: str) -> None:
        try:
            validate_server_config(cfg)
        except ValueError as e:
            print(f"[MCPServerService] Skipping {source} entry: {e}")
            return
        if any(s["name"] == cfg.name for s in servers):
            print(
                f"[MCPServerService] Skipping duplicate platform MCP server "
                f"name {cfg.name!r} from {source} (first definition wins)."
            )
            return
        servers.append(cfg.model_dump())

    if manna_enabled():
        try:
            url = os.getenv("MANNA_MCP_URL", "").strip()
            if url:
                # MANNA serves streamable HTTP at /mcp (SSE is its legacy path).
                _add(
                    MCPServerConfig(name="manna", transport="streamable_http", url=url),
                    "QUASAR_ENABLE_MANNA",
                )
            else:
                argv = _split_command(
                    os.getenv("MANNA_MCP_COMMAND", "").strip() or _MANNA_DEFAULT_COMMAND
                )
                _add(
                    MCPServerConfig(
                        name="manna",
                        transport="stdio",
                        command=argv[0] if argv else "",
                        args=argv[1:],
                    ),
                    "QUASAR_ENABLE_MANNA",
                )
        except Exception as e:
            # e.g. unbalanced quoting in MANNA_MCP_COMMAND — never let the
            # MANNA block suppress the generic platform servers below.
            print(f"[MCPServerService] Skipping MANNA config: {e}")

    raw = os.getenv("QUASAR_PLATFORM_MCP_SERVERS", "").strip()
    if raw:
        try:
            entries = json.loads(raw)
            if not isinstance(entries, list):
                raise ValueError("expected a JSON list")
        except Exception as e:
            # Unparseable JSON: nothing salvageable, log and move on.
            print(f"[MCPServerService] Ignoring QUASAR_PLATFORM_MCP_SERVERS: {e}")
            entries = []
        for i, entry in enumerate(entries):
            try:
                cfg = MCPServerConfig(**entry)
            except Exception as e:
                print(
                    f"[MCPServerService] Skipping QUASAR_PLATFORM_MCP_SERVERS "
                    f"entry {i}: {e}"
                )
                continue
            _add(cfg, f"QUASAR_PLATFORM_MCP_SERVERS[{i}]")
    return servers


class MCPServerService:
    """Service for managing per-user external MCP Server configurations."""
    
    def __init__(self, base_dir: str = "user_tools"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def _user_dir(self, user_id: str) -> Path:
        p = self.base_dir / str(user_id)
        p.mkdir(parents=True, exist_ok=True)
        return p
        
    def _servers_file(self, user_id: str) -> Path:
        return self._user_dir(user_id) / "mcp_servers.json"
        
    def load_servers(self, user_id: str) -> List[dict]:
        """Load all external MCP server configs for a user."""
        sf = self._servers_file(user_id)
        if not sf.exists():
            return []
        try:
            with open(sf, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[MCPServerService] Error loading servers for {user_id}: {e}")
            return []
            
    def save_server(self, user_id: str, server_config: MCPServerConfig) -> dict:
        """Add or update an MCP server configuration."""
        servers = self.load_servers(user_id)
        
        # Structural validation (also normalizes + rejects unknown transports,
        # which the mount path would otherwise execute as stdio — CX-01).
        validate_server_config(server_config)
        if server_config.transport == "stdio" and not mcp_stdio_enabled():
            # RCE guard: refuse to persist a local-command MCP server. Use an
            # http/SSE URL instead (or enable QUASAR_ENABLE_MCP_STDIO in a
            # trusted local/dev environment). (S2)
            raise ValueError(
                "stdio MCP servers (local command spawning) are disabled on this "
                "deployment. Provide an http/SSE server URL instead."
            )
            
        # Update if exists, append if new
        updated = False
        dict_val = server_config.model_dump()
        for i, srv in enumerate(servers):
            if srv["name"] == server_config.name:
                servers[i] = dict_val
                updated = True
                break
                
        if not updated:
            servers.append(dict_val)
            
        with open(self._servers_file(user_id), "w") as f:
            json.dump(servers, f, indent=2)
            
        return dict_val
        
    def delete_server(self, user_id: str, name: str) -> bool:
        """Delete an MCP server config by name."""
        servers = self.load_servers(user_id)
        initial_count = len(servers)
        servers = [s for s in servers if s["name"] != name]
        
        if len(servers) < initial_count:
            with open(self._servers_file(user_id), "w") as f:
                json.dump(servers, f, indent=2)
            return True
        return False
