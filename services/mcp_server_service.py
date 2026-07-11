import json
import os
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
    name: str              # e.g., "sqlite_db" or "github"
    transport: str = "stdio" # "stdio" or "http" (SSE)
    # Stdio args
    command: Optional[str] = None  # e.g., "npx" or "uvx"
    args: List[str] = []   # e.g., ["-y", "@modelcontextprotocol/server-sqlite", "--db", "test.db"]
    # SSE args
    url: Optional[str] = None # e.g. "https://huggingface.co/mcp"
    
    env: Dict[str, str] = {}    # e.g., {"GITHUB_TOKEN": "ghp_..."}

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
        
        # Validations
        if not server_config.name:
            raise ValueError("Server name is required.")
        if server_config.transport == "stdio" and not mcp_stdio_enabled():
            # RCE guard: refuse to persist a local-command MCP server. Use an
            # http/SSE URL instead (or enable QUASAR_ENABLE_MCP_STDIO in a
            # trusted local/dev environment). (S2)
            raise ValueError(
                "stdio MCP servers (local command spawning) are disabled on this "
                "deployment. Provide an http/SSE server URL instead."
            )
        if server_config.transport == "stdio" and not server_config.command:
            raise ValueError("Server command is required for stdio transport.")
        if server_config.transport == "http" and not server_config.url:
            raise ValueError("Server URL is required for http/sse transport.")
            
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
