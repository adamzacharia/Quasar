"""Per-user MCP-server registry endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_current_user
from api.models import MCPServerRequest

router = APIRouter()


# ── MCP Server Endpoints ─────────────────────────────────────
@router.get("/api/mcp-servers")
async def list_mcp_servers(current_user: dict = Depends(get_current_user)):
    from services.mcp_server_service import MCPServerService
    svc = MCPServerService()
    return svc.load_servers(current_user["sub"])


@router.post("/api/mcp-servers")
async def save_mcp_server(req: MCPServerRequest, current_user: dict = Depends(get_current_user)):
    from services.mcp_server_service import MCPServerService, MCPServerConfig
    svc = MCPServerService()
    user_id = current_user["sub"]

    try:
        config = MCPServerConfig(**req.model_dump())
        srv = svc.save_server(user_id, config)
        return {"status": "success", "server": srv}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/mcp-servers/{name}")
async def delete_mcp_server(name: str, current_user: dict = Depends(get_current_user)):
    from services.mcp_server_service import MCPServerService
    svc = MCPServerService()
    user_id = current_user["sub"]

    if svc.delete_server(user_id, name):
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Server not found")
