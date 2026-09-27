"""Per-user MCP-server registry endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_current_user
from api.models import MCPServerRequest

router = APIRouter()


# ── MCP Server Endpoints ─────────────────────────────────────
@router.get("/api/mcp-servers")
async def list_mcp_servers(current_user: dict = Depends(get_current_user)):
    from services.mcp_server_service import MCPServerService
    from services.user_mcp import get_user_mcp_pool
    svc = MCPServerService()
    user_id = current_user["sub"]
    servers = svc.load_servers(user_id)
    # Last known connection state (no new connects here); secrets in env are
    # never echoed back beyond what the user saved.
    status = get_user_mcp_pool().status(user_id)
    for srv in servers:
        if not isinstance(srv, dict):
            continue
        # Secret values (tokens in env / headers) never travel back to the
        # browser; the panel only needs the names.
        for field in ("env", "headers"):
            if isinstance(srv.get(field), dict):
                srv[field] = {k: "********" for k in srv[field]}
        if srv.get("name") in status:
            srv["status"] = status[srv["name"]]
    return servers


@router.post("/api/mcp-servers/{name}/test")
async def test_mcp_server(name: str, current_user: dict = Depends(get_current_user)):
    """Connect to a saved server now and report tools or the real error."""
    import asyncio
    from services.user_mcp import get_user_mcp_pool
    return await asyncio.to_thread(get_user_mcp_pool().test, current_user["sub"], name)


@router.post("/api/mcp-servers")
async def save_mcp_server(req: MCPServerRequest, current_user: dict = Depends(get_current_user)):
    from services.mcp_server_service import MCPServerService, MCPServerConfig
    svc = MCPServerService()
    user_id = current_user["sub"]

    try:
        config = MCPServerConfig(**req.model_dump())
        srv = svc.save_server(user_id, config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    # "Save & Connect" really connects: the result tells the user at once
    # whether the server answered and which tools it offers.
    import asyncio
    from services.user_mcp import get_user_mcp_pool
    result = await asyncio.to_thread(get_user_mcp_pool().test, user_id, srv["name"])
    masked = {**srv, **{f: {k: "********" for k in (srv.get(f) or {})} for f in ("env", "headers")}}
    return {"status": "success", "server": masked, "connection": result}


@router.delete("/api/mcp-servers/{name}")
async def delete_mcp_server(name: str, current_user: dict = Depends(get_current_user)):
    from services.mcp_server_service import MCPServerService
    svc = MCPServerService()
    user_id = current_user["sub"]

    if svc.delete_server(user_id, name):
        from services.user_mcp import get_user_mcp_pool
        get_user_mcp_pool().forget(user_id, name)
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Server not found")
