"""Custom user-tool + per-user MCP-server registry endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_current_user
from api.models import MCPServerRequest, UserToolRequest

router = APIRouter()


# ── Custom Tool Endpoints ────────────────────────────────────
@router.get("/api/user-tools")
async def list_user_tools(current_user: dict = Depends(get_current_user)):
    from services.user_tools_service import UserToolsService
    svc = UserToolsService()
    return svc.load_tools(current_user["sub"])


@router.post("/api/user-tools")
async def save_user_tool(req: UserToolRequest, current_user: dict = Depends(get_current_user)):
    from services.user_tools_service import UserToolsService
    svc = UserToolsService()
    user_id = current_user["sub"]

    # Save secret if provided
    if req.api_key_name and req.api_key_value:
        svc.save_secret(user_id, req.api_key_name.strip(), req.api_key_value.strip())

    try:
        tool = svc.save_tool(
            user_id,
            name=req.name.strip(),
            description=req.description.strip(),
            code=req.code,
            api_key_name=req.api_key_name.strip() if req.api_key_name else None
        )
        return {"status": "success", "tool": tool}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/user-tools/{tool_name}")
async def delete_user_tool(tool_name: str, current_user: dict = Depends(get_current_user)):
    from services.user_tools_service import UserToolsService
    svc = UserToolsService()
    user_id = current_user["sub"]

    if svc.delete_tool(user_id, tool_name):
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Tool not found")


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
