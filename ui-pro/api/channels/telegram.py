"""
Telegram Channel Router for Quasar AI
Receives messages from Telegram bot and routes them through the QuasarAgent.
Setup:
  1. Create a bot via @BotFather on Telegram, get TELEGRAM_BOT_TOKEN
  2. Set webhooks: POST /channels/telegram/set-webhook
  3. Add TELEGRAM_BOT_TOKEN to your .env file
"""

import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Request, Response, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/channels/telegram", tags=["channels"])

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

_executor = ThreadPoolExecutor(max_workers=2)


class TelegramUpdate(BaseModel):
    update_id: int
    message: dict | None = None
    callback_query: dict | None = None


def _get_agent():
    """Lazy import to avoid circular imports."""
    from ui_pro.api.main import get_agent  # type: ignore
    return get_agent()


async def _send_telegram_message(chat_id: int, text: str):
    """Send a message back to Telegram using httpx."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }
            resp = await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json=payload,
                timeout=10.0
            )
            if resp.status_code != 200:
                print(f"[TELEGRAM] Send failed: {resp.text}")
    except Exception as e:
        print(f"[TELEGRAM] Error sending message: {e}")


@router.post("/webhook")
async def telegram_webhook(request: Request):
    """
    Receive updates from Telegram. 
    Each incoming message is routed through the QuasarAgent.
    """
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=503, detail="TELEGRAM_BOT_TOKEN not configured")

    try:
        body = await request.json()
    except Exception:
        return Response(content="ok", status_code=200)

    # Extract the message
    message = body.get("message", {})
    if not message:
        return Response(content="ok", status_code=200)

    chat_id = message.get("chat", {}).get("id")
    user_text = message.get("text", "").strip()
    user_id = str(message.get("from", {}).get("id", "telegram_user"))

    if not user_text or not chat_id:
        return Response(content="ok", status_code=200)

    # Send a "typing..." indicator
    async def typing():
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(f"{TELEGRAM_API}/sendChatAction", json={"chat_id": chat_id, "action": "typing"})
        except Exception:
            pass

    asyncio.create_task(typing())

    # Run the agent in a thread pool (it's synchronous)
    agent = _get_agent()
    if agent is None:
        await _send_telegram_message(chat_id, "⚠️ Quasar backend is starting up. Please try again in a moment.")
        return Response(content="ok", status_code=200)

    loop = asyncio.get_event_loop()
    response_text = await loop.run_in_executor(
        _executor,
        lambda: agent.stream_response_api(
            query=user_text,
            message_placeholder=None,
            user_id=user_id
        )
    )

    # Telegram has a 4096 char limit per message — split if needed
    max_len = 4000
    if len(response_text) <= max_len:
        await _send_telegram_message(chat_id, response_text)
    else:
        chunks = [response_text[i:i+max_len] for i in range(0, len(response_text), max_len)]
        for chunk in chunks:
            await _send_telegram_message(chat_id, chunk)

    return Response(content="ok", status_code=200)


@router.post("/set-webhook")
async def set_webhook(public_url: str):
    """
    Convenience endpoint to register the Telegram webhook.
    Call: POST /channels/telegram/set-webhook?public_url=https://your-domain.com
    """
    try:
        import httpx
        webhook_url = f"{public_url}/channels/telegram/webhook"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{TELEGRAM_API}/setWebhook",
                json={"url": webhook_url}
            )
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def telegram_status():
    """Check the current webhook status."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{TELEGRAM_API}/getWebhookInfo")
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
