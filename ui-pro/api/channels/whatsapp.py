"""
WhatsApp Channel Router for Quasar AI
Receives messages from WhatsApp Business Cloud API and routes them through the QuasarAgent.

Setup:
  1. Create a Meta Business account at https://business.facebook.com
  2. Create a Meta Developer App at https://developers.facebook.com
  3. Add the WhatsApp product to your app and register a phone number
  4. Add to your .env file:
       WHATSAPP_ACCESS_TOKEN=<your permanent system user token>
       WHATSAPP_PHONE_NUMBER_ID=<your phone number ID>
       WHATSAPP_VERIFY_TOKEN=<any random secret string you choose>
  5. Set your webhook URL to: https://<your-domain>/channels/whatsapp/webhook
  6. Subscribe to the "messages" webhook field

Docs: https://developers.facebook.com/docs/whatsapp/cloud-api
"""

import os
import json
import asyncio
import hashlib
import hmac
import re
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Request, Response, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/channels/whatsapp", tags=["channels"])

WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "quasar_whatsapp_verify")
# UIAPI-09: the Meta app secret used to verify X-Hub-Signature-256 on inbound
# webhooks. When set, verification is mandatory (fail closed); when unset, the
# webhook still works for dev setups but logs a warning per request.
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "")

GRAPH_API_VERSION = "v21.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

_executor = ThreadPoolExecutor(max_workers=2)


def _get_agent():
    """Lazy import to avoid circular imports.

    get_agent lives in api.deps since the main.py monolith split — importing
    it from api.main crashed every inbound message (scan UIAPI-01).
    """
    from api.deps import get_agent  # type: ignore
    return get_agent()


def _strip_markdown(text: str) -> str:
    """
    Convert common Markdown formatting to WhatsApp-compatible plain text.
    WhatsApp supports *bold*, _italic_, ~strikethrough~, and ```monospace```,
    but NOT headings (#), links, tables, etc.
    """
    # Remove markdown headings (## Title -> Title)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Convert markdown links [text](url) -> text (url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)
    # Convert markdown bold **text** to WhatsApp bold *text*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    # Remove image tags ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[\1]', text)
    return text


async def _send_whatsapp_message(recipient_id: str, text: str):
    """Send a text message back to the user via WhatsApp Cloud API."""
    try:
        import httpx
        url = f"{GRAPH_API_BASE}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient_id,
            "type": "text",
            "text": {
                "preview_url": True,
                "body": text,
            },
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload, timeout=15.0)
            if resp.status_code != 200:
                print(f"[WHATSAPP] Send failed ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"[WHATSAPP] Error sending message: {e}")


async def _mark_as_read(message_id: str):
    """Mark an incoming message as read (shows blue ticks to the user)."""
    try:
        import httpx
        url = f"{GRAPH_API_BASE}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        async with httpx.AsyncClient() as client:
            await client.post(url, headers=headers, json=payload, timeout=5.0)
    except Exception:
        pass  # Non-critical — just a UX enhancement


@router.get("/webhook")
async def whatsapp_verify(
    request: Request,
):
    """
    Webhook verification endpoint (called once by Meta when you register the webhook).
    Meta sends a GET request with hub.mode, hub.verify_token, and hub.challenge.
    We must respond with the challenge value if the verify_token matches.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        print(f"[WHATSAPP] Webhook verified successfully")
        return Response(content=challenge, status_code=200, media_type="text/plain")

    print(f"[WHATSAPP] Webhook verification failed: mode={mode}, token_match={token == WHATSAPP_VERIFY_TOKEN}")
    raise HTTPException(status_code=403, detail="Verification failed")


def _verify_whatsapp_signature(raw_body: bytes, signature_header: str) -> bool:
    """Constant-time check of Meta's X-Hub-Signature-256 header (UIAPI-09).

    The header is ``sha256=<hex hmac of the raw body with the app secret>``.
    """
    expected = "sha256=" + hmac.new(
        WHATSAPP_APP_SECRET.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header or "")


@router.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    Receive incoming messages from WhatsApp.
    Each text message is routed through the QuasarAgent.
    """
    if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        raise HTTPException(status_code=503, detail="WhatsApp credentials not configured")

    # ── UIAPI-09: authenticate the webhook before acting on it ──────────
    # Without this, any unauthenticated poster to the well-known webhook URL
    # can forge a Cloud-API payload that triggers a full agent run (unmetered
    # LLM spend) and makes the server send WhatsApp messages to arbitrary
    # numbers. Fail closed when the app secret is configured; warn-and-allow
    # when it is not, so dev setups without a Meta app keep working.
    raw_body = await request.body()
    if WHATSAPP_APP_SECRET:
        signature = request.headers.get("x-hub-signature-256", "")
        if not _verify_whatsapp_signature(raw_body, signature):
            print("[WHATSAPP] Rejected webhook POST with missing/invalid X-Hub-Signature-256")
            raise HTTPException(status_code=403, detail="Invalid webhook signature")
    else:
        print(
            "[WHATSAPP] WARNING: WHATSAPP_APP_SECRET is not set — webhook "
            "signature verification is DISABLED and inbound payloads are unauthenticated."
        )

    try:
        body = json.loads(raw_body)
    except Exception:
        return Response(content="ok", status_code=200)

    # WhatsApp Cloud API sends a nested structure:
    # body.entry[].changes[].value.messages[]
    entries = body.get("entry", [])
    for entry in entries:
        for change in entry.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages", [])

            for message in messages:
                # Only handle text messages for now
                if message.get("type") != "text":
                    continue

                sender_id = message.get("from", "")
                message_id = message.get("id", "")
                user_text = message.get("text", {}).get("body", "").strip()

                if not user_text or not sender_id:
                    continue

                # Mark message as read (blue ticks) — fire and forget
                asyncio.create_task(_mark_as_read(message_id))

                # Process the message asynchronously
                asyncio.create_task(
                    _process_and_reply(sender_id, user_text, message_id)
                )

    # Always return 200 to acknowledge receipt — mandatory for WhatsApp webhooks
    return Response(content="ok", status_code=200)


async def _process_and_reply(sender_id: str, user_text: str, message_id: str):
    """
    Run the QuasarAgent synchronously in a thread pool and send the reply.
    Separated from the webhook handler so we can return 200 immediately
    (WhatsApp requires fast acknowledgment to avoid retries).
    """
    agent = _get_agent()
    if agent is None:
        await _send_whatsapp_message(
            sender_id,
            "⏳ Quasar is starting up. Please try again in a moment."
        )
        return

    try:
        loop = asyncio.get_event_loop()

        def _run_with_context():
            # Install a request context (CX-41): without one, conductor
            # sub-agent tool traces have no collector and per-call accounting
            # hooks are absent for bot-channel turns.
            from core.llm_client import llm_request_context

            with llm_request_context(user_id=f"whatsapp_{sender_id}"):
                return agent.stream_response_api(
                    query=user_text,
                    message_placeholder=None,
                    user_id=f"whatsapp_{sender_id}",
                )

        response_text = await loop.run_in_executor(_executor, _run_with_context)
    except Exception as e:
        print(f"[WHATSAPP] Agent error: {e}")
        response_text = "Sorry, I encountered an error processing your request. Please try again."

    if not response_text or not response_text.strip():
        response_text = "I processed your request but didn't generate a response. Could you rephrase your question?"

    # Convert Markdown to WhatsApp-friendly formatting
    response_text = _strip_markdown(response_text)

    # WhatsApp has a 4096 character limit per message — split if needed
    max_len = 4000
    if len(response_text) <= max_len:
        await _send_whatsapp_message(sender_id, response_text)
    else:
        # Split on paragraph boundaries when possible
        chunks = []
        remaining = response_text
        while remaining:
            if len(remaining) <= max_len:
                chunks.append(remaining)
                break
            # Try to split at a paragraph break
            split_pos = remaining.rfind("\n\n", 0, max_len)
            if split_pos == -1:
                # Fall back to splitting at a newline
                split_pos = remaining.rfind("\n", 0, max_len)
            if split_pos == -1:
                # Fall back to splitting at max_len
                split_pos = max_len
            chunks.append(remaining[:split_pos])
            remaining = remaining[split_pos:].lstrip("\n")

        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                chunk = f"({i+1}/{len(chunks)})\n{chunk}"
            await _send_whatsapp_message(sender_id, chunk)


@router.get("/status")
async def whatsapp_status():
    """Check the current WhatsApp integration status."""
    status = {
        "configured": bool(WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID),
        "phone_number_id": WHATSAPP_PHONE_NUMBER_ID[:6] + "..." if WHATSAPP_PHONE_NUMBER_ID else None,
        "token_set": bool(WHATSAPP_ACCESS_TOKEN),
        "verify_token_set": bool(WHATSAPP_VERIFY_TOKEN),
        # UIAPI-09: whether inbound webhook signature verification is active.
        "app_secret_set": bool(WHATSAPP_APP_SECRET),
    }

    # Optionally verify the token is valid by calling the API
    if status["configured"]:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{GRAPH_API_BASE}/{WHATSAPP_PHONE_NUMBER_ID}",
                    headers={"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"},
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    status["phone_number"] = data.get("display_phone_number", "unknown")
                    status["quality_rating"] = data.get("quality_rating", "unknown")
                    status["api_status"] = "connected"
                else:
                    status["api_status"] = f"error ({resp.status_code})"
        except Exception as e:
            status["api_status"] = f"unreachable: {e}"

    return status
