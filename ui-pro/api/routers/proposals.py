"""Red Team TAC (Proposal Critic) streaming endpoint."""

import asyncio
import json

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import StreamingResponse

from api.deps import _executor, get_current_user

router = APIRouter()


@router.post("/api/proposals/review")
async def review_proposal(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload a proposal PDF and stream a Red Team TAC critique.

    S5: requires authentication (the frontend now sends the user's Bearer token).
    """
    import tempfile
    from pathlib import Path

    async def generate():
        def _status(step: str, state: str = "running"):
            return f"data: {json.dumps({'type': 'status', 'step': step, 'state': state})}\n\n"

        tmp_path = None
        try:
            raw = await file.read()
            ext = (Path(file.filename or "file.pdf").suffix or ".pdf").lower()
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

            loop = asyncio.get_event_loop()
            queue = asyncio.Queue()

            def progress_callback(msg: str, pct: int):
                asyncio.run_coroutine_threadsafe(
                    queue.put(("status", msg, "running" if pct < 100 else "completed")), loop
                )

            def _run_critic():
                try:
                    from services.proposal_critic import ProposalCriticService
                    from services.rag_service import RAGService
                    critic = ProposalCriticService()
                    rag = RAGService()
                    res = critic.review_proposal(tmp_path, rag, progress_callback=progress_callback)
                    asyncio.run_coroutine_threadsafe(queue.put(("done", res)), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)

            # Start execution in thread
            loop.run_in_executor(_executor, _run_critic)

            yield _status("Initializing Proposal Critic...", "running")

            while True:
                msg = await queue.get()
                if msg[0] == "status":
                    yield _status(msg[1], msg[2])
                elif msg[0] == "done":
                    res = msg[1]
                    if res.get("success"):
                        yield _status("Critique generated", "completed")
                        # Emitting the critique text as tokens so it can render immediately
                        critique_text = res.get('critique', '')
                        data_event = json.dumps({'type': 'critique', 'content': critique_text})
                        yield f"data: {data_event}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'content': res.get('error', 'Unknown Error')})}\n\n"
                    break
                elif msg[0] == "error":
                    yield f"data: {json.dumps({'type': 'error', 'content': msg[1]})}\n\n"
                    break

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if tmp_path:
                try: Path(tmp_path).unlink()
                except: pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
