# Quasar Deployment Fixes — Backend (Render) + Frontend (Vercel)

This document lists all required changes to make Quasar production-ready for deployment with the backend at **Render** (`https://quasar-oi14.onrender.com`) and the frontend at **Vercel**.

---

## 1. Fix `ui-pro/launch.py` — Dynamic PORT for Render

**Problem:** `launch.py` hardcodes `port=8000`. Render assigns a dynamic port via the `PORT` environment variable. The app will fail to bind on Render.

**Current code (`ui-pro/launch.py`):**
```python
uvicorn.run("api.main:app", host="0.0.0.0", port=8000)
```

**Fixed code:**
```python
import os
port = int(os.environ.get("PORT", 8000))
uvicorn.run("api.main:app", host="0.0.0.0", port=port)
```

**Render Start Command (set in Render Dashboard > Settings > Build & Deploy):**
```
cd ui-pro && python launch.py
```

---

## 2. Add Missing Python Dependencies to `requirements.txt`

**Problem:** Several packages imported by `ui-pro/api/main.py` and `services/auth.py` are not listed in `requirements.txt`. Render's build will fail.

**Add these lines to `requirements.txt`:**
```
fastapi
uvicorn[standard]
PyJWT
google-auth
google-genai
```

| Package | Where it's used |
|---------|----------------|
| `fastapi` | `ui-pro/api/main.py` — the entire backend framework |
| `uvicorn[standard]` | `ui-pro/launch.py` — ASGI server to run FastAPI |
| `PyJWT` | `services/auth.py` — `import jwt` for token signing |
| `google-auth` | `ui-pro/api/main.py` — `google.oauth2.id_token` for Google OAuth |
| `google-genai` | `ui-pro/api/main.py` — Gemini model routing |

---

## 3. Set Environment Variables on Render (Backend)

**Problem:** The `.env` file is gitignored (correctly). These must be set manually in Render Dashboard > Environment.

| Variable | Value | Notes |
|----------|-------|-------|
| `OPENAI_API_KEY` | `sk-proj-...` | Required for GPT models |
| `GEMINI_API_KEY` | `AIzaSy...` | Required for Gemini models |
| `NASA_ADS_API_KEY` | `yrftz7...` | Required for literature search |
| `GOOGLE_CLIENT_ID` | `84870453296-...` | Required for Google OAuth |
| `JWT_SECRET` | *(generate a strong random string)* | **Do NOT use the hardcoded default** in `services/auth.py` |
| `QUASAR_ENV` | `production` | Switch from development mode |
| `PYTHON_VERSION` | `3.12` | Render build setting (avoid 3.13 cgi issues) |

---

## 4. Set Environment Variables on Vercel (Frontend)

**Problem:** Without `NEXT_PUBLIC_API_URL`, the frontend defaults to `http://localhost:8000` which doesn't exist in production.

Set these in **Vercel Dashboard > Settings > Environment Variables**:

| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_API_URL` | `https://quasar-oi14.onrender.com` |
| `NEXT_PUBLIC_GOOGLE_CLIENT_ID` | `84870453296-jmmnt0c85sfb648hvb9sceggooutj5ee.apps.googleusercontent.com` |

---

## 5. SQLite Ephemeral Storage Warning

**Problem:** Render uses an **ephemeral filesystem** — all files are wiped on every deploy or restart. The following SQLite databases will be lost:

- `data/users.db` (user accounts — `services/auth.py`)
- `data/personalization.db` (uploaded documents metadata — `ui-pro/api/main.py`)

**Impact:** Users will need to re-register and re-upload documents after every deploy.

**Options:**
1. **Accept it** for now (fine for beta/demo)
2. **Add a Render Persistent Disk** ($0.25/GB/month) mounted at `/data`
3. **Migrate to PostgreSQL** (Render offers managed Postgres) — requires code changes in `services/auth.py` and the personalization DB logic

---

## 6. ChromaDB Vector Store Warning

**Problem:** The local `chroma_db/` directory (~27MB) stores RAG embeddings. This will also be lost on Render's ephemeral disk.

**Options:**
1. **Rebuild on startup** — add an init script that re-ingests documents
2. **Use Render Persistent Disk** — mount at the chroma_db path
3. **Switch to a hosted vector DB** (Pinecone, Weaviate, etc.)

---

## 7. Google OAuth Redirect URIs

**Problem:** Google OAuth requires authorized redirect URIs. After deploying to Vercel, you must update the Google Cloud Console.

**Steps:**
1. Go to [Google Cloud Console > Credentials](https://console.cloud.google.com/apis/credentials)
2. Edit the OAuth 2.0 Client ID `84870453296-...`
3. Add to **Authorized JavaScript origins**:
   - `https://<your-app>.vercel.app`
4. Add to **Authorized redirect URIs**:
   - `https://<your-app>.vercel.app`

---

## 8. Render Build Settings

Set these in **Render Dashboard > Build & Deploy**:

| Setting | Value |
|---------|-------|
| **Root Directory** | *(leave blank — use repo root)* |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `cd ui-pro && python launch.py` |
| **Python Version** | Set `PYTHON_VERSION=3.12` in env vars |

---

## 9. Vercel Build Settings

Set these in **Vercel Dashboard > Build & Deploy**:

| Setting | Value |
|---------|-------|
| **Root Directory** | `ui-pro` |
| **Framework Preset** | Next.js |
| **Build Command** | `npm run build` |
| **Output Directory** | `.next` |

---

## Quick Checklist

- [ ] Fix `ui-pro/launch.py` to read `PORT` from env
- [ ] Add `fastapi`, `uvicorn[standard]`, `PyJWT`, `google-auth`, `google-genai` to `requirements.txt`
- [ ] Set Render env vars (API keys, JWT_SECRET, GOOGLE_CLIENT_ID)
- [ ] Set Vercel env vars (NEXT_PUBLIC_API_URL, NEXT_PUBLIC_GOOGLE_CLIENT_ID)
- [ ] Update Google OAuth redirect URIs in Google Cloud Console
- [ ] Decide on persistent storage strategy (SQLite + ChromaDB)
- [ ] Set Render start command: `cd ui-pro && python launch.py`
- [ ] Set Vercel root directory to `ui-pro`
