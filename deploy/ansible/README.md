# Quasar VM deploy (Ansible)

Rebuildable-from-scratch provisioning for a single Linux VM (the TACC box):
FastAPI backend on :8000, optional Next.js frontend on :3001, nginx in front
configured for SSE streaming (`proxy_buffering off`, read timeout 3600 s ≥ the
600 s minimum). Re-running the playbook is always safe — it converges the
machine to this recipe, so a dead VM is one command away from a clone.

```
deploy/ansible/
├── quasar.yml                    # the playbook
├── inventory.example.ini         # copy → inventory.ini, add the VM IP
├── group_vars/
│   ├── all.yml                   # ports, paths, repo, nginx knobs (non-secret)
│   └── vault.yml.example         # copy INTO `ansible-vault create` (secrets)
└── templates/                    # .env, systemd units, nginx site
```

## Prerequisites

- A control machine with **Ansible ≥ 2.14** and SSH access to the VM.
  Ansible does not run natively on Windows — use **WSL** (`sudo apt install
  ansible`), macOS, or any Linux box.
- Production target: Ubuntu 24.04 / CPython 3.12 on x86_64. The existing
  Rocky/RHEL 9 system tasks remain, but this lock is not claimed as
  acceptance-tested there.
- Firewall (TACC side): inbound **80/443 only** (nginx is the front door).
  Ports 8000/3001 stay closed — nginx reaches them on localhost.

## First-time setup

```bash
cd deploy/ansible
cp inventory.example.ini inventory.ini        # fill in VM IP + ssh user
ansible-vault create group_vars/vault.yml     # paste vault.yml.example, fill values
```

Secrets come from your local working `.env` (or the Render dashboard). The
vault file is encrypted at rest, so it is safe to keep — but `inventory.ini`
and any *unencrypted* vault are git-ignored regardless.

Check `group_vars/all.yml` before the first run:

- `quasar_server_name` / `quasar_public_origin` — preset to
  **www.quasarassistant.com** (canonical) + apex. `NEXT_PUBLIC_API_URL` is
  **baked into the frontend at build time**, so changing the origin requires
  re-running with `--tags deploy`.
- `quasar_deploy_frontend: false` if the UI stays on Vercel and this VM is
  backend-only.
- `quasar_repo_url` — if the repo is private, switch to the SSH URL and set
  `quasar_git_deploy_key_src` (add the pubkey as a GitHub deploy key).

## Provision / rebuild (identical command)

```bash
ansible-playbook -i inventory.ini quasar.yml --ask-vault-pass
```

~10–15 min on a fresh VM (the scientific Python stack dominates). To ship a
code update later (git pull → locked venv validation/switch → npm build →
restart, no system changes):

```bash
ansible-playbook -i inventory.ini quasar.yml --ask-vault-pass --tags deploy
```

## What it installs

| Piece | Detail |
|---|---|
| System | git, nginx, build tools, Python 3.12, Node 22 (NodeSource) |
| User/layout | `quasar` system user; repo at `/opt/quasar/app`; lock-addressed venvs under `/opt/quasar/venvs/`; stable link at `/opt/quasar/venv` |
| Backend | `quasar-backend.service` → `venv/bin/python launch.py` from `ui-pro/` (CWD matters: DiskCache paths are CWD-relative), `.env` rendered at repo root, journald logs |
| Frontend | `quasar-frontend.service` → `next start` on 127.0.0.1:3001 after `npm ci && npm run build` |
| nginx | `/api/` → :8000 with SSE settings; `/` → :3001; 200 MB upload cap |

Logs: `journalctl -u quasar-backend -f` / `-u quasar-frontend -f`.

## Production Python dependencies

The dependency files have deliberately different roles:

- `requirements/production.in` is the human-edited list of direct production
  dependencies. The root `requirements.txt` includes it for the existing
  development and CI workflow.
- `requirements/production-py312-linux-x86_64.txt` is generated and must not
  be edited by hand. Ansible installs this file with pip hash checking and a
  wheel-only policy.
- `requirements/optional-sparcl.in` is an optional Python 3.13 input. It is
  never included in Python 3.12 production.

SparCL 1.3.0 declares `pandas<2.2` on Python versions below 3.13. Production
keeps `lsdb==0.9.2`, whose nested-pandas dependency requires
`pandas>=2.2.3,<2.4`; the published metadata therefore cannot resolve together
on Python 3.12. Quasar already imports the SparCL client lazily, so only
SparCL-backed spectrum search, retrieval, plotting, stacking, and enrichment
are unavailable. Do not work around the conflict with `--no-deps` or a
resolver override.

Generate the lock with **uv 0.11.28** on Linux from the repository root using
the exact command below:

```bash
uv pip compile requirements/production.in \
  --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 \
  --only-binary :all: \
  --emit-build-options \
  --generate-hashes \
  --exclude-newer 2026-09-02T00:00:00Z \
  --upgrade \
  --no-cache \
  --no-python-downloads \
  --output-file requirements/production-py312-linux-x86_64.txt
```

`--upgrade` prevents an older output lock from influencing regeneration.
`--exclude-newer` freezes the candidate universe, while exact versions and
SHA-256 hashes make installation deterministic. If a future dependency has no
compatible wheel, stop and review that package explicitly instead of silently
removing the wheel-only policy.

Ansible hashes the lock and builds a new venv directly at
`/opt/quasar/venvs/py312-<lock-sha256>`. It runs `pip check`, imports the
critical packages, verifies `pyvo.registry.Freetext`, and asserts SparCL is
absent before switching `/opt/quasar/venv`. If service activation or health
validation fails, the previous symlink (or preserved legacy directory) is
restored before the play fails. Interrupted, unactivated releases without the
validation marker are rebuilt. On the first migration, an existing directory
at `/opt/quasar/venv` is preserved as
`/opt/quasar/venvs/legacy-pre-lock`; restore it to its original path before
trying to use it because Python venvs are not relocatable. Old validated
releases are retained for manual rollback and must be pruned deliberately when
disk usage warrants it.

## DNS cutover (quasarassistant.com)

The domain currently points at Vercel (frontend) with the backend on Render.
Flipping DNS moves **production** to the VM, so do it as a deliberate cutover:

1. Get the VM's public IP from TACC. Run the playbook and confirm the app
   answers on `http://<ip>` (nginx serves the default names too).
2. At the DNS provider, drop the TTL on `www` + apex ahead of time (300 s).
3. Replace the Vercel records: `A` record for `quasarassistant.com` → VM IP,
   and `www` → VM IP (A record, or CNAME to the apex).
4. Issue TLS (below), then load `https://www.quasarassistant.com` and run the
   SSE check. Keep Vercel/Render alive for a day or two as instant rollback —
   reverting DNS undoes the cutover.

Google OAuth needs no change — the authorized origin stays
`https://www.quasarassistant.com`.

## TLS

Once DNS resolves to the VM:

```bash
sudo dnf install certbot python3-certbot-nginx   # or apt on Ubuntu
sudo certbot --nginx -d www.quasarassistant.com -d quasarassistant.com
```

certbot edits the nginx site in place (keeps the SSE proxy settings) and
installs auto-renewal. The frontend is already built against
`https://www.quasarassistant.com`, so no rebuild is needed after this.

## Verifying backend health and SSE

On the VM, verify the backend directly. A successful response must contain
both HTTP 200 and `"agent_loaded": true`:

```bash
curl --fail --silent http://127.0.0.1:8000/health \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("agent_loaded") is True; print(d)'
```

The public nginx proxy preserves `/api/` URIs, while the backend health route
is `/health`; `/api/health` is therefore not a valid public health check under
the current unchanged proxy configuration.

Then send a chat from the UI through nginx and confirm tokens arrive
incrementally (not one burst at the end). If a response stalls exactly at 60 s,
some proxy in the path is still buffering — the settings live in
`templates/nginx-quasar.conf.j2`.
