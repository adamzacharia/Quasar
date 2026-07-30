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
- VM OS: Ubuntu 24.04 or Rocky/RHEL 9, with an SSH user that can `sudo`.
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
code update later (git pull → pip → npm build → restart, no system changes):

```bash
ansible-playbook -i inventory.ini quasar.yml --ask-vault-pass --tags deploy
```

## What it installs

| Piece | Detail |
|---|---|
| System | git, nginx, build tools, Python 3.12, Node 22 (NodeSource) |
| User/layout | `quasar` system user; repo at `/opt/quasar/app`, venv at `/opt/quasar/venv` |
| Backend | `quasar-backend.service` → `venv/bin/python launch.py` from `ui-pro/` (CWD matters: DiskCache paths are CWD-relative), `.env` rendered at repo root, journald logs |
| Frontend | `quasar-frontend.service` → `next start` on 127.0.0.1:3001 after `npm ci && npm run build` |
| nginx | `/api/` → :8000 with SSE settings; `/` → :3001; 200 MB upload cap |

Logs: `journalctl -u quasar-backend -f` / `-u quasar-frontend -f`.

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

## Verifying SSE survives the proxy

```bash
curl -N https://<host>/api/health
```

then send a chat from the UI and confirm tokens arrive incrementally (not one
burst at the end). If a response stalls exactly at 60 s, some proxy in the path
is still buffering — the settings live in `templates/nginx-quasar.conf.j2`.
