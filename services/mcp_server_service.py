import json
import os
import shlex
from datetime import datetime, timezone
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
    name: str              # e.g., "manna" or "github"
    # "stdio" | "http" (legacy SSE) | "streamable_http" (modern MCP HTTP)
    transport: str = "stdio"
    # Stdio args
    command: Optional[str] = None  # e.g., "npx" or "uvx"
    args: List[str] = []   # e.g., ["-y", "@modelcontextprotocol/server-sqlite", "--db", "test.db"]
    # SSE args
    url: Optional[str] = None # e.g. "https://huggingface.co/mcp"

    env: Dict[str, str] = {}    # e.g., {"GITHUB_TOKEN": "ghp_..."}
    # HTTP transports only: request headers, e.g. {"Authorization": "Bearer ..."}
    # for hosted servers that need an API key (env only reaches stdio).
    headers: Dict[str, str] = {}


# The complete transport roster. Anything else is rejected up front: the mount
# path treats every non-HTTP transport as stdio (command execution), so an
# unrecognized string must never reach it (CX-01).
KNOWN_TRANSPORTS = ("stdio", "http", "streamable_http")


def normalize_transport(value: Optional[str]) -> str:
    return (value or "stdio").strip().lower()


def validate_server_config(cfg: MCPServerConfig) -> MCPServerConfig:
    """Shared structural validation for user-saved AND platform configs.

    Normalizes ``transport`` in place and raises ``ValueError`` on a config
    the mount path could not start (or would start as something the author
    did not intend).
    """
    # Normalize, don't just check: a validated config must be canonical, or
    # " manna " becomes a noncanonical tool namespace and " uvx " an
    # unlaunchable executable downstream (CX-25).
    cfg.name = (cfg.name or "").strip()
    cfg.command = (cfg.command or "").strip() or None
    cfg.url = (cfg.url or "").strip() or None
    if not cfg.name:
        raise ValueError("Server name is required.")
    cfg.transport = normalize_transport(cfg.transport)
    if cfg.transport not in KNOWN_TRANSPORTS:
        raise ValueError(
            f"Unknown MCP transport {cfg.transport!r}; expected one of "
            f"{', '.join(KNOWN_TRANSPORTS)}."
        )
    if cfg.transport == "stdio" and not cfg.command:
        raise ValueError("Server command is required for stdio transport.")
    if cfg.transport in ("http", "streamable_http") and not cfg.url:
        raise ValueError("Server URL is required for http transports.")
    return cfg


def manna_enabled() -> bool:
    """Whether the MANNA MCP server (NSF-Simons CosmicAI's IVOA archive server)
    is mounted process-wide at agent startup.

    OFF by default: MANNA's tools overlap Quasar's native ALMA/Data Lab
    families, and the stdio default spawns a child process (``uvx``) whose
    dependency tree (astropy/pyvo) roughly doubles the Python footprint — do
    not enable stdio mode on a small host (e.g. the 2 GB Render box); point
    ``MANNA_MCP_URL`` at a remote server instead.
    """
    return os.getenv("QUASAR_ENABLE_MANNA", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


# Spawns MANNA from PyPI on demand; --stdio keeps it off Quasar's HTTP port
# (MANNA's HTTP default is :8000, which collides with the backend). Pinned to
# the release this integration was verified against — bump deliberately, not
# whenever upstream publishes.
_MANNA_DEFAULT_COMMAND = "uvx --from manna-mcp==0.9.0 manna --stdio"

# The MANNA server every account starts with in Settings > MCP Servers (the
# project's own hosted MANNA, manna 0.9.0, no key needed). QUASAR_DEFAULT_MANNA_URL
# points it elsewhere; "off" (or 0 / false / none / empty) turns the default off.
_DEFAULT_MANNA_URL = "https://48a9-129-114-35-124.ngrok-free.app/manna/mcp"
_OFF_WORDS = {"", "0", "off", "false", "no", "none"}


def default_mcp_servers() -> List[MCPServerConfig]:
    """Servers added to every account's own list the first time it is loaded.

    They are ordinary per-user rows once added: the user can test, edit or
    remove them, and a removed default is never added back. Skipped when the
    platform already mounts MANNA for everyone (QUASAR_ENABLE_MANNA), so the
    same manna__* tool names are never offered twice in one turn.
    """
    raw = os.getenv("QUASAR_DEFAULT_MANNA_URL")
    url = _DEFAULT_MANNA_URL if raw is None else raw.strip()
    if url.lower() in _OFF_WORDS or manna_enabled():
        return []
    try:
        return [validate_server_config(MCPServerConfig(name="manna", transport="streamable_http", url=url))]
    except ValueError as e:
        print(f"[MCPServerService] Ignoring QUASAR_DEFAULT_MANNA_URL: {e}")
        return []


def _split_command(command_line: str) -> List[str]:
    """Split a command line into argv, keeping Windows paths intact.

    POSIX shlex eats backslashes (``C:\\uv\\uvx.exe`` → ``C:uvuvx.exe``), so on
    Windows split in non-POSIX mode and strip the quote characters it leaves
    on quoted tokens.
    """
    if os.name == "nt":
        return [tok.strip('"') for tok in shlex.split(command_line, posix=False)]
    return shlex.split(command_line)


def platform_mcp_servers() -> List[dict]:
    """Operator-level (process-wide) MCP server configs, from environment.

    Unlike the per-user configs persisted by :class:`MCPServerService` (which
    come from the web UI and are RCE-gated by ``mcp_stdio_enabled``), these are
    set by whoever controls the deployment's environment — someone who can
    already run arbitrary commands — so stdio entries are honored as-is.

    Sources, in order:
    - ``QUASAR_ENABLE_MANNA``: mounts MANNA under the ``manna`` namespace.
      ``MANNA_MCP_URL`` selects a remote streamable-HTTP server; otherwise
      ``MANNA_MCP_COMMAND`` (default: pinned ``uvx --from manna-mcp==...``)
      is spawned over stdio.
    - ``QUASAR_PLATFORM_MCP_SERVERS``: JSON list of MCPServerConfig objects
      for any additional servers.

    Robustness contract: one bad entry (or a bad MANNA command) is logged and
    skipped without suppressing the other servers, and duplicate names keep
    the first occurrence — concurrent bridges must never race to register the
    same ``{name}__*`` tools.
    """
    servers: List[dict] = []

    def _add(cfg: MCPServerConfig, source: str) -> None:
        try:
            validate_server_config(cfg)
        except ValueError as e:
            print(f"[MCPServerService] Skipping {source} entry: {e}")
            return
        if any(s["name"] == cfg.name for s in servers):
            print(
                f"[MCPServerService] Skipping duplicate platform MCP server "
                f"name {cfg.name!r} from {source} (first definition wins)."
            )
            return
        servers.append(cfg.model_dump())

    if manna_enabled():
        try:
            url = os.getenv("MANNA_MCP_URL", "").strip()
            if url:
                # MANNA serves streamable HTTP at /mcp (SSE is its legacy path).
                _add(
                    MCPServerConfig(name="manna", transport="streamable_http", url=url),
                    "QUASAR_ENABLE_MANNA",
                )
            else:
                argv = _split_command(
                    os.getenv("MANNA_MCP_COMMAND", "").strip() or _MANNA_DEFAULT_COMMAND
                )
                _add(
                    MCPServerConfig(
                        name="manna",
                        transport="stdio",
                        command=argv[0] if argv else "",
                        args=argv[1:],
                    ),
                    "QUASAR_ENABLE_MANNA",
                )
        except Exception as e:
            # e.g. unbalanced quoting in MANNA_MCP_COMMAND — never let the
            # MANNA block suppress the generic platform servers below.
            print(f"[MCPServerService] Skipping MANNA config: {e}")

    raw = os.getenv("QUASAR_PLATFORM_MCP_SERVERS", "").strip()
    if raw:
        try:
            entries = json.loads(raw)
            if not isinstance(entries, list):
                raise ValueError("expected a JSON list")
        except Exception as e:
            # Unparseable JSON: nothing salvageable, log and move on.
            print(f"[MCPServerService] Ignoring QUASAR_PLATFORM_MCP_SERVERS: {e}")
            entries = []
        for i, entry in enumerate(entries):
            try:
                cfg = MCPServerConfig(**entry)
            except Exception as e:
                print(
                    f"[MCPServerService] Skipping QUASAR_PLATFORM_MCP_SERVERS "
                    f"entry {i}: {e}"
                )
                continue
            _add(cfg, f"QUASAR_PLATFORM_MCP_SERVERS[{i}]")
    return servers


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LEGACY_DIR = "user_tools"
# Server ids saved via the web UI are only ever "http" / "streamable_http"
# unless the operator enabled stdio; "oauth" marks a server whose bearer token
# comes from services/mcp_oauth_store.py rather than from saved headers.
KNOWN_AUTH_KINDS = ("none", "oauth")
_SCHEMA_READY: set = set()
_MARKS_READY: set = set()
# (database, user, default names) already seeded by this process, so a chat
# turn does not pay the seed queries again (each is a round trip on Turso).
# Safe to memoize: seeding only ever adds a default once, and the URL a seeded
# row uses is resolved per load (_load_rows), not written here.
_DEFAULTS_SEEDED: set = set()


def mcp_db_path(explicit: Optional[str] = None) -> str:
    """Local SQLite file for the MCP tables (ignored when Turso is configured,
    see services.db.get_connection). ``QUASAR_MCP_DB_PATH`` overrides it."""
    if explicit:
        return explicit
    env = os.getenv("QUASAR_MCP_DB_PATH", "").strip()
    if env:
        return env
    return str(_REPO_ROOT / "data" / "user_mcp.db")


class MCPServerService:
    """Per-user external MCP server configurations, stored in the database.

    Configs used to be plain JSON files at ``user_tools/<user_id>/mcp_servers.json``
    with API-key headers in clear text, on a disk that does not survive a
    Render redeploy. They now live in the ``user_mcp_servers`` table (Turso in
    production) with ``env`` and ``headers`` values Fernet-encrypted under the
    same key as user provider keys. A legacy file is imported the first time
    its user's servers are loaded and then renamed, never deleted.

    The public API (load_servers / save_server / delete_server returning plain
    dicts) is unchanged, so callers need no changes. ``load_servers`` returns
    DECRYPTED values: never send its output to a browser unmasked.
    """

    def __init__(self, base_dir: Optional[str] = None, db_path: Optional[str] = None):
        # base_dir is now only where legacy JSON files are looked for. A test
        # that passes its own base_dir also gets a private database there.
        self.base_dir = Path(base_dir or _DEFAULT_LEGACY_DIR)
        if db_path is None and base_dir is not None:
            db_path = str(self.base_dir / "user_mcp.db")
        self._db_path = mcp_db_path(db_path)
        self._init_db()

    # ── storage ──────────────────────────────────────────────────────────
    def _conn(self):
        from services.db import get_connection, is_using_turso

        if not is_using_turso():
            os.makedirs(os.path.dirname(os.path.abspath(self._db_path)), exist_ok=True)
        return get_connection(self._db_path)

    def _init_db(self) -> None:
        from services.db import is_using_turso

        # The pool builds a service per chat turn; run the DDL once per
        # database, not once per turn (each is a round trip on Turso).
        key = ("turso",) if is_using_turso() else ("sqlite", os.path.abspath(self._db_path))
        if key in _SCHEMA_READY and (key[0] == "turso" or os.path.exists(key[1])):
            return
        conn = self._conn()
        try:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS user_mcp_servers (
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    transport TEXT NOT NULL,
                    url TEXT,
                    command TEXT,
                    args_json TEXT NOT NULL DEFAULT '[]',
                    secrets_enc TEXT,
                    auth TEXT NOT NULL DEFAULT 'none',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, name)
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_mcp_servers_user ON user_mcp_servers(user_id)"
            )
            conn.commit()
        finally:
            conn.close()
        _SCHEMA_READY.add(key)

    def _marks_ready(self) -> bool:
        """Create the default-server marks table if needed; False if it cannot
        be created (default servers are then skipped, saved ones still load).

        One row per default server ever offered to a user. It outlives the
        server row, which is what keeps a removed default from coming back.
        ``row_stamp`` is set only when the seed itself inserted the server row:
        it is that row's ``updated_at``, so the row is "never edited" exactly
        while the two still match (every save or edit rewrites updated_at).
        """
        from services.db import is_using_turso

        key = ("turso",) if is_using_turso() else ("sqlite", os.path.abspath(self._db_path))
        if key in _MARKS_READY:
            return True
        try:
            conn = self._conn()
            try:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS user_mcp_default_marks (
                        user_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        row_stamp TEXT,
                        marked_at TEXT NOT NULL,
                        PRIMARY KEY (user_id, name)
                    )"""
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:  # noqa: BLE001 - optional table, never fatal
            print(f"[MCPServerService] Default MCP servers disabled (marks table unavailable): {e}")
            return False
        _MARKS_READY.add(key)
        return True

    @staticmethod
    def _encrypt_secrets(env: Dict[str, str], headers: Dict[str, str]) -> Optional[str]:
        from services.provider_key_service import encrypt_secret

        if not env and not headers:
            return None
        return encrypt_secret(json.dumps({"env": env or {}, "headers": headers or {}}))

    @staticmethod
    def _decrypt_secrets(blob: Optional[str]) -> Dict[str, Dict[str, str]]:
        from services.provider_key_service import decrypt_secret

        if not blob:
            return {"env": {}, "headers": {}}
        data = json.loads(decrypt_secret(blob))
        return {"env": dict(data.get("env") or {}), "headers": dict(data.get("headers") or {})}

    def _row_to_dict(self, row) -> Optional[dict]:
        name, transport, url, command, args_json, secrets_enc, auth, updated_at = row
        try:
            secrets = self._decrypt_secrets(secrets_enc)
        except Exception as e:  # noqa: BLE001 - one bad row must not hide the rest
            print(f"[MCPServerService] Could not decrypt secrets for server {name!r}: {type(e).__name__}")
            secrets = {"env": {}, "headers": {}}
        try:
            args = json.loads(args_json or "[]")
        except ValueError:
            args = []
        cfg = MCPServerConfig(
            name=name, transport=transport, url=url, command=command,
            args=list(args) if isinstance(args, list) else [],
            env=secrets["env"], headers=secrets["headers"],
        ).model_dump()
        cfg["auth"] = auth if auth in KNOWN_AUTH_KINDS else "none"
        # Opaque version of this row (microsecond write time), for
        # compare-and-set updates (restore_if_revision).
        cfg["revision"] = updated_at
        return cfg

    # ── legacy JSON import ───────────────────────────────────────────────
    def _legacy_file(self, user_id: str) -> Optional[Path]:
        uid = str(user_id or "")
        # A user id is a path segment here: refuse anything that could walk
        # out of base_dir.
        if not uid or uid in (".", "..") or any(c in uid for c in "/\\:"):
            return None
        return self.base_dir / uid / "mcp_servers.json"

    def _migrate_legacy(self, user_id: str) -> None:
        legacy = self._legacy_file(user_id)
        if legacy is None or not legacy.is_file():
            return
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"[MCPServerService] Legacy MCP file for {user_id} is unreadable, left in place: {e}")
            return
        imported = 0
        existing = {s["name"] for s in self._load_rows(user_id)}
        for entry in entries if isinstance(entries, list) else []:
            try:
                cfg = validate_server_config(MCPServerConfig(**entry))
            except Exception as e:  # noqa: BLE001 - skip a bad entry, keep the rest
                print(f"[MCPServerService] Skipping legacy MCP entry for {user_id}: {e}")
                continue
            if cfg.name in existing:
                continue
            # Imported as stored: the stdio gate is applied again at connect
            # time (services/user_mcp.py), so importing never spawns anything.
            self._write(user_id, cfg, auth="none")
            existing.add(cfg.name)
            imported += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            legacy.rename(legacy.with_name(f"mcp_servers.json.migrated-{stamp}"))
        except OSError as e:
            # Rows are keyed by name and existing names are skipped, so a
            # failed rename only means the import is re-checked next time.
            print(f"[MCPServerService] Could not rename legacy MCP file for {user_id}: {e}")
        print(f"[MCPServerService] Imported {imported} legacy MCP server(s) for {user_id} into the database")

    # ── public API ───────────────────────────────────────────────────────
    def _load_rows(self, user_id: str) -> List[dict]:
        defaults = {c.name: c for c in default_mcp_servers()}
        follow = bool(defaults) and self._marks_ready()
        rows = None
        conn = self._conn()
        try:
            if follow:
                try:
                    rows = conn.execute(
                        """SELECT s.name, s.transport, s.url, s.command, s.args_json, s.secrets_enc, s.auth,
                                  s.updated_at, m.row_stamp
                           FROM user_mcp_servers s LEFT JOIN user_mcp_default_marks m
                             ON m.user_id = s.user_id AND m.name = s.name
                           WHERE s.user_id = ? ORDER BY s.created_at, s.name""",
                        (str(user_id),),
                    ).fetchall()
                except Exception as e:  # noqa: BLE001 - the optional table must never hide saved servers
                    print(f"[MCPServerService] Default MCP marks unreadable, loading without them: {e}")
            if rows is None:
                rows = conn.execute(
                    """SELECT name, transport, url, command, args_json, secrets_enc, auth, updated_at, NULL
                       FROM user_mcp_servers WHERE user_id = ? ORDER BY created_at, name""",
                    (str(user_id),),
                ).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            d = self._row_to_dict(tuple(r[:8]))
            if not d:
                continue
            # A default this account got by seeding and never edited uses the
            # CURRENT default URL, resolved here rather than written back, so
            # moving the hosted server is an env change and workers on
            # different configs never fight over the stored row.
            cfg = defaults.get(d["name"])
            if cfg is not None and r[8] is not None and r[8] == d["revision"] and d["auth"] == "none":
                d["transport"], d["url"] = cfg.transport, cfg.url
            out.append(d)
        return out

    def _seed_defaults(self, user_id: str) -> None:
        """Add the default servers (default_mcp_servers) to this user's list.

        Each default is offered once per user, decided inside one atomic
        write: the server row is inserted only while the user has neither a
        mark nor a row of that name, and the mark records whether this seed
        made the row. So a removed default stays removed (even against a
        request that started before the removal), and a server the user
        already saved under the same name is never touched or taken over.
        """
        from services.db import is_using_turso

        defaults = default_mcp_servers()
        if not defaults or not self._marks_ready():
            return
        db = "turso" if is_using_turso() else os.path.abspath(self._db_path)
        memo = (db, str(user_id), tuple(c.name for c in defaults))
        if memo in _DEFAULTS_SEEDED:
            return
        now = datetime.now(timezone.utc).isoformat()
        uid = str(user_id)
        conn = self._conn()
        try:
            for cfg in defaults:
                conn.execute(
                    """INSERT INTO user_mcp_servers (user_id, name, transport, url, command, args_json,
                           secrets_enc, auth, created_at, updated_at)
                       SELECT ?, ?, ?, ?, NULL, '[]', NULL, 'none', ?, ?
                       WHERE NOT EXISTS (SELECT 1 FROM user_mcp_default_marks WHERE user_id = ? AND name = ?)
                         AND NOT EXISTS (SELECT 1 FROM user_mcp_servers WHERE user_id = ? AND name = ?)""",
                    (uid, cfg.name, cfg.transport, cfg.url, now, now, uid, cfg.name, uid, cfg.name))
                # row_stamp is non-NULL only if the row above is ours (it
                # carries this seed's timestamp); an existing mark is kept.
                conn.execute(
                    """INSERT OR IGNORE INTO user_mcp_default_marks (user_id, name, row_stamp, marked_at)
                       SELECT ?, ?, (SELECT updated_at FROM user_mcp_servers
                                     WHERE user_id = ? AND name = ? AND updated_at = ? AND auth = 'none'), ?""",
                    (uid, cfg.name, uid, cfg.name, now, now))
            conn.commit()
        finally:
            conn.close()
        _DEFAULTS_SEEDED.add(memo)

    def load_servers(self, user_id: str) -> List[dict]:
        """All MCP server configs for a user, secrets decrypted."""
        if not user_id:
            return []
        try:
            self._migrate_legacy(user_id)
        except Exception as e:  # noqa: BLE001 - never block loading on the import
            print(f"[MCPServerService] Legacy MCP import failed for {user_id}: {e}")
        try:
            self._seed_defaults(user_id)
        except Exception as e:  # noqa: BLE001 - a default must never hide saved servers
            print(f"[MCPServerService] Default MCP servers not added for {user_id}: {e}")
        try:
            return self._load_rows(user_id)
        except Exception as e:  # noqa: BLE001
            print(f"[MCPServerService] Error loading servers for {user_id}: {e}")
            return []

    def get_server(self, user_id: str, name: str) -> Optional[dict]:
        return next((s for s in self.load_servers(user_id) if s.get("name") == name), None)

    def _write(self, user_id: str, cfg: MCPServerConfig, *, auth: Optional[str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        secrets_enc = self._encrypt_secrets(cfg.env, cfg.headers)
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT created_at, auth FROM user_mcp_servers WHERE user_id = ? AND name = ?",
                (str(user_id), cfg.name),
            ).fetchone()
            if auth is None:
                auth = row[1] if row else "none"
            if auth not in KNOWN_AUTH_KINDS:
                raise ValueError(f"Unknown MCP auth kind {auth!r}.")
            values = (cfg.transport, cfg.url, cfg.command, json.dumps(list(cfg.args or [])),
                      secrets_enc, auth, now)
            if row:
                conn.execute(
                    """UPDATE user_mcp_servers SET transport = ?, url = ?, command = ?,
                       args_json = ?, secrets_enc = ?, auth = ?, updated_at = ?
                       WHERE user_id = ? AND name = ?""",
                    values + (str(user_id), cfg.name),
                )
            else:
                conn.execute(
                    """INSERT INTO user_mcp_servers (transport, url, command, args_json,
                       secrets_enc, auth, updated_at, user_id, name, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    values + (str(user_id), cfg.name, now),
                )
            conn.commit()
        finally:
            conn.close()

    def save_server(self, user_id: str, server_config: MCPServerConfig,
                    *, auth: Optional[str] = None) -> dict:
        """Add or update an MCP server configuration.

        ``auth`` is "oauth" or "none"; None keeps the stored value (or "none"
        for a new server). Returns the saved config, secrets included: mask
        before returning it to a client.
        """
        if not user_id:
            raise ValueError("A signed-in user is required to save MCP servers.")
        # Structural validation (also normalizes + rejects unknown transports,
        # which the mount path would otherwise execute as stdio — CX-01).
        validate_server_config(server_config)
        if server_config.transport == "stdio" and not mcp_stdio_enabled():
            # RCE guard: refuse to persist a local-command MCP server. Use an
            # http/SSE URL instead (or enable QUASAR_ENABLE_MCP_STDIO in a
            # trusted local/dev environment). (S2)
            raise ValueError(
                "stdio MCP servers (local command spawning) are disabled on this "
                "deployment. Provide an http/SSE server URL instead."
            )
        self._migrate_legacy(user_id)
        self._write(user_id, server_config, auth=auth)
        saved = self.get_server(user_id, server_config.name)
        return saved if saved is not None else {**server_config.model_dump(), "auth": auth or "none"}

    def revision_of(self, user_id: str, name: str) -> Optional[str]:
        conn = self._conn()
        try:
            row = conn.execute("SELECT updated_at FROM user_mcp_servers WHERE user_id = ? AND name = ?",
                               (str(user_id), name)).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def restore_if_revision(self, user_id: str, revision: str, cfg: MCPServerConfig, *, auth: str) -> bool:
        """Overwrite the row with ``cfg`` only if it is still at ``revision``
        (one conditional UPDATE): a save that landed in between wins over a
        rollback computed from an older snapshot. Returns whether it wrote."""
        validate_server_config(cfg)
        if auth not in KNOWN_AUTH_KINDS:
            raise ValueError(f"Unknown MCP auth kind {auth!r}.")
        now = datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        try:
            cur = conn.execute(
                """UPDATE user_mcp_servers SET transport = ?, url = ?, command = ?, args_json = ?,
                   secrets_enc = ?, auth = ?, updated_at = ?
                   WHERE user_id = ? AND name = ? AND updated_at = ?""",
                (cfg.transport, cfg.url, cfg.command, json.dumps(list(cfg.args or [])),
                 self._encrypt_secrets(cfg.env, cfg.headers), auth, now, str(user_id), cfg.name, revision),
            )
            conn.commit()
            return getattr(cur, "rowcount", 0) == 1
        finally:
            conn.close()

    def delete_server(self, user_id: str, name: str) -> bool:
        """Delete an MCP server config by name (its OAuth rows are removed by
        the caller through services.mcp_oauth_store)."""
        if not user_id:
            return False
        self._migrate_legacy(user_id)
        conn = self._conn()
        try:
            existed = conn.execute(
                "SELECT 1 FROM user_mcp_servers WHERE user_id = ? AND name = ?",
                (str(user_id), name),
            ).fetchone()
            if not existed:
                return False
            conn.execute("DELETE FROM user_mcp_servers WHERE user_id = ? AND name = ?",
                         (str(user_id), name))
            conn.commit()
        finally:
            conn.close()
        return True


def mask_server(cfg: dict) -> dict:
    """A copy safe to send to a browser: env/header VALUES replaced, names
    kept, and credential-looking URL query values (``?api_key=...``, which
    some hosted servers use instead of a header) redacted."""
    from services.secret_redaction import redact_url

    out = {k: v for k, v in (cfg or {}).items()}
    for field in ("env", "headers"):
        if isinstance(out.get(field), dict):
            out[field] = {k: "********" for k in out[field]}
    if isinstance(out.get("url"), str) and out["url"]:
        out["url"] = redact_url(out["url"])
    return out
