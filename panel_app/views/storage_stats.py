"""Panel view: PostgreSQL size metrics, artifact storage config, Azure probe, backup pointers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

import panel as pn
from sqlalchemy import text

from study_query_llm.config import config, database_connection_summary, redact_database_url
from study_query_llm.db.models_v2 import CallArtifact, Group, RawCall
from study_query_llm.services.artifact_service import DEFAULT_ARTIFACT_DIR, ArtifactService
from study_query_llm.storage.azure_blob import AzureBlobStorageBackend
from study_query_llm.storage.factory import StorageBackendFactory
from study_query_llm.utils.logging_config import get_logger

from panel_app.helpers import get_db_connection

logger = get_logger(__name__)

_CLONE_RUNBOOK = "docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md"
_MAX_PROBE_BLOBS = 50_000
_MAX_PROBE_SECONDS = 30.0


def _is_postgres_url(url: str) -> bool:
    u = (url or "").lower()
    return "postgresql" in u or "postgres" in u


def _is_truthy(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_blob_container_name(runtime_env: str) -> str:
    """Mirror ArtifactService._resolve_blob_container for display and probe (same rules)."""
    base = (os.environ.get("AZURE_STORAGE_CONTAINER") or "artifacts").strip()
    explicit = os.environ.get(f"AZURE_STORAGE_CONTAINER_{runtime_env.upper()}")
    if explicit and explicit.strip():
        container = explicit.strip()
    elif runtime_env in ("dev", "stage", "prod"):
        container = f"{base}-{runtime_env}"
    else:
        container = base
    if not container:
        raise ValueError("Resolved empty Azure blob container name.")
    lowered = container.lower()
    allow_cross_env = _is_truthy(
        os.environ.get("ARTIFACT_ALLOW_CROSS_ENV_CONTAINER"), default=False
    )
    if runtime_env != "prod" and "prod" in lowered and not allow_cross_env:
        raise ValueError(
            f"Refusing prod-like container {container!r} in runtime {runtime_env!r}. "
            "Set ARTIFACT_ALLOW_CROSS_ENV_CONTAINER=true to override."
        )
    return container


def _fmt_bytes(num: int) -> str:
    x = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(x)} B"
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{num} B"


def _section_db_markdown() -> str:
    url = config.database.connection_string
    summary = database_connection_summary(url)
    redacted = redact_database_url(url)
    lines = [
        "### Database",
        "",
        f"**Connection (redacted):** `{redacted}`  ",
        f"**Summary:** {summary}  ",
        "",
    ]
    if not _is_postgres_url(url):
        lines.extend(
            [
                "Catalog size queries (`pg_database_size`, relation sizes) require **PostgreSQL**. "
                "For SQLite or other engines, use native tools.",
                "",
            ]
        )
        return "\n".join(lines)

    try:
        db = get_db_connection()
    except Exception as exc:
        logger.exception("Storage stats: DB connection failed")
        return "\n".join(
            lines
            + [
                "**Status:** error  ",
                "",
                f"```\n{exc!s}\n```",
                "",
            ]
        )

    try:
        with db.session_scope() as session:
            n_groups = session.query(Group).count()
            n_raw = session.query(RawCall).count()
            n_artifacts = session.query(CallArtifact).count()

        lines.extend(
            [
                "**Status:** connected  ",
                f"**Groups:** {n_groups:,} &nbsp;·&nbsp; **Raw calls:** {n_raw:,} &nbsp;·&nbsp; "
                f"**Call artifacts (rows):** {n_artifacts:,}  ",
                "",
            ]
        )

        with db.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT pg_size_pretty(pg_database_size(current_database())) AS pretty, "
                    "pg_database_size(current_database()) AS bytes"
                )
            ).fetchone()
            if row:
                lines.append(f"**Database size:** {row[0]} (`{int(row[1]):,}` bytes)  ")
                lines.append("")

            rel_sql = text(
                """
                SELECT n.nspname AS schema_name,
                       c.relname AS relation_name,
                       CASE c.relkind
                           WHEN 'r' THEN 'table'
                           WHEN 'i' THEN 'index'
                           WHEN 'm' THEN 'materialized_view'
                           ELSE c.relkind::text
                       END AS kind,
                       pg_size_pretty(pg_total_relation_size(c.oid)) AS total_pretty,
                       pg_total_relation_size(c.oid) AS total_bytes
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
                  AND c.relkind IN ('r', 'i', 'm')
                ORDER BY pg_total_relation_size(c.oid) DESC
                LIMIT 25
                """
            )
            rel_rows = conn.execute(rel_sql).fetchall()

        if rel_rows:
            lines.append("**Largest relations (tables, indexes, matviews):**  ")
            lines.append("")
            lines.append("| schema | relation | kind | total |")
            lines.append("| --- | --- | --- | --- |")
            for r in rel_rows:
                lines.append(
                    f"| `{r[0]}` | `{r[1]}` | {r[2]} | {r[3]} |"
                )
            lines.append("")
    except Exception as exc:
        logger.exception("Storage stats: Postgres metrics failed")
        lines.extend(
            [
                "**Metrics:** error  ",
                "",
                f"```\n{exc!s}\n```",
                "",
            ]
        )

    return "\n".join(lines)


def _section_artifact_markdown() -> str:
    runtime_env = (os.environ.get("ARTIFACT_RUNTIME_ENV") or "dev").strip().lower()
    backend_env = (os.environ.get("ARTIFACT_STORAGE_BACKEND") or "local").strip().lower()
    strict = _is_truthy(os.environ.get("ARTIFACT_STORAGE_STRICT_MODE"))
    if runtime_env in {"stage", "prod"}:
        strict = True

    lines = [
        "### Artifact storage (configuration)",
        "",
        f"- `ARTIFACT_STORAGE_BACKEND`: `{backend_env}`",
        f"- `ARTIFACT_RUNTIME_ENV`: `{runtime_env}`",
        f"- `ARTIFACT_STORAGE_STRICT_MODE`: `{strict}` (forced true for stage/prod runtime)",
        f"- `ARTIFACT_AUTH_MODE`: `{(os.environ.get('ARTIFACT_AUTH_MODE') or 'connection_string').strip()}`",
        f"- `AZURE_STORAGE_CONTAINER`: `{os.environ.get('AZURE_STORAGE_CONTAINER') or 'artifacts'}`",
        f"- `AZURE_STORAGE_PREFIX`: `{os.environ.get('AZURE_STORAGE_PREFIX') or f'(defaults to runtime: {runtime_env})'}`",
        "",
    ]

    try:
        svc = ArtifactService(repository=None, artifact_dir=DEFAULT_ARTIFACT_DIR)
        st = svc.storage
        lines.append(f"- **Resolved backend:** `{st.backend_type}`  ")
        if st.backend_type == "local":
            base = Path(getattr(st, "base_dir", Path(DEFAULT_ARTIFACT_DIR))).resolve()
            exists = base.exists()
            lines.append(f"- **Local base directory (resolved):** `{base}` (exists={exists})  ")
        elif st.backend_type == "azure_blob":
            lines.append(
                f"- **Blob container:** `{getattr(st, 'container_name', '?')}`  "
            )
            prefix = (getattr(st, "_blob_prefix", "") or "").strip()
            lines.append(
                f"- **Blob prefix (logical):** `{prefix or '(none — keys at container root)'}`  "
            )
            lines.append(
                "- **Note:** Connection strings and secrets are never shown here.  "
            )
    except Exception as exc:
        lines.append(f"- **Resolved backend:** _error_: `{exc!s}`  ")

    try:
        cname = _resolve_blob_container_name(runtime_env)
        lines.append(f"- **Resolved container name (for Azure lane):** `{cname}`  ")
    except ValueError as exc:
        lines.append(f"- **Resolved container name:** _unavailable_: `{exc!s}`  ")

    lines.append("")
    return "\n".join(lines)


def _section_ops_markdown() -> str:
    local_url = os.environ.get("LOCAL_DATABASE_URL", "").strip()
    local_line = (
        f"`{redact_database_url(local_url)}`"
        if local_url
        else "_not set (`LOCAL_DATABASE_URL`)_"
    )
    return "\n".join(
        [
            "### Backups and local clone",
            "",
            f"- **`LOCAL_DATABASE_URL` (redacted):** {local_line}",
            "- **Jetstream → local clone runbook:** "
            f"`{_CLONE_RUNBOOK}` (repo root; uses `pg_migration_dumps/` for dumps).",
            "- **CLI inventory check:** `scripts/verify_db_backup_inventory.py` "
            "(run locally; Panel does not execute it).",
            "",
            "### Cache / ephemeral data",
            "",
            "Deployment metadata TTL and embedding key versioning live in application code "
            "(`ModelRegistry`, embedding helpers); there is **no single on-disk cache path** "
            "exposed for the whole app. Hugging Face / TEI Docker setups may use a host cache "
            "directory — see provider/docker docs if applicable.",
            "",
        ]
    )


def _resolve_azure_backend_for_probe() -> Tuple[Optional[AzureBlobStorageBackend], str]:
    """Return an Azure backend suitable for listing, or (None, reason)."""
    try:
        svc = ArtifactService(repository=None, artifact_dir=DEFAULT_ARTIFACT_DIR)
        st = svc.storage
        if st.backend_type == "azure_blob":
            return st, ""
    except Exception as exc:
        logger.debug("ArtifactService for probe: %s", exc)

    runtime_env = (os.environ.get("ARTIFACT_RUNTIME_ENV") or "dev").strip().lower()
    try:
        container_name = _resolve_blob_container_name(runtime_env)
    except ValueError as exc:
        return None, str(exc)

    auth_mode = (os.environ.get("ARTIFACT_AUTH_MODE") or "connection_string").strip().lower()
    try:
        backend = StorageBackendFactory.create(
            "azure_blob",
            container_name=container_name,
            auth_mode=auth_mode,
            account_url=os.environ.get("AZURE_STORAGE_ACCOUNT_URL"),
            managed_identity_client_id=os.environ.get(
                "AZURE_STORAGE_MANAGED_IDENTITY_CLIENT_ID"
            ),
            blob_prefix=(os.environ.get("AZURE_STORAGE_PREFIX") or runtime_env),
            max_retries=int(os.environ.get("AZURE_STORAGE_MAX_RETRIES") or "3"),
            retry_backoff_seconds=float(
                os.environ.get("AZURE_STORAGE_RETRY_BACKOFF_SECONDS") or "0.5"
            ),
            verify_uploads=_is_truthy(
                os.environ.get("AZURE_STORAGE_VERIFY_UPLOADS"), default=True
            ),
            runtime_env=runtime_env,
        )
    except Exception as exc:
        return (
            None,
            f"Could not build Azure client ({exc!s}). "
            "Ensure `azure-storage-blob` is installed and "
            "`AZURE_STORAGE_CONNECTION_STRING` or account URL + credential is configured.",
        )

    if not isinstance(backend, AzureBlobStorageBackend):
        return None, "Internal error: backend is not AzureBlobStorageBackend."
    return backend, ""


def _run_azure_probe() -> str:
    try:
        backend, err = _resolve_azure_backend_for_probe()
        if backend is None:
            return f"**Azure probe:** unavailable — {err}"
        stats: dict[str, Any] = backend.estimate_prefix_blob_usage(
            max_blobs=_MAX_PROBE_BLOBS,
            max_seconds=_MAX_PROBE_SECONDS,
        )
        trunc = "yes (partial sum; increase limits in code if needed)" if stats["truncated"] else "no"
        return "\n".join(
            [
                "### Azure blob probe (capped)",
                "",
                f"- **List prefix:** `{stats['list_prefix']}`",
                f"- **Blobs counted:** {stats['blob_count']:,}",
                f"- **Sum of sizes (counted blobs only):** {_fmt_bytes(int(stats['total_bytes']))} (`{int(stats['total_bytes']):,}` bytes)",
                f"- **Truncated / stopped early:** {trunc}",
                f"- **Elapsed:** {stats['elapsed_seconds']}s (caps: {_MAX_PROBE_BLOBS} blobs, {_MAX_PROBE_SECONDS}s)",
                "",
                "_Full container totals may be higher if the probe stopped early._",
                "",
            ]
        )
    except Exception as exc:
        logger.exception("Azure probe failed")
        return f"**Azure probe:** error — `{exc!s}`"


def create_storage_stats_ui() -> pn.viewable.Viewable:
    """Build Storage / DB stats tab."""
    refresh = pn.widgets.Button(name="Refresh", button_type="primary", width=100)
    probe_btn = pn.widgets.Button(
        name="Probe Azure usage (capped)",
        button_type="default",
        width=220,
    )

    db_pane = pn.pane.Markdown("", sizing_mode="stretch_width")
    artifact_pane = pn.pane.Markdown("", sizing_mode="stretch_width")
    ops_pane = pn.pane.Markdown("", sizing_mode="stretch_width")
    probe_pane = pn.pane.Markdown("", sizing_mode="stretch_width")

    def refresh_all(event=None):
        db_pane.object = _section_db_markdown()
        artifact_pane.object = _section_artifact_markdown()
        ops_pane.object = _section_ops_markdown()

    def do_probe(event=None):
        probe_pane.object = _run_azure_probe()

    refresh.on_click(refresh_all)
    probe_btn.on_click(do_probe)

    refresh_all()
    probe_pane.object = (
        "_Click **Probe Azure usage (capped)** to sum blob sizes under the configured prefix._"
    )

    return pn.Column(
        pn.pane.Markdown("## Storage and database stats"),
        pn.Row(refresh, pn.Spacer()),
        db_pane,
        pn.layout.Divider(),
        artifact_pane,
        pn.layout.Divider(),
        ops_pane,
        pn.layout.Divider(),
        pn.pane.Markdown("### Optional: Azure blob listing"),
        pn.Row(probe_btn, pn.Spacer()),
        probe_pane,
        sizing_mode="stretch_width",
    )
