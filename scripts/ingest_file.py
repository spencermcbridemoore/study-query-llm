#!/usr/bin/env python3
"""Generic file ingestion wrapper for method-execution data runners."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import default_write_intent_for_connection
from study_query_llm.services.method_execution_service import MethodExecutionService
from study_query_llm.services.method_runtime_registry import (
    ensure_default_method_runtimes_registered,
    get_method_runtime,
)
from study_query_llm.services.method_service import MethodService


def _slug(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return token or "value"


def _parse_expected_columns(raw: Optional[str]) -> Optional[list[str]]:
    if raw is None:
        return None
    parts = [part.strip() for part in str(raw).split(",")]
    cleaned = [part for part in parts if part]
    return cleaned or None


def _resolve_database_url() -> str:
    return (os.environ.get("DATABASE_URL") or "").strip()


def _compute_sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _assert_method_registered(
    *,
    method_service: MethodService,
    method_name: str,
    method_version: str,
) -> None:
    row = method_service.get_method(method_name, version=method_version)
    if row is None:
        raise ValueError(
            f"method_not_registered:{method_name}@{method_version}; "
            "run the appropriate scripts/register_*_methods.py registrar first."
        )
    runtime = get_method_runtime(method_name, method_version)
    if runtime is None:
        raise ValueError(
            f"runtime_not_registered:{method_name}@{method_version}; "
            "verify method_runtime_registry defaults include this method."
        )


def _resolve_request_group_id(
    *,
    repo: RawCallRepository,
    request_group_id: Optional[int],
    dry_run: bool,
    dataset_name: str,
    dataset_version: str,
) -> Optional[int]:
    if request_group_id is not None:
        group = repo.get_group_by_id(int(request_group_id))
        if group is None:
            raise ValueError(f"request_group_not_found:{request_group_id}")
        return int(request_group_id)
    if dry_run:
        return None
    return int(
        repo.create_group(
            group_type="analysis_request",
            name=f"file_ingest:{_slug(dataset_name)}:{_slug(dataset_version)}",
            description="Method-execution ingestion request group",
            metadata_json={
                "dataset_name": dataset_name,
                "dataset_version": dataset_version,
                "ingestion_surface": "scripts/ingest_file.py",
            },
        )
    )


async def _execute_ingestion(
    *,
    service: MethodExecutionService,
    request_group_id: int,
    base_run_key: str,
    file_path: Path,
    dataset_name: str,
    dataset_version: str,
    content_type: str,
    expected_sha256: str,
    parse_as_csv: bool,
    expected_columns: Optional[list[str]],
) -> tuple[int, Optional[int], dict]:
    source_outcome = await service.execute(
        request_group_id=int(request_group_id),
        base_run_key=base_run_key,
        method_name="file_artifact.basic",
        method_version="0.1",
        parameters={
            "file_path": str(file_path),
            "registered_name": dataset_name,
            "registered_version": dataset_version,
            "content_type": content_type,
            "expected_sha256": expected_sha256,
        },
    )

    imported_run_id: Optional[int] = None
    parse_context: dict = {}
    if parse_as_csv:
        parse_params: dict = {
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
        }
        if expected_columns is not None:
            parse_params["expected_columns"] = expected_columns
        imported_outcome = await service.execute(
            request_group_id=int(request_group_id),
            base_run_key=base_run_key,
            method_name="csv_parse.basic",
            method_version="0.1",
            parameters=parse_params,
            imported_run_id=int(source_outcome.run_id),
        )
        imported_run_id = int(imported_outcome.run_id)
        parse_context = dict(imported_outcome.pipeline_stage_context or {})

    return int(source_outcome.run_id), imported_run_id, parse_context


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=(
            "Ingest one file as a source artifact and optionally parse CSV into "
            "an imported parquet artifact."
        )
    )
    parser.add_argument("--path", required=True, help="Input file path.")
    parser.add_argument("--name", required=True, help="Dataset name label.")
    parser.add_argument("--version", required=True, help="Dataset version label.")
    parser.add_argument("--content-type", required=True, help="Input file MIME type.")
    parser.add_argument(
        "--parse-as",
        choices=["csv"],
        default=None,
        help="Optional transform chain. Supported: csv.",
    )
    parser.add_argument(
        "--expected-columns",
        default=None,
        help="Optional comma-separated expected column names for CSV strict mode.",
    )
    parser.add_argument(
        "--request-group-id",
        type=int,
        default=None,
        help="Optional existing request group id. If omitted, one is created.",
    )
    parser.add_argument(
        "--base-run-key",
        default=None,
        help="Optional base run key. Default: ingest__<name>@<version>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup and print planned invocation(s) without writes.",
    )
    args = parser.parse_args(argv)

    file_path = Path(args.path).expanduser().resolve()
    if not file_path.exists():
        print(f"ERROR: file does not exist: {file_path}", file=sys.stderr)
        return 1
    if not file_path.is_file():
        print(f"ERROR: path is not a file: {file_path}", file=sys.stderr)
        return 1

    expected_columns = _parse_expected_columns(args.expected_columns)
    expected_sha256 = _compute_sha256(file_path)
    base_run_key = args.base_run_key or f"ingest__{_slug(args.name)}@{_slug(args.version)}"

    db_url = _resolve_database_url()
    if not db_url:
        print("ERROR: DATABASE_URL not set.", file=sys.stderr)
        return 1

    db = DatabaseConnectionV2(
        db_url,
        enable_pgvector=False,
        write_intent=default_write_intent_for_connection(db_url),
    )
    if not args.dry_run:
        db.init_db()

    ensure_default_method_runtimes_registered()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_service = MethodService(repo)

        try:
            _assert_method_registered(
                method_service=method_service,
                method_name="file_artifact.basic",
                method_version="0.1",
            )
            if args.parse_as == "csv":
                _assert_method_registered(
                    method_service=method_service,
                    method_name="csv_parse.basic",
                    method_version="0.1",
                )
            resolved_request_group_id = _resolve_request_group_id(
                repo=repo,
                request_group_id=args.request_group_id,
                dry_run=bool(args.dry_run),
                dataset_name=str(args.name),
                dataset_version=str(args.version),
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"ERROR: ingestion_setup_validation_failed:{exc}", file=sys.stderr)
            return 1

        if args.dry_run:
            print("dry-run: no writes performed")
            print(f"file_path={file_path}")
            print(f"dataset_name={args.name}")
            print(f"dataset_version={args.version}")
            print(f"content_type={args.content_type}")
            print(f"expected_sha256={expected_sha256}")
            print(f"request_group_id={resolved_request_group_id or '<new-group-on-apply>'}")
            print(f"base_run_key={base_run_key}")
            print("planned_calls:")
            print("  - file_artifact.basic@0.1")
            if args.parse_as == "csv":
                print("  - csv_parse.basic@0.1 (imported_run_id=<source_run_id>)")
                if expected_columns is not None:
                    print(f"    expected_columns={expected_columns}")
            return 0

        if resolved_request_group_id is None:
            print("ERROR: request_group_id resolution failed.", file=sys.stderr)
            return 1

        service = MethodExecutionService(repo)
        try:
            source_run_id, imported_run_id, parse_context = asyncio.run(
                _execute_ingestion(
                    service=service,
                    request_group_id=int(resolved_request_group_id),
                    base_run_key=base_run_key,
                    file_path=file_path,
                    dataset_name=str(args.name),
                    dataset_version=str(args.version),
                    content_type=str(args.content_type),
                    expected_sha256=expected_sha256,
                    parse_as_csv=(args.parse_as == "csv"),
                    expected_columns=expected_columns,
                )
            )
        except Exception as exc:
            print(f"ERROR: ingestion_failed:{exc}", file=sys.stderr)
            return 1

        print(f"request_group_id={resolved_request_group_id}")
        print(f"base_run_key={base_run_key}")
        print(f"dataset_name={args.name}")
        print(f"sha256={expected_sha256}")
        print(f"source_run_id={source_run_id}")
        if imported_run_id is not None:
            print(f"imported_run_id={imported_run_id}")
            print(
                "schema_summary="
                f"row_count={parse_context.get('row_count')},"
                f"column_count={parse_context.get('column_count')}"
            )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

