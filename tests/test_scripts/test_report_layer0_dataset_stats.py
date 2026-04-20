"""Unit tests for report_layer0_dataset_stats helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "report_layer0_dataset_stats.py"


@pytest.fixture(scope="module")
def stats_mod():
    spec = importlib.util.spec_from_file_location("report_layer0_dataset_stats", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_csv_stats_counts_rows(stats_mod):
    data = b"a,b,c\n1,2,3\n4,5,6\n"
    st = stats_mod._csv_stats(data)
    assert st["rows"] == 3
    assert st["cols_max"] == 3


def test_line_stats_non_empty(stats_mod):
    data = b"hello\n\nworld\n"
    st = stats_mod._line_stats(data)
    assert st["non_empty_lines"] == 2
    assert st["max_line_chars"] == 5


def test_xlsx_row_count_minimal(stats_mod):
    # Minimal valid .xlsx (zip) with one sheet row tag
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            b'<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            b'<Default Extension="xml" ContentType="application/xml"/></Types>',
        )
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            b'<?xml version="1.0"?><worksheet><sheetData><row r="1"></row><row r="2"></row></sheetData></worksheet>',
        )
    n = stats_mod._xlsx_sheet1_row_count(buf.getvalue())
    assert n == 2


def test_main_write_doc_flag_is_deprecated_and_short_circuits(stats_mod, monkeypatch, capsys):
    def _unexpected_slug_stats(_slug):
        raise AssertionError("_slug_stats should not be called for --write-doc")

    monkeypatch.setattr(stats_mod, "_slug_stats", _unexpected_slug_stats)
    monkeypatch.setattr(
        stats_mod.sys,
        "argv",
        ["report_layer0_dataset_stats.py", "--write-doc"],
    )

    rc = stats_mod.main()
    out = capsys.readouterr()

    assert rc == 2
    assert "--write-doc is no longer supported" in out.err
