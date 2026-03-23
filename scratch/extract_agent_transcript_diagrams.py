"""
Scan Cursor agent transcript JSONL files for text-based diagrams (ASCII trees/boxes,
arrow pipelines, Mermaid) and write a single markdown digest.

Transcripts do not include model id in JSON; mtime is used as an approximate date.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

TRANSCRIPTS_ROOT = Path(
    r"C:\Users\spenc\.cursor\projects\c-Users-spenc-Cursor-Repos-study-query-llm\agent-transcripts"
)
OUT_PATH = Path(__file__).resolve().parent / "AGENT_TRANSCRIPT_TEXT_DIAGRAMS.md"

FENCE_RE = re.compile(r"```(?:([^\n`]*)\n)?([\s\S]*?)```", re.MULTILINE)

CURSOR_LINE_REF_RE = re.compile(r"^\d+:\d+:")
CODE_FENCE_LANGS = frozenset(
    {
        "python",
        "py",
        "javascript",
        "js",
        "typescript",
        "ts",
        "tsx",
        "jsx",
        "rust",
        "go",
        "java",
        "c",
        "cpp",
        "csharp",
        "cs",
        "sql",
        "bash",
        "sh",
        "zsh",
        "powershell",
        "ps1",
        "json",
        "yaml",
        "yml",
        "toml",
        "html",
        "css",
        "dockerfile",
    }
)

def _assistant_text_chunks(record: dict) -> list[str]:
    if record.get("role") != "assistant":
        return []
    msg = record.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    out: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "text":
            continue
        t = part.get("text")
        if isinstance(t, str) and t.strip():
            out.append(t)
    return out


def _has_unicode_structure(body: str) -> bool:
    return (
        "├──" in body
        or "└──" in body
        or ("┌" in body and "─" in body and "│" in body)
        or (body.count("│") >= 3 and "\n" in body)
    )


def _is_diagram_block(lang: str | None, body: str) -> bool:
    b = body.strip()
    if not b:
        return False
    first = b.split("\n", 1)[0].strip().lower()
    raw_lang = (lang or "").strip()
    lang_l = raw_lang.lower()
    if CURSOR_LINE_REF_RE.match(raw_lang):
        return False
    if lang_l == "mermaid" or first.startswith("flowchart") or first.startswith("graph "):
        return True
    if "mermaid" in lang_l:
        return True
    if lang_l in CODE_FENCE_LANGS and not _has_unicode_structure(body):
        return False
    if _has_unicode_structure(body):
        return True
    # Arrow pipeline (short blocks only); avoid Python type hints (-> None)
    if "→" in body:
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        arrow_lines = sum(1 for ln in lines if "→" in ln)
        if len(body) <= 800 and arrow_lines >= 1 and len(lines) <= 12:
            return True
    if "->" in body and ") ->" not in body and "-> None" not in body:
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        arrow_lines = sum(1 for ln in lines if "->" in ln)
        if len(body) <= 500 and arrow_lines >= 2 and len(lines) <= 12:
            return True
    # ASCII-only tree (| +-- or |   +--)
    if re.search(r"^\s*\|", body, re.MULTILINE) and re.search(r"[\|`]\s*[-+]", body):
        return True
    return False


def _conversation_label(path: Path) -> tuple[str, str]:
    """Return (conversation_id, human label including subagent if any)."""
    parts = path.parts
    try:
        idx = parts.index("agent-transcripts")
    except ValueError:
        return path.stem, path.stem
    rest = parts[idx + 1 :]
    if not rest:
        return path.stem, path.stem
    parent = rest[0]
    if len(rest) >= 3 and rest[1] == "subagents":
        sub = rest[2].replace(".jsonl", "")
        return parent, f"{parent} (subagent {sub})"
    return parent, parent


def _blurb_before(text: str, fence_start: int) -> str:
    prefix = text[:fence_start].strip()
    if not prefix:
        return "Assistant message containing a diagram."
    # Prefer last markdown heading before the fence
    lines = prefix.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i].strip()
        if ln.startswith("#"):
            return ln.lstrip("#").strip()[:220]
    # Last non-empty line(s)
    tail = "\n".join(lines[-4:]).strip()
    one = " ".join(tail.split())
    return (one[:220] + "…") if len(one) > 220 else one


def iter_diagrams():
    if not TRANSCRIPTS_ROOT.is_dir():
        return
    files = sorted(TRANSCRIPTS_ROOT.rglob("*.jsonl"))
    for fp in files:
        conv_id, conv_label = _conversation_label(fp)
        try:
            mtime = fp.stat().st_mtime
        except OSError:
            mtime = 0
        date_s = _dt.datetime.fromtimestamp(mtime, tz=_dt.timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC (file mtime)"
        )
        try:
            raw = fp.read_text(encoding="utf-8")
        except OSError:
            continue
        line_no = 0
        for line in raw.splitlines():
            line_no += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for chunk in _assistant_text_chunks(record):
                for m in FENCE_RE.finditer(chunk):
                    lang = m.group(1)
                    body = m.group(2) or ""
                    if not _is_diagram_block(lang, body):
                        continue
                    start = m.start()
                    blurb = _blurb_before(chunk, start)
                    lang_display = (lang or "").strip() or "(none)"
                    yield {
                        "file": str(fp),
                        "line": line_no,
                        "conversation_id": conv_id,
                        "conversation_label": conv_label,
                        "date_approx": date_s,
                        "lang_fence": lang_display,
                        "blurb": blurb,
                        "body": body.rstrip(),
                    }


def main() -> None:
    sections: list[str] = []
    sections.append("# Agent transcript text diagrams (auto-extracted)\n")
    sections.append(
        "Generated by `scratch/extract_agent_transcript_diagrams.py`. "
        "**Agent model** is not stored in these JSONL files. "
        "**Date** is the transcript file modification time (approximate).\n"
    )
    n = 0
    for item in iter_diagrams():
        n += 1
        fence = item["lang_fence"].lower()
        if fence == "mermaid" or item["body"].lstrip().lower().startswith("flowchart"):
            kind = "Mermaid diagram"
        elif "┌" in item["body"] and "─" in item["body"]:
            kind = "ASCII box / layer diagram"
        elif "├──" in item["body"] or "└──" in item["body"] or item["body"].count("│") >= 2:
            kind = "ASCII tree / hierarchy"
        elif "→" in item["body"] or "->" in item["body"]:
            kind = "Text flow / pipeline"
        else:
            kind = "Text diagram"
        sections.append(f"## Entry {n}\n")
        sections.append(f"- **Date (approx.)**: {item['date_approx']}")
        sections.append(f"- **Conversation**: `{item['conversation_label']}`")
        sections.append(f"- **Agent model**: not recorded in transcript JSON")
        sections.append(f"- **Source file**: `{item['file']}` (JSONL line ~{item['line']})")
        sections.append(f"- **Diagram kind**: {kind}")
        sections.append(f"- **Fence lang**: `{item['lang_fence']}`")
        sections.append(f"- **Descriptive sentence**: {item['blurb']}\n")
        lang = item["lang_fence"]
        if lang and lang != "(none)":
            sections.append(f"```{lang}")
        else:
            sections.append("```")
        sections.append(item["body"])
        sections.append("```\n")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {n} diagram(s) to {OUT_PATH}")


if __name__ == "__main__":
    main()
