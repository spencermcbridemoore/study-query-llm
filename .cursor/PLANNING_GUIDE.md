# Planning Guide - Session-Aware Plan Files

## Overview

When creating plans in "plan mode", plans are saved with session-specific identifiers to prevent confusion across different agent sessions.

## Session Identification

### Available Identifiers

1. **CURSOR_TRACE_ID** (Available)
   - Unique per session
   - Accessible via `os.environ.get('CURSOR_TRACE_ID')`
   - Format: `d8a116dedb7740fbb4ccb9f5b7205780`
   - Used as: `trace_d8a116de` (first 8 chars)

2. **Session Title** (Not programmatically accessible)
   - Persists across context compression
   - Visible in Cursor UI
   - Example: "SSL connection error during run group creation"
   - **Note**: Currently not available in environment variables or files
   - **Workaround**: Can be provided manually when creating plans

3. **Timestamp Fallback**
   - Used if neither trace ID nor title available
   - Format: `YYYYMMDD_HHMMSS`

## Usage

### Basic Usage (with CURSOR_TRACE_ID)

```python
from study_query_llm.utils.session_utils import get_plan_filename, ensure_plans_dir

# Ensure .plans directory exists and is gitignored
ensure_plans_dir()

# Get plan filename (uses CURSOR_TRACE_ID automatically)
plan_path = get_plan_filename(prefix="PLAN")

# Create plan
plan_path.write_text("# Plan\n\n...", encoding='utf-8')
```

### With Session Title (if you have it)

```python
from study_query_llm.utils.session_utils import get_plan_filename

# If you know the session title (e.g., from user or UI)
session_title = "SSL connection error during run group creation"
plan_path = get_plan_filename(
    prefix="PLAN",
    session_title=session_title
)

# Result: .plans/PLAN_SSL_connection_error_during_run_group_creation_20260204_233956.md
```

### Example: Creating a Plan in Agent Mode

```python
from pathlib import Path
from study_query_llm.utils.session_utils import get_plan_filename, ensure_plans_dir

# Setup
ensure_plans_dir()

# Option 1: Use trace ID (automatic)
plan_path = get_plan_filename(prefix="REFACTORING_PLAN")

# Option 2: Use session title if available
# (You'd need to get this from the user or UI)
plan_path = get_plan_filename(
    prefix="REFACTORING_PLAN",
    session_title="Refactor async code to use tqdm.asyncio"
)

# Write plan
plan_content = """# Refactoring Plan

## Overview
...

## Steps
1. ...
"""
plan_path.write_text(plan_content, encoding='utf-8')
print(f"Plan saved to: {plan_path}")
```

## File Naming

Plans are named: `{PREFIX}_{SESSION_ID}_{TIMESTAMP}.md`

Examples:
- `PLAN_trace_d8a116de_20260204_233956.md` (using trace ID)
- `PLAN_SSL_connection_error_during_run_group_creation_20260204_233956.md` (using title)
- `PLAN_20260204_233956.md` (fallback timestamp)

## Directory Structure

```
.plans/                    # Gitignored directory
├── README.md             # This guide
└── PLAN_*.md             # Session-specific plan files
```

## Best Practices

1. **Always use `ensure_plans_dir()`** - Creates directory and updates .gitignore
2. **Use descriptive prefixes** - e.g., `REFACTORING_PLAN`, `BUGFIX_PLAN`, `FEATURE_PLAN`
3. **Clean up old plans** - Delete plans when no longer needed
4. **Session title** - If you have access to it (from user or UI), pass it for better identification

## Future Improvements

If Cursor exposes session title programmatically in the future:
- Update `get_session_identifier()` to automatically detect it
- Check environment variables: `CURSOR_SESSION_TITLE`, `SESSION_TITLE`
- Check metadata files in `.cursor/` directory

## Current Limitations

- Session title is not accessible programmatically
- Must be provided manually if you want to use it
- CURSOR_TRACE_ID is the most reliable automatic identifier
