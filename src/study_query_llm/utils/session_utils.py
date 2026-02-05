"""
Session utilities for Cursor integration.

Provides utilities for getting session identifiers for plan file naming
and other session-specific operations.
"""
import os
import re
from datetime import datetime
from typing import Optional
from pathlib import Path


def get_cursor_session_id() -> Optional[str]:
    """
    Get Cursor session trace ID if available.
    
    Returns:
        CURSOR_TRACE_ID if available, None otherwise.
    """
    return os.environ.get('CURSOR_TRACE_ID')


def sanitize_for_filename(text: str, max_length: int = 50) -> str:
    """
    Sanitize text for use in filenames.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length of output
    
    Returns:
        Sanitized string safe for filenames
    """
    # Remove special characters, keep alphanumeric, spaces, hyphens, underscores
    safe = re.sub(r'[^\w\s-]', '', text)
    # Replace spaces and multiple hyphens/underscores with single underscore
    safe = re.sub(r'[-\s]+', '_', safe)
    # Remove leading/trailing underscores
    safe = safe.strip('_')
    # Limit length
    return safe[:max_length] if safe else safe


def get_session_identifier(session_title: Optional[str] = None) -> str:
    """
    Get a unique session identifier for plan file naming.
    
    This function tries multiple sources in order:
    1. Provided session_title (if given)
    2. CURSOR_TRACE_ID from environment
    3. Timestamp fallback
    
    Args:
        session_title: Optional session title to use (if you have access to it)
    
    Returns:
        A sanitized identifier suitable for use in filenames.
    """
    if session_title:
        return sanitize_for_filename(session_title)
    
    # Try CURSOR_TRACE_ID
    trace_id = get_cursor_session_id()
    if trace_id:
        return f"trace_{trace_id[:8]}"
    
    # Final fallback: timestamp
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def get_plan_filename(
    prefix: str = "PLAN",
    session_title: Optional[str] = None,
    plans_dir: str = ".plans"
) -> Path:
    """
    Generate a plan filename with session identifier.
    
    Args:
        prefix: Prefix for the plan file (default: "PLAN")
        session_title: Optional session title (if available)
        plans_dir: Directory for plans (default: ".plans")
    
    Returns:
        Path object for the plan file
    
    Example:
        >>> get_plan_filename(session_title="SSL connection error")
        Path('.plans/PLAN_SSL_connection_error_20260205_123456.md')
    """
    session_id = get_session_identifier(session_title)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{prefix}_{session_id}_{timestamp}.md"
    
    plans_path = Path(plans_dir)
    plans_path.mkdir(exist_ok=True)
    
    return plans_path / filename


def ensure_plans_dir(plans_dir: str = ".plans") -> Path:
    """
    Ensure the plans directory exists and is gitignored.
    
    Args:
        plans_dir: Directory path (default: ".plans")
    
    Returns:
        Path to plans directory
    """
    plans_path = Path(plans_dir)
    plans_path.mkdir(exist_ok=True)
    
    # Ensure .gitignore entry exists
    gitignore_path = Path(".gitignore")
    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding='utf-8')
        if plans_dir not in content:
            with gitignore_path.open('a', encoding='utf-8') as f:
                f.write(f"\n# Temporary plan files\n{plans_dir}/\n")
    else:
        gitignore_path.write_text(f"# Temporary plan files\n{plans_dir}/\n", encoding='utf-8')
    
    return plans_path
