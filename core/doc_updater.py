"""
Auto-Updating Documentation Module
Automatically updates ARCHITECTURE.md, FEATURES.md, and ISSUES.md 
based on significant code changes or manual logs.
"""
import os
import re
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"

ISSUES_FILE = DOCS_DIR / "ISSUES.md"
FEATURES_FILE = DOCS_DIR / "FEATURES.md"
ARCHITECTURE_FILE = DOCS_DIR / "ARCHITECTURE.md"

def _ensure_file_exists(filepath: Path, default_content: str):
    if not filepath.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(default_content)

def log_issue(title: str, error: str, files: list[str], cause: str, status: str = "Open 🔴", solution: str = ""):
    """Logs an issue to ISSUES.md"""
    _ensure_file_exists(ISSUES_FILE, "# Quasar Issues Log\n\n")
    
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    entry = f"\n### [{date_str}] — {title}\n"
    entry += f"- **Error/Issue**: {error}\n"
    entry += f"- **Files Affected**: {', '.join(files)}\n"
    entry += f"- **Cause**: {cause}\n"
    if solution:
        entry += f"- **Solution**: {solution}\n"
    entry += f"- **Status**: {status}\n"

    with open(ISSUES_FILE, "a", encoding="utf-8") as f:
        f.write(entry)
        
    print(f"Logged issue: {title}")

def log_fix(title: str, solution: str, files: list[str]):
    """Logs a fix for an issue"""
    log_issue(title, "N/A", files, "N/A", "Fixed ✅", solution)

def update_feature_status(feature_name: str, new_status: str):
    """Updates the status of a feature in FEATURES.md
    Expects markdown tables with columns including feature names and status.
    """
    _ensure_file_exists(FEATURES_FILE, "# Quasar Features\n\n")
    
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Search for the row containing the feature name and replace its status column
    # Assuming standard markdown table format
    lines = content.split('\n')
    updated = False
    
    for i, line in enumerate(lines):
        if '|' in line and feature_name.lower() in line.lower():
            parts = [p.strip() for p in line.split('|')]
            # Let's assume status is usually the last or second to last column
            # We'll just replace the whole line with a notice for now if we can't parse it
            # Actually, standardizing on a simple regex might be better, or just adding a log.
            pass
            
    # For now, we append a status update log if we don't do complex table parsing
    if not updated:
        with open(FEATURES_FILE, "a", encoding="utf-8") as f:
            date_str = datetime.now().strftime("%Y-%m-%d")
            f.write(f"\n- **{date_str} Status Update**: {feature_name} -> {new_status}\n")

def check_architecture():
    """Validates if ARCHITECTURE.md is somewhat up to date with core files."""
    print("Checking ARCHITECTURE.md sync...")
    if not ARCHITECTURE_FILE.exists():
        print("ARCHITECTURE.md missing.")
        return False
        
    core_files = [f.name for f in (ROOT_DIR / "core").glob("*.py")]
    
    with open(ARCHITECTURE_FILE, "r", encoding="utf-8") as f:
        arch_content = f.read()
        
    missing = [cf for cf in core_files if cf not in arch_content and cf != "__init__.py"]
    if missing:
        print(f"Warning: These core files are not mentioned in ARCHITECTURE.md: {', '.join(missing)}")
        return False
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        if not check_architecture():
            sys.exit(1)
        print("Documentation check passed.")
        sys.exit(0)
