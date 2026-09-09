"""Resolve the Chromium binary for the browser suites; never assume a platform path."""
import os
from pathlib import Path

def chromium_path(playwright):
    """CHROMIUM_PATH wins so CI pins its own build; otherwise use the Playwright-managed browser."""
    pinned=os.getenv('CHROMIUM_PATH')
    if pinned:
        if not Path(pinned).exists():raise SystemExit(f'CHROMIUM_PATH does not exist: {pinned}')
        return pinned
    try:resolved=playwright.chromium.executable_path
    except Exception as exc:raise SystemExit(f'Cannot resolve a Chromium build: {exc}. Install one with "uv run playwright install chromium" or set CHROMIUM_PATH.')
    if not Path(resolved).exists():raise SystemExit(f'Playwright reports {resolved}, which is not installed. Run "uv run playwright install chromium" or set CHROMIUM_PATH.')
    return resolved
