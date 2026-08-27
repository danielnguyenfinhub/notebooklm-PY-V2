"""Chromium build discovery for launches Playwright's own resolution can't serve.

Playwright pins one exact Chromium build revision per release and reports every
other revision as "not installed". An environment that already ships a Chromium
for a *different* Playwright version — a CI image, a container sandbox, a
machine whose Playwright moved past its cached build — therefore reads as empty
to the ``notebooklm login`` pre-flight, which then downloads a second copy.
Where that download is blocked (proxied network, air-gapped runner, a 403 from
the Playwright CDN) login fails outright with a usable browser sitting one
directory away.

This leaf answers two questions for that case — "is another build on disk?"
(:func:`find_installed_chromium`) and "should this launch override Playwright's
executable?" (:func:`resolve_chromium_executable`) — and packages the answer as
the launch kwargs the capture core spreads (:func:`browser_target_kwargs`).

It lives outside ``browser_capture`` deliberately: that module is shrink-locked
by the ADR-0033 size ratchet, and this is ordinary growth, not a sanctioned
merge. ``browser_capture`` re-exports :func:`find_installed_chromium` because it
is the only ``_auth`` import site the CLI boundary sanctions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .browser_launch_errors import CHANNEL_BROWSERS

logger = logging.getLogger(__name__)

# Executable layouts Playwright uses inside a ``chromium-<revision>`` build
# directory: the historical ``chrome-linux`` / ``chrome-win`` naming and the
# Chrome-for-Testing ``chrome-linux64`` / ``chrome-win64`` naming that replaced
# it. The scan takes the first hit per build directory, so carrying both
# spellings keeps one lookup working across the Playwright versions shipping
# either layout.
CHROMIUM_EXECUTABLE_LAYOUTS = (
    "chrome-linux/chrome",
    "chrome-linux64/chrome",
    "chrome-win/chrome.exe",
    "chrome-win64/chrome.exe",
    "chrome-mac/Chromium.app/Contents/MacOS/Chromium",
    "chrome-mac-arm64/Chromium.app/Contents/MacOS/Chromium",
    "chrome-mac-x64/Chromium.app/Contents/MacOS/Chromium",
)


def _chromium_build_revision(build_dir: Path) -> int:
    """Sort key for ``chromium-<revision>`` dirs; an unparseable name sorts last."""
    revision = build_dir.name.partition("-")[2]
    return int(revision) if revision.isdigit() else -1


def find_installed_chromium(browsers_root: str | None = None) -> Path | None:
    """Return a Chromium already installed under ``PLAYWRIGHT_BROWSERS_PATH``.

    Newest revision wins. ``chromium_headless_shell-*`` builds are not
    candidates — the shell cannot render the headed Google sign-in this exists
    to reach. An unset variable and ``PLAYWRIGHT_BROWSERS_PATH=0`` (Playwright's
    "keep builds inside the package" mode) both return ``None``: neither names a
    shared root to scan, and Playwright's own resolution already covers the
    in-package layout.
    """
    root_value = (
        browsers_root if browsers_root is not None else os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    )
    if not root_value or root_value == "0":
        return None
    try:
        build_dirs = [path for path in Path(root_value).glob("chromium-*") if path.is_dir()]
    except OSError as exc:  # unreadable or absent root: nothing to fall back to
        logger.debug("Could not scan %s for an installed Chromium: %s", root_value, exc)
        return None
    for build_dir in sorted(build_dirs, key=_chromium_build_revision, reverse=True):
        for layout in CHROMIUM_EXECUTABLE_LAYOUTS:
            candidate = build_dir / layout
            if candidate.exists():
                return candidate
    return None


def resolve_chromium_executable(expected_path: object) -> Path | None:
    """Pick an installed Chromium when the build Playwright expects is absent.

    ``expected_path`` is ``BrowserType.executable_path`` — the build this
    Playwright release resolves to. Returns ``None``, meaning "leave Playwright's
    own resolution alone", when that build is on disk *and* when the value is not
    a usable string: under a mocked Playwright there is no real path to compare,
    and overriding the executable on a guess is worse than letting the launch
    proceed untouched.
    """
    if not isinstance(expected_path, str) or not expected_path:
        return None
    if Path(expected_path).exists():
        return None
    return find_installed_chromium()


def browser_target_kwargs(playwright: Any, *, browser: str) -> dict[str, Any]:
    """Return the launch kwargs that select which browser binary to run.

    A :data:`CHANNEL_BROWSERS` value routes to a system install via Playwright's
    ``channel``, which resolves the binary itself — an ``executable_path``
    alongside it would fight that resolution, so the two are mutually exclusive
    by construction here. The bundled-Chromium arm carries an override only when
    Playwright's pinned build is missing and another one is on disk.

    The substitution is logged, not printed: the interactive path already says it
    once in the CLI pre-flight (printing here too would say it twice), and the
    headless re-auth arm has no one watching.
    """
    if browser in CHANNEL_BROWSERS:
        return {"channel": browser}
    fallback = resolve_chromium_executable(getattr(playwright.chromium, "executable_path", None))
    if fallback is None:
        return {}
    logger.info("Playwright's pinned Chromium build is absent; launching %s instead", fallback)
    return {"executable_path": str(fallback)}
