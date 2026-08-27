"""Unit tests for the installed-Chromium fallback (blocked / offline downloads).

Playwright pins one exact Chromium build revision per release and reports every
other revision as "not installed". An environment that already ships a Chromium
for a *different* Playwright version — a CI image, a container sandbox, a
machine whose Playwright moved past its cached build — therefore reads as empty
to the ``notebooklm login`` pre-flight, which then tries to download a second
copy. Where that download is blocked (proxied network, air-gapped runner, a 403
from the Playwright CDN) login used to fail outright with a usable browser
sitting one directory away.

These cover the two halves of the fallback:

* :func:`find_installed_chromium` / :func:`resolve_chromium_executable` — the
  filesystem scan and the "should we override at all?" decision.
* ``run_browser_capture`` — that a resolved fallback actually reaches Playwright
  as ``executable_path``, and that a ``channel`` launch never carries one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from notebooklm._auth.browser_capture import BrowserCapturePlan, run_browser_capture
from notebooklm._auth.chromium_discovery import (
    find_installed_chromium,
    resolve_chromium_executable,
)
from notebooklm._env import get_base_url


def _install_build(root: Path, name: str, layout: str) -> Path:
    """Materialise ``<root>/<name>/<layout>`` and return the executable path."""
    executable = root / name / layout
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text("#!/bin/sh\n")
    return executable


# ---------------------------------------------------------------------------
# find_installed_chromium — the filesystem scan
# ---------------------------------------------------------------------------


def test_unset_browsers_path_finds_nothing(monkeypatch) -> None:
    """No shared root configured: Playwright's own resolution stands."""
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
    assert find_installed_chromium() is None


def test_in_package_mode_finds_nothing(monkeypatch) -> None:
    """``PLAYWRIGHT_BROWSERS_PATH=0`` keeps builds inside the package — nothing to scan."""
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "0")
    assert find_installed_chromium() is None


def test_absent_root_finds_nothing(tmp_path: Path) -> None:
    assert find_installed_chromium(str(tmp_path / "nope")) is None


def test_empty_root_finds_nothing(tmp_path: Path) -> None:
    assert find_installed_chromium(str(tmp_path)) is None


def test_reads_the_env_var_when_no_root_is_passed(tmp_path: Path, monkeypatch) -> None:
    executable = _install_build(tmp_path, "chromium-1194", "chrome-linux/chrome")
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))

    assert find_installed_chromium() == executable


@pytest.mark.parametrize(
    "layout",
    [
        "chrome-linux/chrome",
        "chrome-linux64/chrome",
        "chrome-win/chrome.exe",
        "chrome-win64/chrome.exe",
        "chrome-mac/Chromium.app/Contents/MacOS/Chromium",
        "chrome-mac-arm64/Chromium.app/Contents/MacOS/Chromium",
        "chrome-mac-x64/Chromium.app/Contents/MacOS/Chromium",
    ],
)
def test_every_known_build_layout_is_found(tmp_path: Path, layout: str) -> None:
    """Both the historical and the Chrome-for-Testing directory names resolve.

    The paths are created directly rather than by the host's Playwright, so each
    layout is exercised on every OS in the matrix.
    """
    executable = _install_build(tmp_path, "chromium-1194", layout)

    assert find_installed_chromium(str(tmp_path)) == executable


def test_newest_revision_wins(tmp_path: Path) -> None:
    _install_build(tmp_path, "chromium-1194", "chrome-linux/chrome")
    newest = _install_build(tmp_path, "chromium-1234", "chrome-linux64/chrome")
    _install_build(tmp_path, "chromium-999", "chrome-linux/chrome")

    assert find_installed_chromium(str(tmp_path)) == newest


def test_revision_ordering_is_numeric_not_lexicographic(tmp_path: Path) -> None:
    """``999`` must not outrank ``1234`` — string ordering would pick the older build."""
    _install_build(tmp_path, "chromium-999", "chrome-linux/chrome")
    newest = _install_build(tmp_path, "chromium-1234", "chrome-linux/chrome")

    assert find_installed_chromium(str(tmp_path)) == newest


def test_unparseable_revision_sorts_last(tmp_path: Path) -> None:
    _install_build(tmp_path, "chromium-tip-of-tree", "chrome-linux/chrome")
    numbered = _install_build(tmp_path, "chromium-1194", "chrome-linux/chrome")

    assert find_installed_chromium(str(tmp_path)) == numbered


def test_headless_shell_build_is_not_offered(tmp_path: Path) -> None:
    """``chromium_headless_shell-*`` cannot drive a headed Google sign-in."""
    _install_build(tmp_path, "chromium_headless_shell-1194", "chrome-linux/headless_shell")

    assert find_installed_chromium(str(tmp_path)) is None


def test_build_dir_without_an_executable_is_skipped(tmp_path: Path) -> None:
    """A half-removed build directory is not a usable browser."""
    (tmp_path / "chromium-1234" / "chrome-linux64").mkdir(parents=True)
    older = _install_build(tmp_path, "chromium-1194", "chrome-linux/chrome")

    assert find_installed_chromium(str(tmp_path)) == older


def test_a_file_named_like_a_build_dir_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "chromium-1234").write_text("not a directory")
    older = _install_build(tmp_path, "chromium-1194", "chrome-linux/chrome")

    assert find_installed_chromium(str(tmp_path)) == older


def test_unreadable_root_is_swallowed(tmp_path: Path, monkeypatch) -> None:
    """A scan that raises must degrade to "no fallback", never propagate."""

    def boom(self: Path, pattern: str) -> Any:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "glob", boom)

    assert find_installed_chromium(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# resolve_chromium_executable — the "should we override at all?" decision
# ---------------------------------------------------------------------------


def test_present_expected_build_is_left_alone(tmp_path: Path, monkeypatch) -> None:
    """Playwright's pinned build is installed: no override, no scan."""
    expected = _install_build(tmp_path, "chromium-1234", "chrome-linux64/chrome")
    fallback_root = tmp_path / "other"
    _install_build(fallback_root, "chromium-1194", "chrome-linux/chrome")
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(fallback_root))

    assert resolve_chromium_executable(str(expected)) is None


def test_absent_expected_build_takes_the_fallback(tmp_path: Path, monkeypatch) -> None:
    fallback = _install_build(tmp_path, "chromium-1194", "chrome-linux/chrome")
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))

    missing = tmp_path / "chromium-1234" / "chrome-linux64" / "chrome"

    assert resolve_chromium_executable(str(missing)) == fallback


def test_absent_expected_build_without_a_fallback_stays_none(tmp_path: Path, monkeypatch) -> None:
    """Nothing on disk to reuse: let the pre-flight install / Playwright report it."""
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)

    assert resolve_chromium_executable(str(tmp_path / "absent" / "chrome")) is None


@pytest.mark.parametrize(
    "expected",
    [
        pytest.param(None, id="none"),
        pytest.param("", id="empty-string"),
        pytest.param(MagicMock(), id="mocked-playwright"),
        pytest.param(Path("/x/chrome"), id="path-object"),
    ],
)
def test_unreadable_expected_path_never_overrides(tmp_path: Path, monkeypatch, expected) -> None:
    """Without a concrete string there is nothing to compare — don't guess.

    Notably this keeps a mocked Playwright (whose ``executable_path`` is a
    ``MagicMock``) on Playwright's own resolution, so the override cannot leak
    into unrelated suites on a host that exports ``PLAYWRIGHT_BROWSERS_PATH``.
    """
    _install_build(tmp_path, "chromium-1194", "chrome-linux/chrome")
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))

    assert resolve_chromium_executable(expected) is None


# ---------------------------------------------------------------------------
# run_browser_capture — the fallback reaches Playwright
# ---------------------------------------------------------------------------


class _RecordingIO:
    def __init__(self) -> None:
        self.emitted: list[str] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        self.emitted.append(str(args[0]) if args else "")

    def fail(self, code: int) -> Any:  # pragma: no cover - not reached
        raise AssertionError(f"unexpected io.fail({code})")

    def run_async(self, coro: Any) -> Any:  # pragma: no cover - not reached
        raise AssertionError("run_async not used here")


class _FakeSyncPlaywright:
    def __init__(self, playwright: Any) -> None:
        self._playwright = playwright

    def __enter__(self) -> Any:
        return self._playwright

    def __exit__(self, *exc: Any) -> bool:
        return False


def _fake_playwright(executable_path: Any) -> Any:
    page = MagicMock()
    page.url = f"{get_base_url()}/"
    page.goto.return_value = None
    page.content.return_value = "<html></html>"
    context = MagicMock()
    context.pages = [page]
    context.storage_state.return_value = {"cookies": [], "origins": []}
    playwright = MagicMock()
    playwright.chromium.executable_path = executable_path
    playwright.chromium.launch_persistent_context.return_value = context
    return playwright


def _capture(plan: BrowserCapturePlan, playwright: Any, io: Any) -> None:
    with patch(
        "playwright.sync_api.sync_playwright",
        side_effect=lambda: _FakeSyncPlaywright(playwright),
    ):
        run_browser_capture(plan, io, headless=True, interactive=False)


def _plan(tmp_path: Path, *, browser: str = "chromium") -> BrowserCapturePlan:
    profile = tmp_path / "browser_profile"
    profile.mkdir(exist_ok=True)
    return BrowserCapturePlan(
        browser=browser,
        browser_profile=profile,
        storage_path=tmp_path / "storage_state.json",
    )


@pytest.mark.requires_playwright
def test_launch_uses_the_fallback_executable(tmp_path: Path, monkeypatch) -> None:
    """The whole point: a blocked download no longer stops a launchable browser."""
    browsers_root = tmp_path / "browsers"
    fallback = _install_build(browsers_root, "chromium-1194", "chrome-linux/chrome")
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(browsers_root))
    playwright = _fake_playwright(
        str(browsers_root / "chromium-1234" / "chrome-linux64" / "chrome")
    )
    io = _RecordingIO()

    _capture(_plan(tmp_path), playwright, io)

    kwargs = playwright.chromium.launch_persistent_context.call_args.kwargs
    assert kwargs["executable_path"] == str(fallback)
    # Logged, not printed: the CLI pre-flight already told the user once.
    assert not any(str(fallback) in line for line in io.emitted)


@pytest.mark.requires_playwright
def test_launch_is_untouched_when_the_pinned_build_is_installed(
    tmp_path: Path, monkeypatch
) -> None:
    browsers_root = tmp_path / "browsers"
    expected = _install_build(browsers_root, "chromium-1234", "chrome-linux64/chrome")
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(browsers_root))
    playwright = _fake_playwright(str(expected))

    _capture(_plan(tmp_path), playwright, _RecordingIO())

    kwargs = playwright.chromium.launch_persistent_context.call_args.kwargs
    assert "executable_path" not in kwargs


@pytest.mark.requires_playwright
@pytest.mark.parametrize("browser", ["chrome", "msedge"])
def test_channel_launch_never_carries_an_executable_override(
    tmp_path: Path, monkeypatch, browser: str
) -> None:
    """``--browser chrome`` resolves a system install; an override would fight it."""
    browsers_root = tmp_path / "browsers"
    _install_build(browsers_root, "chromium-1194", "chrome-linux/chrome")
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(browsers_root))
    playwright = _fake_playwright(
        str(browsers_root / "chromium-1234" / "chrome-linux64" / "chrome")
    )

    _capture(_plan(tmp_path, browser=browser), playwright, _RecordingIO())

    kwargs = playwright.chromium.launch_persistent_context.call_args.kwargs
    assert kwargs["channel"] == browser
    assert "executable_path" not in kwargs
