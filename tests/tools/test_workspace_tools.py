import json
import platform
import sys
from pathlib import Path

import pytest

from tools import workspace


pytestmark = pytest.mark.core_headless


@pytest.fixture(autouse=True)
def sandbox_workspace(tmp_path):
    original_root = workspace.WORKSPACE_ROOT
    extra_roots = workspace.list_additional_roots()
    workspace.set_workspace_root(tmp_path)
    workspace.set_additional_allowed_roots([])
    try:
        yield tmp_path
    finally:
        workspace.set_workspace_root(original_root)
        workspace.set_additional_allowed_roots(extra_roots)


def test_search_and_stat_file(tmp_path):
    target = tmp_path / "notes" / "readme.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello", encoding="utf-8")

    search = workspace.search_files("readme")
    assert search["ok"]
    assert "notes/readme.txt" in search["paths"]

    stats = workspace.stat_file("notes/readme.txt")
    assert stats["ok"]
    assert stats["exists"]
    assert stats["is_file"]
    assert stats["size"] == 5


def test_windows_style_paths_are_normalized(tmp_path):
    target = tmp_path / "Desktop" / "linux commands.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "ls -la\n"
    target.write_text(payload, encoding="utf-8")

    search = workspace.search_files(r"Desktop\linux commands.txt")
    assert search["ok"]
    assert "Desktop/linux commands.txt" in search["paths"]

    read = workspace.read_file(r"Desktop\linux commands.txt")
    assert read["ok"]
    assert read["content"] == payload


def test_register_allowed_root_permits_external_access(tmp_path_factory):
    outside_root = tmp_path_factory.mktemp("outside")
    target = outside_root / "external.txt"
    payload = "external data\n"
    target.write_text(payload, encoding="utf-8")

    workspace.register_allowed_root(str(outside_root))
    allowed_roots = workspace.list_allowed_roots()
    assert any(str(outside_root) in entry for entry in allowed_roots)
    read = workspace.read_file(str(target))
    assert read["ok"], read
    assert read["content"] == payload


def test_search_across_allowed_roots(tmp_path_factory):
    external_root = tmp_path_factory.mktemp("external_desktop")
    desktop_dir = external_root / "Desktop"
    target = desktop_dir / "casings.txt"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    target.write_text("example", encoding="utf-8")

    workspace.register_allowed_root(str(external_root))

    search = workspace.search_files("casings.txt", glob="Desktop/**/*")
    assert search["ok"], search
    assert "Desktop/casings.txt" in search["paths"]


def test_list_directory_filters_extensions(tmp_path):
    desktop = tmp_path / "Desktop"
    desktop.mkdir(parents=True)
    (desktop / "alpha.txt").write_text("a", encoding="utf-8")
    (desktop / "beta.log").write_text("b", encoding="utf-8")
    nested = desktop / "nested"
    nested.mkdir()
    (nested / "gamma.txt").write_text("c", encoding="utf-8")

    listing = workspace.list_directory("Desktop", extensions=["txt"], recursive=True)
    assert listing["ok"], listing
    entries = listing["entries"]
    assert "Desktop/alpha.txt" in entries
    assert "Desktop/nested/gamma.txt" in entries
    assert all(item.endswith(".txt") for item in entries)

def test_json_patch_flow(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"features": {"beta": False}}, indent=2), encoding="utf-8")

    read_result = workspace.read_file("config.json", mode="json")
    assert read_result["ok"]
    current_sha = read_result["sha256"]

    patch = [{"op": "add", "path": "/features/betaSearch", "value": True}]
    dry_run = workspace.dry_run_json_patch("config.json", patch, expect_sha256=current_sha)
    assert dry_run["ok"]
    assert dry_run["preview_json"]["features"]["betaSearch"] is True

    apply = workspace.apply_json_patch("config.json", patch, expect_sha256=current_sha)
    assert apply["ok"]
    assert apply["new_sha256"] != current_sha

    validate = workspace.validate_json("config.json")
    assert validate["ok"]

    mismatch = workspace.dry_run_json_patch("config.json", patch, expect_sha256=current_sha)
    assert not mismatch["ok"]
    assert mismatch["errors"][0]["code"] == "sha_mismatch"


def test_text_patch_flow(tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("line1\nline2\n", encoding="utf-8")

    current_sha = workspace.read_file("script.txt", mode="text")["sha256"]
    diff = (
        "--- a/script.txt\n"
        "+++ b/script.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-line1\n"
        "+line1 updated\n"
        " line2\n"
    )

    dry_run = workspace.dry_run_text_patch("script.txt", diff, expect_sha256=current_sha)
    assert dry_run["ok"]
    assert "line1 updated" in dry_run["preview_text"]

    apply = workspace.apply_text_patch("script.txt", diff, expect_sha256=current_sha)
    assert apply["ok"]

    final_text = Path(tmp_path / "script.txt").read_text(encoding="utf-8")
    assert "line1 updated" in final_text


def test_analyze_and_validate_python(tmp_path):
    module = tmp_path / "example.py"
    module.write_text(
        (
            "import os\n\n"
            "class Greeter:\n    def hello(self):\n        return 'hi'\n\n"
            "def helper():\n    return 42\n"
        ),
        encoding="utf-8",
    )

    analysis = workspace.analyze_python("example.py")
    assert analysis["ok"]
    assert "os" in analysis["imports"]
    assert "Greeter" in analysis["classes"]
    assert "helper" in analysis["functions"]

    validation = workspace.validate_python("example.py")
    assert validation["ok"]


def test_format_file_runs_without_errors(tmp_path):
    module = tmp_path / "format_me.py"
    module.write_text("x=1\n", encoding="utf-8")

    result = workspace.format_file("format_me.py")
    assert result["ok"]
    assert "changed" in result


def test_get_system_info_reflects_host_platform():
    info = workspace.get_system_info()
    assert info["ok"]
    assert info["platform"] == sys.platform
    assert info["system"].lower() == platform.system().lower()
    assert isinstance(info["is_windows"], bool)
    assert isinstance(info["is_linux"], bool)
    assert isinstance(info["is_macos"], bool)
