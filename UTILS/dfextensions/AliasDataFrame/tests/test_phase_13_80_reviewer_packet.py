"""PHASE_13_80 A1-A8 reviewer packet losslessness/dedup acceptance tests."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import shlex
import zipfile

import pytest


def _tool_path() -> Path:
    override = os.environ.get("ADF_PACKET_TOOL_UNDER_TEST")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "scripts" / "reviewer_packet.py"


def _runner_path() -> Path:
    override = os.environ.get("ADF_RUNNER_UNDER_TEST")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "run_tests.sh"


def _bundler_path() -> Path:
    override = os.environ.get("ADF_MAKE_LLM_BUNDLE_UNDER_TEST")
    if override:
        return Path(override)
    repo = Path(__file__).resolve().parents[1]
    candidates = [
        repo.parent / "scripts" / "make_llm_bundle.py",
        repo / "scripts" / "make_llm_bundle.py",
        repo / "tests" / "scripts" / "make_llm_bundle.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("existing make_llm_bundle.py not found")


def _load_tool():
    spec = importlib.util.spec_from_file_location("reviewer_packet_under_test", _tool_path())
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path):
    root = tmp_path / "root"
    (root / "docs").mkdir(parents=True)
    (root / "test_logs").mkdir()
    (root / "tests").mkdir()
    (root / "docs" / "CAPABILITY_MATRIX.json").write_text('{"x":1}\n', encoding="utf-8")
    (root / "test_logs" / "CAPABILITY_MATRIX_20260906.json").write_text('{"x":1}\n', encoding="utf-8")
    (root / "docs" / "CAPABILITY_MATRIX.html").write_text('<p>x</p>\n', encoding="utf-8")
    (root / "test_logs" / "CAPABILITY_MATRIX_20260906.html").write_text('<p>x</p>\n', encoding="utf-8")
    # One-byte-different control: must never join the JSON duplicate group.
    (root / "tests" / "one_byte_different.txt").write_text('{"x":2}\n', encoding="utf-8")
    (root / "tests" / "test_candidate.py").write_text('def test_x():\n    assert True\n', encoding="utf-8")
    paths = [
        "docs/CAPABILITY_MATRIX.json",
        "test_logs/CAPABILITY_MATRIX_20260906.json",
        "docs/CAPABILITY_MATRIX.html",
        "test_logs/CAPABILITY_MATRIX_20260906.html",
        "tests/one_byte_different.txt",
        "tests/test_candidate.py",
    ]
    return root, paths


def _build_all(tmp_path: Path):
    tool = _load_tool()
    root, paths = _fixture(tmp_path)
    manifest = tool.build_manifest(root, paths)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(tool.manifest_bytes(manifest))
    zip_path = tmp_path / "reviewer.zip"
    tar_path = tmp_path / "reviewer.tar"
    bundle_path = tmp_path / "reviewer.llmbundle.txt"
    tool.build_zip(root, manifest, zip_path)
    tool.build_tar(root, manifest, tar_path)
    subprocess.run(
        [sys.executable, str(_bundler_path()), str(zip_path), "-o", str(bundle_path), "--allow-delimiters"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return tool, root, manifest, manifest_path, zip_path, tar_path, bundle_path


def test_manifest_schema_and_digest_are_deterministic(tmp_path):
    tool = _load_tool()
    root, paths = _fixture(tmp_path)
    a = tool.build_manifest(root, paths)
    b = tool.build_manifest(root, list(reversed(paths)))
    assert a["schema"] == "ADF_REVIEW_PACKET_LOGICAL_MANIFEST/1"
    assert a["schema_version"] == 1
    assert a["logical_manifest_sha256"] == b["logical_manifest_sha256"]
    assert a["members"] == b["members"]


def test_exact_duplicates_are_measured_but_zip_paths_are_regular_real_files(tmp_path):
    tool, root, manifest, _, zip_path, _, _ = _build_all(tmp_path)
    assert manifest["logical_member_count"] == 6
    assert manifest["physical_payload_count"] == 4
    assert manifest["alias_count"] == 2
    assert manifest["duplicate_group_count"] == 2
    with zipfile.ZipFile(zip_path) as zf:
        infos = {i.filename: i for i in zf.infolist() if not i.is_dir()}
        assert set(infos) == {"REVIEW_PACKET_MANIFEST.json"} | {m["logical_path"] for m in manifest["members"]}
        for m in manifest["members"]:
            info = infos[m["logical_path"]]
            assert not tool._zip_info_is_symlink(info)
            assert zf.read(info) == (root / m["logical_path"]).read_bytes()

def test_one_byte_different_control_is_not_deduplicated(tmp_path):
    tool = _load_tool()
    root, paths = _fixture(tmp_path)
    manifest = tool.build_manifest(root, paths)
    rows = {m["logical_path"]: m for m in manifest["members"]}
    assert rows["tests/one_byte_different.txt"]["content_sha256"] != rows["docs/CAPABILITY_MATRIX.json"]["content_sha256"]
    assert not rows["tests/one_byte_different.txt"]["is_alias"]


def test_zip_tar_and_llmbundle_roundtrip_exact_logical_bytes(tmp_path):
    tool, root, manifest, _, zip_path, tar_path, bundle_path = _build_all(tmp_path)
    tool.verify_packet(root, manifest, zip_path, tar_path, bundle_path)
    expected = {m["logical_path"]: (root / m["logical_path"]).read_bytes() for m in manifest["members"]}
    _, zlogical = tool.reconstruct_from_zip(zip_path)
    _, tlogical = tool.reconstruct_from_tar(tar_path)
    _, blogical = tool.parse_llmbundle_for_verification(bundle_path)
    assert zlogical == expected
    assert tlogical == expected
    assert blogical == expected


def test_shared_llmbundle_consumes_completed_consumer_safe_zip_and_preserves_manifest_identity(tmp_path):
    tool, _, manifest, _, zip_path, _, bundle_path = _build_all(tmp_path)
    text = bundle_path.read_text(encoding="utf-8")
    assert text.startswith("LLMBUNDLE/")
    assert "type: symlink" not in text
    assert "path: test_logs/CAPABILITY_MATRIX_20260906.json" in text
    assert "REVIEW_PACKET_MANIFEST.json" in text
    zm, _ = tool.reconstruct_from_zip(zip_path)
    bm, _ = tool.parse_llmbundle_for_verification(bundle_path)
    assert zm["logical_manifest_sha256"] == bm["logical_manifest_sha256"] == manifest["logical_manifest_sha256"]


def test_existing_shared_bundler_remains_single_zip_to_llmbundle_owner():
    tool_source = _tool_path().read_text(encoding="utf-8")
    runner = _runner_path().read_text(encoding="utf-8")
    assert "def build_llmbundle" not in tool_source
    assert "build_llmbundle_from_zip" not in tool_source
    assert "def _cmd_bundle" not in tool_source
    assert 'make_llm_bundle.py' in runner
    assert 'python3 "$BUNDLE_SCRIPT" "$REVIEWER_ZIP"' in runner
    assert '"$PACKET_TOOL" llmbundle' not in runner


def _assert_extracted_logical_bytes(root, manifest, extracted_root):
    for m in manifest["members"]:
        p = extracted_root / m["logical_path"]
        assert p.is_file(), m["logical_path"]
        data = p.read_bytes()
        assert len(data) == m["byte_count"], m["logical_path"]
        assert hashlib.sha256(data).hexdigest() == m["content_sha256"], m["logical_path"]
        assert data == (root / m["logical_path"]).read_bytes(), m["logical_path"]


def test_external_consumer_python_zipfile_extractall_is_byte_exact(tmp_path):
    _, root, manifest, _, zip_path, _, _ = _build_all(tmp_path)
    extracted = tmp_path / "extract_python"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extracted)
    _assert_extracted_logical_bytes(root, manifest, extracted)


def test_external_consumer_infozip_unzip_is_byte_exact(tmp_path):
    assert shutil.which("unzip"), "Info-ZIP unzip is required for PHASE_13_80 external-consumer falsifier"
    _, root, manifest, _, zip_path, _, _ = _build_all(tmp_path)
    extracted = tmp_path / "extract_infozip"
    extracted.mkdir()
    proc = subprocess.run(
        ["unzip", "-q", str(zip_path), "-d", str(extracted)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert proc.returncode == 0, proc.stdout
    _assert_extracted_logical_bytes(root, manifest, extracted)


def test_changed_payload_after_manifest_is_detected_not_stale_aliased(tmp_path):
    tool, root, manifest, _, zip_path, _, _ = _build_all(tmp_path)
    (root / "test_logs" / "CAPABILITY_MATRIX_20260906.json").write_text('{"x":9}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source drift after manifest build"):
        tool.verify_packet(root, manifest, zip_path=zip_path)


def test_llmbundle_is_derived_from_completed_zip_not_later_worktree_bytes(tmp_path):
    tool, root, manifest, _, zip_path, _, _ = _build_all(tmp_path)
    zip_manifest, zip_logical = tool.reconstruct_from_zip(zip_path)

    # Mutate and remove worktree files after ZIP custody is frozen.
    (root / "docs" / "CAPABILITY_MATRIX.json").write_text('{"MUTATED":true}\n', encoding="utf-8")
    (root / "tests" / "test_candidate.py").unlink()

    bundle = tmp_path / "from_zip_after_worktree_mutation.llmbundle.txt"
    subprocess.run(
        [sys.executable, str(_bundler_path()), str(zip_path), "-o", str(bundle), "--allow-delimiters"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    bundle_manifest, bundle_logical = tool.parse_llmbundle_for_verification(bundle)

    assert bundle_manifest["logical_manifest_sha256"] == zip_manifest["logical_manifest_sha256"]
    assert bundle_logical == zip_logical


def test_duplicate_aliases_preserve_distinct_logical_roles(tmp_path):
    tool, _, manifest, _, _, _, _ = _build_all(tmp_path)
    rows = {m["logical_path"]: m for m in manifest["members"]}
    assert rows["docs/CAPABILITY_MATRIX.json"]["logical_role"] == "generated_canonical"
    assert rows["test_logs/CAPABILITY_MATRIX_20260906.json"]["logical_role"] == "generated_snapshot"
    assert rows["docs/CAPABILITY_MATRIX.json"]["physical_payload_id"] == rows["test_logs/CAPABILITY_MATRIX_20260906.json"]["physical_payload_id"]


def test_unexpected_extra_physical_zip_member_is_fatal(tmp_path):
    tool, _, _, _, zip_path, _, _ = _build_all(tmp_path)
    with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("unexpected_extra.txt", b"must fail closed\n")
    with pytest.raises(ValueError, match="ZIP member-set mismatch"):
        tool.reconstruct_from_zip(zip_path)


def test_profile_reports_exact_bytes_categories_and_labeled_token_estimate(tmp_path):
    tool, root, manifest, _, zip_path, tar_path, bundle_path = _build_all(tmp_path)
    before = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in [m["logical_path"] for m in manifest["members"]]}
    profile = tool.build_profile(root, manifest, zip_path, tar_path, bundle_path)
    after = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in before}
    assert before == after
    assert profile["logical_expanded_bytes"] > profile["unique_payload_bytes"]
    assert profile["exact_duplicate_bytes"] == profile["logical_expanded_bytes"] - profile["unique_payload_bytes"]
    assert profile["zip_physical_bytes"] == zip_path.stat().st_size
    assert profile["llmbundle_physical_bytes"] == bundle_path.stat().st_size
    assert profile["token_measurement"] == "estimated"
    assert profile["token_estimate_method"] == "chars_div_4"
    assert profile["provider_quota"] is None
    assert "generated_json" in profile["by_category"]
    assert "tests" in profile["by_category"]


def test_cli_manifest_recomputes_duplicates_each_build(tmp_path):
    tool_path = _tool_path()
    root, paths = _fixture(tmp_path)
    files = tmp_path / "inputs.txt"
    files.write_text("\n".join(paths) + "\n", encoding="utf-8")
    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    subprocess.run([sys.executable, str(tool_path), "manifest", "--root", str(root), "--files-file", str(files), "--manifest", str(m1)], check=True)
    # Break one duplicate group by one byte and rebuild from the same logical inventory.
    (root / "test_logs" / "CAPABILITY_MATRIX_20260906.json").write_text('{"x":3}\n', encoding="utf-8")
    subprocess.run([sys.executable, str(tool_path), "manifest", "--root", str(root), "--files-file", str(files), "--manifest", str(m2)], check=True)
    a = json.loads(m1.read_text())
    b = json.loads(m2.read_text())
    assert a["physical_payload_count"] == 4
    assert b["physical_payload_count"] == 5
    assert a["logical_manifest_sha256"] != b["logical_manifest_sha256"]


def test_runner_integrates_manifest_duplicates_shared_bundler_roundtrip_and_profiler():
    text = _runner_path().read_text(encoding="utf-8")
    required = [
        "scripts/reviewer_packet.py",
        "reviewer_manifest_${TS}.json",
        "reviewer_profile_${TS}.txt",
        "reviewer_inputs_${TS}.txt",
        '"$PACKET_TOOL" manifest',
        '"$PACKET_TOOL" zip',
        '"$PACKET_TOOL" tar',
        'make_llm_bundle.py',
        'python3 "$BUNDLE_SCRIPT" "$REVIEWER_ZIP"',
        '--allow-delimiters',
        'PACKAGE_EXIT=$?',
        'Reviewer package construction/custody failed',
        "VERIFY_ARGS=(",
        "PROFILE_ARGS=(",
        'python3 "$PACKET_TOOL" "${VERIFY_ARGS[@]}"',
        'python3 "$PACKET_TOOL" "${PROFILE_ARGS[@]}"',
        "logical_manifest_sha256",
    ]
    missing = [token for token in required if token not in text]
    assert not missing, missing
    package_section = text.split('echo "--- Packaging reviewer.zip ---"', 1)[1]
    # Fixed custody + reuse invariant: existing shared converter consumes the
    # completed ZIP; reviewer_packet.py is not an alternate converter.
    bundle_block = package_section.split('python3 "$BUNDLE_SCRIPT" "$REVIEWER_ZIP"', 1)[1].split('BUNDLE_EXIT=$?', 1)[0]
    assert '--root ' not in bundle_block
    assert '--manifest ' not in bundle_block
    assert '"$PACKET_TOOL" llmbundle' not in package_section
    assert 'ordinary reviewer files' in package_section
    assert 'ZIP/TAR symlink aliases' not in package_section

def test_actual_runner_propagates_packet_failure_after_green_pytest(tmp_path):
    """F1: actual run_tests.sh must fail closed after a genuine green pytest path."""
    root = tmp_path / "mini_adf"
    (root / "tests").mkdir(parents=True)
    (root / "scripts").mkdir()
    shutil.copy2(_runner_path(), root / "run_tests.sh")
    (root / "tests" / "test_smoke.py").write_text("def test_green():\n    assert True\n", encoding="utf-8")

    # Keep this actual-runner falsifier portable to lean reviewer sandboxes:
    # use a tiny python3 shim that removes only the runner's -n <workers> pair
    # from pytest invocations, then delegates to the real Python.  This still
    # executes genuine pytest; it only avoids requiring pytest-xdist here.
    bindir = root / "bin"
    bindir.mkdir()
    shim = bindir / "python3"
    real_python = shlex.quote(sys.executable)
    shim.write_text(
        "#!/bin/bash\n"
        "if [[ \"$1\" == \"-m\" && \"$2\" == \"pytest\" ]]; then\n"
        "  out=()\n"
        "  skip=0\n"
        "  for arg in \"$@\"; do\n"
        "    if [[ $skip -eq 1 ]]; then skip=0; continue; fi\n"
        "    if [[ \"$arg\" == \"-n\" ]]; then skip=1; continue; fi\n"
        "    out+=(\"$arg\")\n"
        "  done\n"
        f"  exec {real_python} \"${{out[@]}}\"\n"
        "fi\n"
        f"exec {real_python} \"$@\"\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "phase1380@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "PHASE_13_80 test"], cwd=root, check=True)
    subprocess.run(["git", "add", "run_tests.sh", "tests/test_smoke.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=root, check=True)

    env = os.environ.copy()
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    env.update({
        "PYTEST_WORKERS": "1",
        "ADF_SKIP_TAG_DRIFT_CHECK": "1",
        "ADF_REVIEWER_TAR": "0",
        "ADF_REVIEWER_BUNDLE": "0",
    })
    # scripts/reviewer_packet.py intentionally does not exist: genuine package-stage failure.
    proc = subprocess.run(
        ["bash", "run_tests.sh", "--quick"],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    assert "1 passed" in proc.stdout
    assert "reviewer packet tool not found" in proc.stdout
    assert "Reviewer package construction/custody failed" in proc.stdout
    assert proc.returncode != 0
    assert "✅ All tests passed" not in proc.stdout


def test_runner_keeps_existing_pipestatus_custody_after_packet_change():
    lines = _runner_path().read_text(encoding="utf-8").splitlines()
    full = next(i for i, line in enumerate(lines) if '2>&1 | tee "$LOG_FILE"' in line)
    assert next(line.strip() for line in lines[full + 1:] if line.strip()) == 'TEST_EXIT=${PIPESTATUS[0]}'
    focused = next(i for i, line in enumerate(lines) if '2>&1 | tee "$FOCUSED_LOG"' in line)
    assert next(line.strip() for line in lines[focused + 1:] if line.strip()) == 'FOCUSED_EXIT=${PIPESTATUS[0]}'
