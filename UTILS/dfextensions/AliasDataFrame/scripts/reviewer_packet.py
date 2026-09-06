#!/usr/bin/env python3
"""Lossless reviewer-packet manifest, exact deduplication and profiling.

PHASE_13_80_ADF PART-A A1-A8.

The logical packet is authoritative. ZIP/TAR retain every logical path as an
ordinary real member so ordinary extraction yields byte-exact reviewer files.
Exact duplicate relationships remain explicit in the logical manifest for
measurement and later bundle-level optimization. LLMBUNDLE creation remains
owned by the existing shared make_llm_bundle.py; this module only verifies the
resulting representation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import stat
import tarfile
from typing import Any, Iterable
import zipfile

SCHEMA = "ADF_REVIEW_PACKET_LOGICAL_MANIFEST/1"
SCHEMA_VERSION = 1
MANIFEST_MEMBER = "REVIEW_PACKET_MANIFEST.json"



def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(root: Path, logical_path: str) -> bytes:
    p = (root / logical_path).resolve()
    root_resolved = root.resolve()
    try:
        p.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"logical path escapes root: {logical_path}") from exc
    return p.read_bytes()


def _normalize_path(raw: str) -> str:
    raw = raw.strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    p = PurePosixPath(raw)
    if not raw or p.is_absolute() or ".." in p.parts:
        raise ValueError(f"invalid logical path: {raw!r}")
    return str(p)


def _text_char_count(data: bytes) -> int | None:
    try:
        return len(data.decode("utf-8"))
    except UnicodeDecodeError:
        return None


def _classify(path: str) -> tuple[str, str, str]:
    """Return (logical_role, content_category, generated_status)."""
    name = PurePosixPath(path).name
    lower = path.lower()
    if path == "docs/ARCHITECT_DECISIONS.md":
        return "governance_registry", "governance", "authored"
    if path.startswith("docs/CAPABILITY_MATRIX."):
        ext = PurePosixPath(path).suffix.lstrip(".") or "text"
        return "generated_canonical", f"generated_{ext}", "generated_unattested"
    if path.startswith("test_logs/CAPABILITY_MATRIX_"):
        ext = PurePosixPath(path).suffix.lstrip(".") or "text"
        return "generated_snapshot", f"generated_{ext}", "generated_unattested"
    if "diff_last_commit_" in name:
        return "raw_diff_last_commit", "authored_diff", "runtime_evidence"
    if "diff_to_phase_" in name:
        return "raw_diff_to_phase", "authored_diff", "runtime_evidence"
    if path.startswith("tests/"):
        if path.endswith(".json"):
            return "test_contract", "tests", "authored"
        return "test_source", "tests", "authored"
    if name.startswith("test_full_") and name.endswith(".log"):
        return "full_test_log", "logs", "runtime_evidence"
    if name.startswith("test_focused_") and name.endswith(".log"):
        return "focused_test_log", "logs", "runtime_evidence"
    if name.startswith("runxfail_focused_"):
        return "raw_xfail_log", "logs", "runtime_evidence"
    if name.startswith("test_failures_"):
        return "failure_ledger", "logs", "runtime_evidence"
    if name.startswith("focused_nodes_"):
        return "focused_node_manifest", "metadata", "runtime_evidence"
    if name.startswith("SUMMARY_"):
        return "run_summary", "metadata", "runtime_evidence"
    if name.startswith("git_status_"):
        return "git_status", "metadata", "runtime_evidence"
    if name.startswith("md5_manifest_"):
        return "candidate_hash_manifest", "metadata", "runtime_evidence"
    if name.startswith("timing_packet_"):
        return "timing_snapshot", "metadata", "runtime_evidence"
    if path.endswith(".py") or path.endswith(".sh") or path.endswith(".C"):
        return "candidate_source", "source", "authored"
    if lower.endswith(".md"):
        return "documentation", "markdown", "authored"
    if lower.endswith(".json"):
        return "data", "json", "authored"
    if lower.endswith(".html"):
        return "document", "html", "authored"
    if lower.endswith(".log") or lower.endswith(".txt"):
        return "evidence", "logs", "runtime_evidence"
    return "evidence", "other", "authored"


def _normalized_digest_payload(members: list[dict[str, Any]]) -> bytes:
    fields = []
    for m in sorted(members, key=lambda x: x["logical_path"]):
        fields.append(
            {
                "logical_path": m["logical_path"],
                "logical_role": m["logical_role"],
                "byte_count": m["byte_count"],
                "content_sha256": m["content_sha256"],
                "physical_payload_id": m["physical_payload_id"],
                "generated_status": m["generated_status"],
                "generation_attestation_id": m["generation_attestation_id"],
            }
        )
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_logical_manifest_sha256(members: list[dict[str, Any]]) -> str:
    return _sha256(_normalized_digest_payload(members))


def _manifest_stats(members: list[dict[str, Any]]) -> dict[str, Any]:
    logical_bytes = sum(m["byte_count"] for m in members)
    physical_ids: dict[str, int] = {}
    logical_chars = 0
    all_text = True
    for m in members:
        physical_ids.setdefault(m["physical_payload_id"], m["byte_count"])
        if m["character_count"] is None:
            all_text = False
        else:
            logical_chars += m["character_count"]
    unique_bytes = sum(physical_ids.values())
    duplicate_bytes = logical_bytes - unique_bytes
    return {
        "logical_expanded_bytes": logical_bytes,
        "unique_payload_bytes": unique_bytes,
        "exact_duplicate_bytes": duplicate_bytes,
        "deduplication_ratio_unique_over_logical": (unique_bytes / logical_bytes) if logical_bytes else 1.0,
        "logical_text_characters": logical_chars if all_text else None,
        "physical_payload_count": len(physical_ids),
    }


def build_manifest(root: Path, paths: Iterable[str]) -> dict[str, Any]:
    logical_paths = sorted({_normalize_path(p) for p in paths if p.strip()})
    if not logical_paths:
        raise ValueError("logical packet is empty")

    records: list[dict[str, Any]] = []
    by_payload: dict[str, list[dict[str, Any]]] = {}
    for logical_path in logical_paths:
        data = _read_bytes(root, logical_path)
        digest = _sha256(data)
        payload_id = f"sha256:{digest}"
        role, category, generated_status = _classify(logical_path)
        rec = {
            "logical_path": logical_path,
            "logical_role": role,
            "content_category": category,
            "byte_count": len(data),
            "character_count": _text_char_count(data),
            "content_sha256": digest,
            "physical_payload_id": payload_id,
            "physical_path": None,
            "is_alias": False,
            "alias_of": None,
            "generated_status": generated_status,
            "generation_attestation_id": None,
        }
        records.append(rec)
        by_payload.setdefault(payload_id, []).append(rec)

    duplicate_groups: list[dict[str, Any]] = []
    for payload_id, group in sorted(by_payload.items()):
        # Canonical physical path is deterministic and independent of packet order.
        canonical = min(m["logical_path"] for m in group)
        for m in group:
            m["physical_path"] = canonical
            m["is_alias"] = m["logical_path"] != canonical
            m["alias_of"] = canonical if m["is_alias"] else None
        if len(group) > 1:
            duplicate_groups.append(
                {
                    "physical_payload_id": payload_id,
                    "content_sha256": group[0]["content_sha256"],
                    "byte_count": group[0]["byte_count"],
                    "canonical_physical_path": canonical,
                    "logical_paths": sorted(m["logical_path"] for m in group),
                }
            )

    stats = _manifest_stats(records)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "logical_manifest_sha256": compute_logical_manifest_sha256(records),
        "logical_member_count": len(records),
        "physical_payload_count": stats["physical_payload_count"],
        "alias_count": len(records) - stats["physical_payload_count"],
        "duplicate_group_count": len(duplicate_groups),
        **{k: v for k, v in stats.items() if k != "physical_payload_count"},
        "members": sorted(records, key=lambda x: x["logical_path"]),
        "duplicate_groups": duplicate_groups,
    }
    return manifest


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported reviewer manifest schema: {manifest.get('schema')!r}")
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("manifest has no members")
    expected = compute_logical_manifest_sha256(members)
    if manifest.get("logical_manifest_sha256") != expected:
        raise ValueError("logical manifest digest mismatch")
    paths = [m["logical_path"] for m in members]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("manifest logical paths are not unique/sorted")
    by_path = {m["logical_path"]: m for m in members}
    for m in members:
        if m["physical_path"] not in by_path:
            raise ValueError(f"physical path missing from logical set: {m['physical_path']}")
        target = by_path[m["physical_path"]]
        if target["content_sha256"] != m["content_sha256"] or target["byte_count"] != m["byte_count"]:
            raise ValueError(f"alias payload mismatch for {m['logical_path']}")
        if m["is_alias"] != (m["logical_path"] != m["physical_path"]):
            raise ValueError(f"alias flag mismatch for {m['logical_path']}")


def _physical_members(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in manifest["members"] if not m["is_alias"]]


def _zip_write_bytes(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    zf.writestr(info, data)



def _zip_info_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def build_zip(root: Path, manifest: dict[str, Any], output: Path) -> None:
    """Build the canonical consumer-safe reviewer ZIP.

    Every logical reviewer path is stored as an ordinary real ZIP member with
    its exact bytes. Duplicate relationships remain in REVIEW_PACKET_MANIFEST
    but are not represented as symlinks or other extraction-sensitive aliases.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        _zip_write_bytes(zf, MANIFEST_MEMBER, manifest_bytes(manifest))
        for m in manifest["members"]:
            _zip_write_bytes(zf, m["logical_path"], _read_bytes(root, m["logical_path"]))


def build_tar(root: Path, manifest: dict[str, Any], output: Path) -> None:
    """Build a consumer-safe TAR companion with ordinary real members."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w") as tf:
        mdata = manifest_bytes(manifest)
        mi = tarfile.TarInfo(MANIFEST_MEMBER)
        mi.size = len(mdata)
        mi.mtime = 0
        mi.uid = mi.gid = 0
        mi.uname = mi.gname = ""
        mi.mode = 0o644
        tf.addfile(mi, io.BytesIO(mdata))
        for m in manifest["members"]:
            data = _read_bytes(root, m["logical_path"])
            info = tarfile.TarInfo(m["logical_path"])
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.size = len(data)
            info.mode = 0o644
            tf.addfile(info, io.BytesIO(data))


# LLMBUNDLE generation deliberately does NOT live here. The existing shared
# ../scripts/make_llm_bundle.py remains the single ZIP -> LLMBUNDLE owner.
# The following reader is verification-only because the shared converter has no
# read/round-trip API and A1-A8 requires exact logical reconstruction proof.
_LL_ENTRY_BEGIN = b"===== LLMBUNDLE ENTRY BEGIN =====\n"
_LL_ENTRY_END = b"===== LLMBUNDLE ENTRY END =====\n"
_LL_CONTENT_BEGIN = b"===== CONTENT BEGIN =====\n"
_LL_CONTENT_END = b"===== CONTENT END =====\n"
_LL_TARGET_BEGIN = b"===== TARGET BEGIN =====\n"
_LL_TARGET_END = b"===== TARGET END =====\n"


def _readline_required(fh: io.BufferedReader, where: str) -> bytes:
    line = fh.readline()
    if not line:
        raise ValueError(f"unexpected EOF in LLMBUNDLE while reading {where}")
    return line


def _parse_prefixed(line: bytes, key: str) -> str:
    prefix = (key + ": ").encode("ascii")
    if not line.startswith(prefix):
        raise ValueError(f"expected {key}: line, got {line[:100]!r}")
    return line[len(prefix):].rstrip(b"\n").decode("utf-8")


def parse_llmbundle_for_verification(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Read the existing shared LLMBUNDLE representation for verification only.

    Supports the existing text/symlink entry wire semantics used by
    make_llm_bundle.py. Binary-metadata-only entries are rejected because they
    cannot satisfy A1-A8's byte-exact round-trip requirement.
    """
    files: dict[str, bytes] = {}
    symlinks: dict[str, str] = {}
    entry_paths: list[str] = []
    header_entries: int | None = None
    header_manifest_sha: str | None = None
    trailer_line: bytes | None = None

    with path.open("rb") as fh:
        version = _readline_required(fh, "version").rstrip(b"\n").decode("ascii", "strict")
        if not version.startswith("LLMBUNDLE/"):
            raise ValueError(f"unsupported LLMBUNDLE header: {version!r}")

        # Existing LLMBUNDLE/2 has a header/manifest; LLMBUNDLE/1 proceeds
        # directly to entries. Scan without assuming a private ADF format.
        line = _readline_required(fh, "header/entry")
        while True:
            if line.startswith(b"entries: "):
                header_entries = int(line.split(b":", 1)[1].strip())
            elif line.startswith(b"manifest-sha256: "):
                header_manifest_sha = line.split(b":", 1)[1].strip().decode("ascii")
            if line == _LL_ENTRY_BEGIN:
                break
            if line.startswith(b"===== LLMBUNDLE END "):
                trailer_line = line
                break
            line = _readline_required(fh, "first entry")

        while trailer_line is None:
            logical_path = _parse_prefixed(_readline_required(fh, "entry path"), "path")
            kind = _parse_prefixed(_readline_required(fh, f"type for {logical_path}"), "type")
            entry_paths.append(logical_path)
            if logical_path in files or logical_path in symlinks:
                raise ValueError(f"duplicate LLMBUNDLE logical path: {logical_path}")

            if kind == "text":
                encoding = _parse_prefixed(_readline_required(fh, "encoding"), "encoding")
                if encoding.lower() != "utf-8":
                    raise ValueError(f"unsupported text encoding for {logical_path}: {encoding}")
                byte_count = int(_parse_prefixed(_readline_required(fh, "bytes"), "bytes"))
                digest = _parse_prefixed(_readline_required(fh, "sha256"), "sha256")
                if _readline_required(fh, "content begin") != _LL_CONTENT_BEGIN:
                    raise ValueError(f"missing CONTENT BEGIN for {logical_path}")
                payload = fh.read(byte_count)
                if len(payload) != byte_count:
                    raise ValueError(f"truncated LLMBUNDLE payload: {logical_path}")
                if fh.read(1) != b"\n":
                    raise ValueError(f"missing LLMBUNDLE content separator: {logical_path}")
                if _readline_required(fh, "content end") != _LL_CONTENT_END:
                    raise ValueError(f"missing CONTENT END for {logical_path}")
                if _sha256(payload) != digest:
                    raise ValueError(f"LLMBUNDLE payload hash mismatch: {logical_path}")
                files[logical_path] = payload
            elif kind == "symlink":
                target_count = int(_parse_prefixed(_readline_required(fh, "target bytes"), "target-bytes"))
                target_digest = _parse_prefixed(_readline_required(fh, "target sha256"), "target-sha256")
                if _readline_required(fh, "target begin") != _LL_TARGET_BEGIN:
                    raise ValueError(f"missing TARGET BEGIN for {logical_path}")
                target_bytes = fh.read(target_count)
                if len(target_bytes) != target_count:
                    raise ValueError(f"truncated LLMBUNDLE symlink target: {logical_path}")
                if fh.read(1) != b"\n":
                    raise ValueError(f"missing LLMBUNDLE target separator: {logical_path}")
                if _readline_required(fh, "target end") != _LL_TARGET_END:
                    raise ValueError(f"missing TARGET END for {logical_path}")
                if _sha256(target_bytes) != target_digest:
                    raise ValueError(f"LLMBUNDLE target hash mismatch: {logical_path}")
                symlinks[logical_path] = target_bytes.decode("utf-8")
            elif kind == "binary-metadata-only":
                raise ValueError(
                    f"LLMBUNDLE omits binary payload bytes for {logical_path}; "
                    "cannot satisfy exact reviewer-packet round-trip"
                )
            else:
                raise ValueError(f"unknown LLMBUNDLE entry type {kind!r} for {logical_path}")

            if _readline_required(fh, "entry end") != _LL_ENTRY_END:
                raise ValueError(f"missing LLMBUNDLE ENTRY END for {logical_path}")

            line = _readline_required(fh, "next entry/trailer")
            while line == b"\n":
                line = _readline_required(fh, "next entry/trailer")
            if line == _LL_ENTRY_BEGIN:
                continue
            if line.startswith(b"===== LLMBUNDLE END "):
                trailer_line = line
                break
            raise ValueError(f"unexpected LLMBUNDLE record after {logical_path}: {line[:100]!r}")

    if header_entries is not None and header_entries != len(entry_paths):
        raise ValueError(f"LLMBUNDLE header entry count mismatch: {header_entries} != {len(entry_paths)}")
    if trailer_line is None:
        raise ValueError("LLMBUNDLE has no trailer")
    trailer = trailer_line.decode("utf-8", "strict")
    m_entries = re.search(r"entries=(\d+)", trailer)
    if m_entries and int(m_entries.group(1)) != len(entry_paths):
        raise ValueError("LLMBUNDLE trailer entry count mismatch")
    m_manifest = re.search(r"manifest-sha256=([0-9a-f]{64})", trailer)
    if header_manifest_sha and m_manifest and m_manifest.group(1) != header_manifest_sha:
        raise ValueError("LLMBUNDLE header/trailer manifest-sha256 mismatch")

    if MANIFEST_MEMBER not in files:
        raise ValueError("LLMBUNDLE missing embedded REVIEW_PACKET_MANIFEST.json")
    manifest = json.loads(files[MANIFEST_MEMBER].decode("utf-8"))
    validate_manifest(manifest)
    expected_entries = {MANIFEST_MEMBER} | {m["logical_path"] for m in manifest["members"]}
    actual_entries = set(entry_paths)
    if actual_entries != expected_entries:
        raise ValueError(
            f"LLMBUNDLE logical member-set mismatch: "
            f"extra={sorted(actual_entries - expected_entries)} "
            f"missing={sorted(expected_entries - actual_entries)}"
        )

    logical: dict[str, bytes] = {}
    rows = {m["logical_path"]: m for m in manifest["members"]}
    for logical_path, m in rows.items():
        if logical_path in symlinks:
            raise ValueError(
                f"LLMBUNDLE logical reviewer path unexpectedly encoded as symlink: {logical_path}"
            )
        if logical_path not in files:
            raise ValueError(f"LLMBUNDLE reviewer payload missing: {logical_path}")
        payload = files[logical_path]
        if len(payload) != m["byte_count"] or _sha256(payload) != m["content_sha256"]:
            raise ValueError(f"LLMBUNDLE reconstructed payload mismatch: {logical_path}")
        logical[logical_path] = payload
    return manifest, logical


def reconstruct_from_zip(zip_path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if MANIFEST_MEMBER not in names:
            raise ValueError("ZIP missing reviewer manifest")
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names")
        manifest = json.loads(zf.read(MANIFEST_MEMBER).decode("utf-8"))
        validate_manifest(manifest)
        expected_names = {MANIFEST_MEMBER} | {m["logical_path"] for m in manifest["members"]}
        actual_names = set(names)
        if actual_names != expected_names:
            raise ValueError(
                f"ZIP member-set mismatch: extra={sorted(actual_names - expected_names)} "
                f"missing={sorted(expected_names - actual_names)}"
            )
        info_by_name = {info.filename: info for info in infos}
        logical: dict[str, bytes] = {}
        for m in manifest["members"]:
            info = info_by_name[m["logical_path"]]
            if _zip_info_is_symlink(info):
                raise ValueError(f"ZIP logical reviewer path must be a regular file: {m['logical_path']}")
            payload = zf.read(info)
            if len(payload) != m["byte_count"] or _sha256(payload) != m["content_sha256"]:
                raise ValueError(f"ZIP reconstructed payload mismatch: {m['logical_path']}")
            logical[m["logical_path"]] = payload
        return manifest, logical


def reconstruct_from_tar(tar_path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    with tarfile.open(tar_path, "r") as tf:
        members = [m for m in tf.getmembers() if m.isfile() or m.issym() or m.islnk()]
        names = [m.name for m in members]
        if MANIFEST_MEMBER not in names:
            raise ValueError("TAR missing reviewer manifest")
        if len(names) != len(set(names)):
            raise ValueError("TAR contains duplicate member names")
        mf = tf.extractfile(MANIFEST_MEMBER)
        if mf is None:
            raise ValueError("TAR missing reviewer manifest")
        manifest = json.loads(mf.read().decode("utf-8"))
        validate_manifest(manifest)
        expected_names = {MANIFEST_MEMBER} | {m["logical_path"] for m in manifest["members"]}
        actual_names = set(names)
        if actual_names != expected_names:
            raise ValueError(
                f"TAR member-set mismatch: extra={sorted(actual_names - expected_names)} "
                f"missing={sorted(expected_names - actual_names)}"
            )
        by_name = {m.name: m for m in members}
        logical: dict[str, bytes] = {}
        for row in manifest["members"]:
            member = by_name[row["logical_path"]]
            if not member.isfile():
                raise ValueError(f"TAR logical reviewer path must be a regular file: {row['logical_path']}")
            pf = tf.extractfile(member)
            if pf is None:
                raise ValueError(f"TAR payload unavailable: {row['logical_path']}")
            payload = pf.read()
            if len(payload) != row["byte_count"] or _sha256(payload) != row["content_sha256"]:
                raise ValueError(f"TAR reconstructed payload mismatch: {row['logical_path']}")
            logical[row["logical_path"]] = payload
        return manifest, logical


def verify_root_against_manifest(root: Path, manifest: dict[str, Any]) -> None:
    for m in manifest["members"]:
        data = _read_bytes(root, m["logical_path"])
        if len(data) != m["byte_count"] or _sha256(data) != m["content_sha256"]:
            raise ValueError(f"source drift after manifest build: {m['logical_path']}")


def _assert_logical_equal(expected: dict[str, bytes], actual: dict[str, bytes], label: str) -> None:
    if set(expected) != set(actual):
        raise ValueError(f"{label} logical path-set mismatch")
    for path, data in expected.items():
        if actual[path] != data:
            raise ValueError(f"{label} logical byte mismatch: {path}")


def verify_packet(
    root: Path,
    manifest: dict[str, Any],
    zip_path: Path | None = None,
    tar_path: Path | None = None,
    bundle_path: Path | None = None,
) -> None:
    verify_root_against_manifest(root, manifest)
    expected = {m["logical_path"]: _read_bytes(root, m["logical_path"]) for m in manifest["members"]}
    if zip_path:
        zm, logical = reconstruct_from_zip(zip_path)
        if zm["logical_manifest_sha256"] != manifest["logical_manifest_sha256"]:
            raise ValueError("ZIP logical-manifest digest mismatch")
        _assert_logical_equal(expected, logical, "ZIP")
    if tar_path:
        tm, logical = reconstruct_from_tar(tar_path)
        if tm["logical_manifest_sha256"] != manifest["logical_manifest_sha256"]:
            raise ValueError("TAR logical-manifest digest mismatch")
        _assert_logical_equal(expected, logical, "TAR")
    if bundle_path:
        bm, logical = parse_llmbundle_for_verification(bundle_path)
        if bm["logical_manifest_sha256"] != manifest["logical_manifest_sha256"]:
            raise ValueError("LLMBUNDLE logical-manifest digest mismatch")
        _assert_logical_equal(expected, logical, "LLMBUNDLE")


def build_profile(
    root: Path,
    manifest: dict[str, Any],
    zip_path: Path | None,
    tar_path: Path | None,
    bundle_path: Path | None,
) -> dict[str, Any]:
    # Observational: only reads source/member metadata and container sizes.
    by_category: dict[str, dict[str, int]] = {}
    logical_chars_total = 0
    text_only = True
    for m in manifest["members"]:
        cat = m["content_category"]
        bucket = by_category.setdefault(cat, {"logical_members": 0, "logical_bytes": 0, "characters": 0})
        bucket["logical_members"] += 1
        bucket["logical_bytes"] += m["byte_count"]
        if m["character_count"] is None:
            text_only = False
        else:
            bucket["characters"] += m["character_count"]
            logical_chars_total += m["character_count"]

    profile = {
        "schema": "ADF_REVIEW_PACKET_PROFILE/1",
        "logical_manifest_sha256": manifest["logical_manifest_sha256"],
        "logical_member_count": manifest["logical_member_count"],
        "physical_payload_count": manifest["physical_payload_count"],
        "alias_count": manifest["alias_count"],
        "duplicate_group_count": manifest["duplicate_group_count"],
        "logical_expanded_bytes": manifest["logical_expanded_bytes"],
        "unique_payload_bytes": manifest["unique_payload_bytes"],
        "exact_duplicate_bytes": manifest["exact_duplicate_bytes"],
        "deduplication_ratio_unique_over_logical": manifest["deduplication_ratio_unique_over_logical"],
        "logical_text_characters": logical_chars_total if text_only else None,
        "token_estimate": (logical_chars_total + 3) // 4 if text_only else None,
        "token_estimate_method": "chars_div_4" if text_only else "unavailable_non_utf8_payload",
        "token_measurement": "estimated",
        "provider_quota": None,
        "provider_quota_note": "not inferred from bytes/characters/token estimate",
        "zip_physical_bytes": zip_path.stat().st_size if zip_path and zip_path.exists() else None,
        "tar_physical_bytes": tar_path.stat().st_size if tar_path and tar_path.exists() else None,
        "llmbundle_physical_bytes": bundle_path.stat().st_size if bundle_path and bundle_path.exists() else None,
        "by_category": dict(sorted(by_category.items())),
    }
    return profile


def profile_text(profile: dict[str, Any]) -> str:
    lines = [
        "ADF_REVIEW_PACKET_PROFILE/1",
        f"logical_manifest_sha256: {profile['logical_manifest_sha256']}",
        f"logical_members: {profile['logical_member_count']}",
        f"physical_payloads: {profile['physical_payload_count']}",
        f"aliases: {profile['alias_count']}",
        f"duplicate_groups: {profile['duplicate_group_count']}",
        f"logical_expanded_bytes: {profile['logical_expanded_bytes']}",
        f"unique_payload_bytes: {profile['unique_payload_bytes']}",
        f"exact_duplicate_bytes: {profile['exact_duplicate_bytes']}",
        f"deduplication_ratio_unique_over_logical: {profile['deduplication_ratio_unique_over_logical']:.6f}",
        f"logical_text_characters: {profile['logical_text_characters']}",
        f"token_estimate: {profile['token_estimate']}",
        f"token_estimate_method: {profile['token_estimate_method']}",
        "token_measurement: estimated",
        "provider_quota: NOT_MEASURED",
        f"zip_physical_bytes: {profile['zip_physical_bytes']}",
        f"tar_physical_bytes: {profile['tar_physical_bytes']}",
        f"llmbundle_physical_bytes: {profile['llmbundle_physical_bytes']}",
        "",
        "by_category:",
    ]
    for cat, row in profile["by_category"].items():
        lines.append(
            f"  {cat}: members={row['logical_members']} bytes={row['logical_bytes']} chars={row['characters']}"
        )
    return "\n".join(lines) + "\n"


def _paths_from_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _cmd_manifest(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    manifest = build_manifest(root, _paths_from_file(Path(args.files_file)))
    out = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(manifest_bytes(manifest))
    print(
        f"logical={manifest['logical_member_count']} physical={manifest['physical_payload_count']} "
        f"aliases={manifest['alias_count']} duplicate_groups={manifest['duplicate_group_count']} "
        f"logical_manifest_sha256={manifest['logical_manifest_sha256']}"
    )
    return 0


def _cmd_zip(args: argparse.Namespace) -> int:
    build_zip(Path(args.root).resolve(), load_manifest(Path(args.manifest)), Path(args.output))
    return 0


def _cmd_tar(args: argparse.Namespace) -> int:
    build_tar(Path(args.root).resolve(), load_manifest(Path(args.manifest)), Path(args.output))
    return 0




def _cmd_verify(args: argparse.Namespace) -> int:
    manifest = load_manifest(Path(args.manifest))
    verify_packet(
        Path(args.root).resolve(),
        manifest,
        Path(args.zip) if args.zip else None,
        Path(args.tar) if args.tar else None,
        Path(args.bundle) if args.bundle else None,
    )
    print(
        f"OK logical={manifest['logical_member_count']} physical={manifest['physical_payload_count']} "
        f"aliases={manifest['alias_count']} logical_manifest_sha256={manifest['logical_manifest_sha256']}"
    )
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    manifest = load_manifest(Path(args.manifest))
    before = {m["logical_path"]: _sha256(_read_bytes(root, m["logical_path"])) for m in manifest["members"]}
    profile = build_profile(
        root,
        manifest,
        Path(args.zip) if args.zip else None,
        Path(args.tar) if args.tar else None,
        Path(args.bundle) if args.bundle else None,
    )
    after = {m["logical_path"]: _sha256(_read_bytes(root, m["logical_path"])) for m in manifest["members"]}
    if before != after:
        raise RuntimeError("profiler mutated canonical evidence")
    Path(args.output).write_text(profile_text(profile), encoding="utf-8")
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"logical_bytes={profile['logical_expanded_bytes']} unique_bytes={profile['unique_payload_bytes']} "
        f"duplicate_bytes={profile['exact_duplicate_bytes']} token_estimate={profile['token_estimate']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="command", required=True)

    m = sp.add_parser("manifest", help="build per-packet logical manifest from current files")
    m.add_argument("--root", required=True)
    m.add_argument("--files-file", required=True)
    m.add_argument("--manifest", required=True)
    m.set_defaults(func=_cmd_manifest)

    for name, func in (("zip", _cmd_zip), ("tar", _cmd_tar)):
        q = sp.add_parser(name)
        q.add_argument("--root", required=True)
        q.add_argument("--manifest", required=True)
        q.add_argument("--output", required=True)
        q.set_defaults(func=func)


    v = sp.add_parser("verify", help="byte-for-byte logical expansion verification")
    v.add_argument("--root", required=True)
    v.add_argument("--manifest", required=True)
    v.add_argument("--zip")
    v.add_argument("--tar")
    v.add_argument("--bundle")
    v.set_defaults(func=_cmd_verify)

    pr = sp.add_parser("profile", help="read-only packet volume profiler")
    pr.add_argument("--root", required=True)
    pr.add_argument("--manifest", required=True)
    pr.add_argument("--zip")
    pr.add_argument("--tar")
    pr.add_argument("--bundle")
    pr.add_argument("--output", required=True)
    pr.add_argument("--json-output")
    pr.set_defaults(func=_cmd_profile)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"reviewer_packet.py: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
