#!/usr/bin/env python3
"""
make_llm_bundle.py

Create a deterministic plain-text source bundle from:
  - a directory
  - .zip
  - .tar
  - .tar.gz / .tgz
  - other tar formats recognized by Python's tarfile module

Output format: LLMBUNDLE/1

Text files:
  - path
  - byte size
  - SHA256
  - type=text
  - exact original UTF-8 bytes embedded in the bundle

Binary files:
  - path
  - byte size
  - SHA256
  - type=binary
  - no binary payload embedded

Symlinks:
  - path
  - type=symlink
  - target

The bundle itself is written as bytes and is deterministic for identical
logical input contents and paths. Archive timestamps, permissions, and member
ordering do not affect the output.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Optional


FORMAT_VERSION = b"LLMBUNDLE/2\n"

# A delimiter-scanning reader mis-splits the bundle only when a file contains
# a LINE that is exactly a delimiter.  A quoted mention inside code — a grep
# pattern in run_tests.sh, this script's own DELIMITERS constant, a diff that
# touches either — is harmless and must NOT be refused: those occurrences are
# permanent and unavoidable in any tree that ships these tools, so a substring
# test makes the guard fire always and therefore mean nothing.
DELIMITERS = (
    b"===== LLMBUNDLE ENTRY BEGIN =====",
    b"===== LLMBUNDLE ENTRY END =====",
    b"===== CONTENT BEGIN =====",
    b"===== CONTENT END =====",
    b"===== TARGET BEGIN =====",
    b"===== TARGET END =====",
    b"===== LLMBUNDLE END",
)


@dataclass(frozen=True)
class Entry:
    path: str
    kind: str  # "file" or "symlink"
    data: bytes = b""
    target: str = ""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_member_path(name: str) -> str:
    """
    Normalize archive/member paths to portable relative POSIX paths.
    Refuse absolute paths and '..' traversal components.
    """
    p = PurePosixPath(name.replace("\\", "/"))

    while p.parts and p.parts[0] == ".":
        p = PurePosixPath(*p.parts[1:])

    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe path in input: {name!r}")

    normalized = str(p)
    if normalized in ("", "."):
        raise ValueError(f"Invalid empty path in input: {name!r}")

    return normalized


def iter_directory(root: Path) -> Iterator[Entry]:
    root = root.resolve()

    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()

        if path.is_symlink():
            yield Entry(
                path=normalize_member_path(rel),
                kind="symlink",
                target=os.readlink(path),
            )
        elif path.is_file():
            yield Entry(
                path=normalize_member_path(rel),
                kind="file",
                data=path.read_bytes(),
            )


def zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def iter_zip(path: Path) -> Iterator[Entry]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = sorted(zf.infolist(), key=lambda i: normalize_member_path(i.filename.rstrip("/")) if i.filename.rstrip("/") else "")

        for info in infos:
            if info.is_dir():
                continue

            member_path = normalize_member_path(info.filename)

            if zip_member_is_symlink(info):
                target_bytes = zf.read(info)
                try:
                    target = target_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"ZIP symlink target is not UTF-8: {member_path}"
                    ) from exc
                yield Entry(path=member_path, kind="symlink", target=target)
            else:
                yield Entry(path=member_path, kind="file", data=zf.read(info))


def iter_tar(path: Path) -> Iterator[Entry]:
    with tarfile.open(path, "r:*") as tf:
        members = sorted(
            (m for m in tf.getmembers() if m.isfile() or m.issym() or m.islnk()),
            key=lambda m: normalize_member_path(m.name),
        )

        for member in members:
            member_path = normalize_member_path(member.name)

            if member.issym() or member.islnk():
                yield Entry(
                    path=member_path,
                    kind="symlink",
                    target=member.linkname,
                )
                continue

            f = tf.extractfile(member)
            if f is None:
                raise RuntimeError(f"Could not read TAR member: {member.name}")
            yield Entry(path=member_path, kind="file", data=f.read())


def detect_input_kind(path: Path) -> str:
    if path.is_dir():
        return "directory"
    if zipfile.is_zipfile(path):
        return "zip"
    if tarfile.is_tarfile(path):
        return "tar"
    raise ValueError(
        f"Unsupported input: {path}\n"
        "Expected a directory, ZIP, TAR, TAR.GZ/TGZ, or another tarfile-supported archive."
    )


def default_output_path(input_path: Path) -> Path:
    name = input_path.name

    lower = name.lower()
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".zip", ".tar"):
        if lower.endswith(suffix):
            name = name[: -len(suffix)]
            break

    if not name:
        name = "bundle"

    return input_path.with_name(name + ".llmbundle.txt")


def contains_delimiter_line(data: bytes) -> Optional[str]:
    """
    Return the offending delimiter if DATA contains a line that IS a delimiter.

    Line-anchored on purpose: `b"===== CONTENT END ====="` appearing inside a
    string literal or a grep pattern cannot confuse a reader that matches whole
    lines, whereas a bare delimiter line can.
    """
    padded = b"\n" + data + b"\n"
    for d in DELIMITERS:
        if b"\n" + d + b"\n" in padded:
            return d.decode("ascii")
        # trailing-content forms: "===== LLMBUNDLE END entries=... ====="
        idx = padded.find(b"\n" + d)
        while idx != -1:
            eol = padded.find(b"\n", idx + 1)
            if eol != -1 and padded[idx + 1:eol].startswith(d):
                seg = padded[idx + 1:eol]
                if seg == d or seg.endswith(b"====="):
                    return d.decode("ascii")
            idx = padded.find(b"\n" + d, idx + 1)
    return None


def is_utf8_text(data: bytes) -> bool:
    """
    Treat valid UTF-8 without NUL bytes as text.
    This preserves source/document files while avoiding embedding obvious binary data.
    """
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def iter_entries(input_path: Path) -> Iterable[Entry]:
    kind = detect_input_kind(input_path)
    if kind == "directory":
        return iter_directory(input_path)
    if kind == "zip":
        return iter_zip(input_path)
    if kind == "tar":
        return iter_tar(input_path)
    raise AssertionError(kind)


def make_llm_bundle(
    input_path: str | os.PathLike[str],
    output_path: Optional[str | os.PathLike[str]] = None,
    allow_delimiters: bool = False,
) -> dict:
    """
    Convert INPUT_PATH into one deterministic LLMBUNDLE/1 text file.

    Parameters
    ----------
    input_path:
        Directory, ZIP, TAR, TAR.GZ/TGZ, or another tarfile-supported archive.

    output_path:
        Optional output filename.
        Default: <input-base>.llmbundle.txt next to the input.

    Returns
    -------
    dict with:
        output_path
        output_sha256
        files
        text_files
        binary_files
        symlinks
        embedded_text_bytes
    """
    src = Path(input_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(src)

    dst = Path(output_path).expanduser() if output_path else default_output_path(src)

    # Materialize and sort so directory/archive input ordering cannot affect output.
    entries = sorted(iter_entries(src), key=lambda e: e.path)

    # Reject duplicate logical paths: they are ambiguous in a review artifact.
    seen = set()
    for entry in entries:
        if entry.path in seen:
            raise ValueError(f"Duplicate logical path in input: {entry.path}")
        seen.add(entry.path)

    # --- delimiter collision guard -------------------------------------
    if not allow_delimiters:
        collisions = []
        for entry in entries:
            if entry.kind == "symlink":
                continue
            hit = contains_delimiter_line(entry.data)
            if hit is not None:
                collisions.append(f"{entry.path}   (line is exactly: {hit})")
        if collisions:
            raise ValueError(
                "Refusing to build: these files contain an LLMBUNDLE delimiter, "
                "which makes the bundle ambiguous to a delimiter-scanning reader:\n  "
                + "\n  ".join(collisions)
                + "\nRe-run with allow_delimiters=True / --allow-delimiters only if "
                  "every consumer parses by the 'bytes:' length prefix."
            )

    # --- manifest digest, so a truncated bundle is DETECTABLE ------------
    manifest_lines = []
    for entry in entries:
        if entry.kind == "symlink":
            manifest_lines.append(
                f"symlink {sha256_bytes(entry.target.encode('utf-8'))} "
                f"{len(entry.target.encode('utf-8'))} {entry.path}"
            )
        else:
            manifest_lines.append(
                f"file {sha256_bytes(entry.data)} {len(entry.data)} {entry.path}"
            )
    manifest_blob = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    manifest_digest = sha256_bytes(manifest_blob)

    n_files = sum(1 for e in entries if e.kind != "symlink")
    n_symlinks = sum(1 for e in entries if e.kind == "symlink")
    n_text = sum(1 for e in entries if e.kind != "symlink" and is_utf8_text(e.data))
    n_binary = n_files - n_text

    stats = {
        "files": 0,
        "text_files": 0,
        "binary_files": 0,
        "symlinks": 0,
        "embedded_text_bytes": 0,
    }

    with dst.open("wb") as out:
        out.write(FORMAT_VERSION)
        out.write(f"entries: {len(entries)}\n".encode("ascii"))
        out.write(f"files: {n_files}\n".encode("ascii"))
        out.write(f"text-files: {n_text}\n".encode("ascii"))
        out.write(f"binary-files: {n_binary}\n".encode("ascii"))
        out.write(f"symlinks: {n_symlinks}\n".encode("ascii"))
        out.write(f"manifest-sha256: {manifest_digest}\n".encode("ascii"))
        out.write(b"===== LLMBUNDLE MANIFEST BEGIN =====\n")
        out.write(manifest_blob)
        out.write(b"===== LLMBUNDLE MANIFEST END =====\n")

        for entry in entries:
            out.write(b"\n===== LLMBUNDLE ENTRY BEGIN =====\n")
            out.write(f"path: {entry.path}\n".encode("utf-8"))

            if entry.kind == "symlink":
                stats["symlinks"] += 1
                target_bytes = entry.target.encode("utf-8")
                out.write(b"type: symlink\n")
                out.write(f"target-bytes: {len(target_bytes)}\n".encode("ascii"))
                out.write(f"target-sha256: {sha256_bytes(target_bytes)}\n".encode("ascii"))
                out.write(b"===== TARGET BEGIN =====\n")
                out.write(target_bytes)
                out.write(b"\n===== TARGET END =====\n")
                out.write(b"===== LLMBUNDLE ENTRY END =====\n")
                continue

            stats["files"] += 1
            data = entry.data
            digest = sha256_bytes(data)

            if is_utf8_text(data):
                stats["text_files"] += 1
                stats["embedded_text_bytes"] += len(data)

                out.write(b"type: text\n")
                out.write(b"encoding: utf-8\n")
                out.write(f"bytes: {len(data)}\n".encode("ascii"))
                out.write(f"sha256: {digest}\n".encode("ascii"))
                out.write(b"===== CONTENT BEGIN =====\n")
                out.write(data)
                out.write(b"\n===== CONTENT END =====\n")
            else:
                stats["binary_files"] += 1

                out.write(b"type: binary-metadata-only\n")
                out.write(f"bytes: {len(data)}\n".encode("ascii"))
                out.write(f"sha256: {digest}\n".encode("ascii"))
                out.write(b"content-embedded: no\n")

            out.write(b"===== LLMBUNDLE ENTRY END =====\n")

        out.write(
            f"\n===== LLMBUNDLE END entries={len(entries)} "
            f"manifest-sha256={manifest_digest} =====\n".encode("ascii")
        )

    bundle_hash = hashlib.sha256(dst.read_bytes()).hexdigest()

    result = {
        "output_path": str(dst),
        "output_sha256": bundle_hash,
        "manifest_sha256": manifest_digest,
        "entries": len(entries),
        **stats,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a deterministic plain-text LLMBUNDLE/1 from a directory or archive."
    )
    parser.add_argument(
        "input",
        help="Input directory, .zip, .tar, .tar.gz/.tgz, or other tarfile-supported archive",
    )
    parser.add_argument(
        "--allow-delimiters",
        action="store_true",
        help="Emit even if a file contains an LLMBUNDLE delimiter (ambiguous to eye-readers)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path. Default: <input-base>.llmbundle.txt",
    )

    args = parser.parse_args()

    result = make_llm_bundle(args.input, args.output, allow_delimiters=args.allow_delimiters)

    print(f"output:              {result['output_path']}")
    print(f"entries:             {result['entries']}")
    print(f"manifest_sha256:     {result['manifest_sha256']}")
    print(f"output_sha256:       {result['output_sha256']}")
    print(f"files:               {result['files']}")
    print(f"text_files:          {result['text_files']}")
    print(f"binary_files:        {result['binary_files']}")
    print(f"symlinks:            {result['symlinks']}")
    print(f"embedded_text_bytes: {result['embedded_text_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
