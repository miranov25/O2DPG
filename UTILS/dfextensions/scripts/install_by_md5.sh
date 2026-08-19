# shellcheck shell=bash
#
# scripts/install_by_md5.sh — deliver a reviewed file into the working tree,
# but ONLY if it is byte-for-byte the file that was reviewed.
#
# WHY THIS EXISTS
#   Files arrive from a coder session by download. A stale file left in
#   Downloads under the same name is indistinguishable from the new one by
#   eye, and installing the wrong bytes has already cost this project a
#   round: on 29 July an identically-named stale download was installed and
#   the MD5 mismatch was only noticed after a full test run. The fix is to
#   make the checksum a precondition of the copy, not a check afterwards.
#
#   It also used to live only in the shell history, so a machine restart lost
#   it. It belongs in the repository.
#
# USAGE
#   source scripts/install_by_md5.sh          # once per shell
#   install_by_md5 <src> <dst> <expected_md5>
#
#   install_by_md5 "$Downloads/foo_rev3c.py" tests/foo.py 3a57e2e4...
#
# BEHAVIOUR
#   * refuses if <src> does not exist
#   * refuses if <src> does not match <expected_md5>       -- nothing is copied
#   * backs up an existing <dst> to <dst>.bak.<timestamp>
#   * re-verifies <dst> after copying and refuses to report success otherwise
#   * returns non-zero on every failure; NEVER calls `exit`, because this file
#     is sourced into an interactive shell and `exit` would close it
#
# NOTES
#   Linux/alma2: uses md5sum. A darwin fallback to `md5 -q` is included so the
#   same file works if the tree is ever checked out elsewhere.

_ibm5_md5() {
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" | awk '{print $1}'
    elif command -v md5 >/dev/null 2>&1; then
        md5 -q "$1"
    else
        echo "install_by_md5: neither md5sum nor md5 is available" >&2
        return 1
    fi
}

install_by_md5() {
    local src dst expected actual backup

    if [ "$#" -ne 3 ]; then
        cat >&2 <<'USAGE'
usage: install_by_md5 <src> <dst> <expected_md5>

  <src>           file to install, e.g. "$Downloads/thing_rev3c.py"
  <dst>           path in the working tree, e.g. tests/thing.py
  <expected_md5>  the MD5 quoted in the CRR

Nothing is copied unless <src> matches <expected_md5> exactly.
USAGE
        return 2
    fi

    src="$1"; dst="$2"; expected="$3"

    if [ ! -f "$src" ]; then
        echo "install_by_md5: SOURCE NOT FOUND: $src" >&2
        return 1
    fi

    actual="$(_ibm5_md5 "$src")" || return 1

    if [ "$actual" != "$expected" ]; then
        echo "install_by_md5: MD5 MISMATCH -- nothing was installed" >&2
        echo "  source   : $src" >&2
        echo "  expected : $expected" >&2
        echo "  actual   : $actual" >&2
        echo "  Most likely a STALE file of the same name is in the download" >&2
        echo "  directory. Delete it and download the delivered file again." >&2
        return 1
    fi

    mkdir -p "$(dirname "$dst")" || return 1

    if [ -e "$dst" ]; then
        backup="${dst}.bak.$(date +%Y%m%d_%H%M%S)"
        cp -p "$dst" "$backup" || return 1
        echo "install_by_md5: previous version saved to $backup"
    fi

    cp -p "$src" "$dst" || return 1

    actual="$(_ibm5_md5 "$dst")" || return 1
    if [ "$actual" != "$expected" ]; then
        echo "install_by_md5: POST-COPY VERIFY FAILED for $dst" >&2
        echo "  expected : $expected" >&2
        echo "  actual   : $actual" >&2
        return 1
    fi

    echo "install_by_md5: OK  $dst  $actual"
    return 0
}

# Convenience: many of these commands begin with D="$Downloads". If the
# variable is not set, point it at the obvious location -- but never override
# a value the user has already chosen.
if [ -z "${Downloads:-}" ] && [ -d "$HOME/Downloads" ]; then
    Downloads="$HOME/Downloads"
    export Downloads
fi

if [ -n "${BASH_SOURCE:-}" ] && [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "install_by_md5.sh defines a shell function; SOURCE it, do not run it:" >&2
    echo "    source scripts/install_by_md5.sh" >&2
    return 1 2>/dev/null || exit 1
fi
