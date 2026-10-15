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
#   install_newest_by_md5 <glob> <dst> <expected_md5> [--delete-matching]
#   check_ref <file> <expected_md5> [expected_sha256]
#
#   install_by_md5 "$Downloads/foo_rev3c.py" tests/foo.py 3a57e2e4...
#   install_newest_by_md5 'foo_rev3c*.py' tests/foo.py 3a57e2e4...
#   install_newest_by_md5 'foo_rev3c*.py' tests/foo.py 3a57e2e4... --delete-matching
#   check_ref run_tests.sh <md5> <sha256>
#
# BEHAVIOUR
#   install_by_md5:
#   * refuses if <src> does not exist
#   * refuses if <src> does not match <expected_md5>       -- nothing is copied
#   * backs up an existing <dst> to <dst>.bak.<timestamp>
#   * re-verifies <dst> after copying and refuses to report success otherwise
#
#   install_newest_by_md5:
#   * searches ALL $Downloads files matching <glob>
#   * computes each candidate's MD5 and keeps only exact <expected_md5> matches
#   * installs the NEWEST exact-MD5 match through install_by_md5
#   * reports total glob matches and exact-MD5 matches
#   * KEEPS Downloads files by default
#   * with --delete-matching, removes only glob candidates whose MD5 equals
#     <expected_md5>, and only AFTER a successful install
#
#   check_ref:
#   * compares an existing file against a recorded MD5 and optional SHA256
#   * prints only OK / NOT OK / MISSING for normal interactive use
#   * returns 0 only when every supplied digest matches
#   * never copies or modifies the checked file
#
#   All functions return non-zero on failure and do not terminate the
#   interactive shell.
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

_ibm5_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$1" | awk '{print $NF}'
    else
        echo "check_ref: no SHA256 tool available (sha256sum/shasum/openssl)" >&2
        return 1
    fi
}

check_ref() {
    local file expected_md5 expected_sha256 actual_md5 actual_sha256

    if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
        cat >&2 <<'USAGE'
usage: check_ref <file> <expected_md5> [expected_sha256]

  <file>             existing worktree/reference file
  <expected_md5>     recorded MD5 for the exact reference bytes
  [expected_sha256]  optional recorded SHA256; if supplied, BOTH must match

Output is intentionally compact:
  OK      <file>
  NOT OK  <file>
  MISSING <file>

No file is modified.
USAGE
        return 2
    fi

    file="$1"
    expected_md5="$2"
    expected_sha256="${3:-}"

    if [ ! -f "$file" ]; then
        printf 'MISSING %s\n' "$file"
        return 1
    fi

    actual_md5="$(_ibm5_md5 "$file")" || {
        printf 'NOT OK  %s\n' "$file"
        return 1
    }

    if [ "$actual_md5" != "$expected_md5" ]; then
        printf 'NOT OK  %s\n' "$file"
        return 1
    fi

    if [ -n "$expected_sha256" ]; then
        actual_sha256="$(_ibm5_sha256 "$file")" || {
            printf 'NOT OK  %s\n' "$file"
            return 1
        }
        if [ "$actual_sha256" != "$expected_sha256" ]; then
            printf 'NOT OK  %s\n' "$file"
            return 1
        fi
    fi

    printf 'OK      %s\n' "$file"
    return 0
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

install_newest_by_md5() {
    local pattern dst want delete_matching=0
    local src="" candidate actual
    local n=0 n_match=0 n_deleted=0

    if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
        cat >&2 <<'USAGE'
usage: install_newest_by_md5 <glob> <dst> <expected_md5> [--delete-matching]

  <glob>              shell glob relative to $Downloads
  <dst>               destination path
  <expected_md5>      reviewed MD5 to select among all glob matches
  --delete-matching   optional; after successful install, delete only
                      Downloads candidates whose MD5 equals <expected_md5>

Default: keep every file in $Downloads.
USAGE
        return 2
    fi

    pattern="$1"
    dst="$2"
    want="$3"

    if [ "$#" -eq 4 ]; then
        if [ "$4" != "--delete-matching" ]; then
            echo "install_newest_by_md5: unknown option: $4" >&2
            echo "usage: install_newest_by_md5 <glob> <dst> <expected_md5> [--delete-matching]" >&2
            return 2
        fi
        delete_matching=1
    fi

    if [ -z "${Downloads:-}" ]; then
        echo "install_newest_by_md5: \$Downloads is not set" >&2
        return 2
    fi

    # Scan every filename matching the glob.  Selection is by reviewed MD5
    # first, then by modification time among the exact-byte matches.
    while IFS= read -r candidate; do
        [ -n "$candidate" ] || continue
        [ -f "$candidate" ] || continue

        n=$((n + 1))
        actual="$(_ibm5_md5 "$candidate")" || {
            echo "install_newest_by_md5: cannot compute MD5: $candidate" >&2
            continue
        }

        if [ "$actual" = "$want" ]; then
            n_match=$((n_match + 1))
            if [ -z "$src" ] || [ "$candidate" -nt "$src" ]; then
                src="$candidate"
            fi
        fi
    done < <(compgen -G "$Downloads/$pattern")

    if [ "$n" -eq 0 ]; then
        echo "install_newest_by_md5: nothing matches $Downloads/$pattern" >&2
        return 1
    fi

    if [ "$n_match" -eq 0 ]; then
        echo "install_newest_by_md5: $n file(s) match the glob, but none has the expected MD5" >&2
        echo "  pattern  : $Downloads/$pattern" >&2
        echo "  expected : $want" >&2
        echo "  candidates:" >&2

        while IFS= read -r candidate; do
            [ -n "$candidate" ] || continue
            [ -f "$candidate" ] || continue
            actual="$(_ibm5_md5 "$candidate" 2>/dev/null || printf '<MD5-ERROR>')"
            printf '    %s  %s\n' "$actual" "$candidate" >&2
        done < <(compgen -G "$Downloads/$pattern")

        echo "  Nothing was installed or deleted." >&2
        return 1
    fi

    echo "  using $src"
    echo "  glob matches: $n; exact-MD5 matches: $n_match; downloads: $([ "$delete_matching" -eq 1 ] && printf 'delete exact-MD5 matches after success' || printf 'keep')"

    install_by_md5 "$src" "$dst" "$want" || return 1

    if [ "$delete_matching" -eq 1 ]; then
        while IFS= read -r candidate; do
            [ -n "$candidate" ] || continue
            [ -f "$candidate" ] || continue

            actual="$(_ibm5_md5 "$candidate")" || {
                echo "install_newest_by_md5: warning: cannot compute MD5 for cleanup candidate: $candidate" >&2
                continue
            }

            if [ "$actual" = "$want" ]; then
                rm -f -- "$candidate" || {
                    echo "install_newest_by_md5: warning: could not delete $candidate" >&2
                    continue
                }
                n_deleted=$((n_deleted + 1))
            fi
        done < <(compgen -G "$Downloads/$pattern")

        echo "install_newest_by_md5: deleted $n_deleted exact-MD5 matching download(s); non-matching glob files were kept"
    fi

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
