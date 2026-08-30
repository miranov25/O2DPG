#!/bin/bash
# ============================================================================
# make_audit_snapshot.sh  v03 — anchored source snapshots for reviewers
# PHASE_13_78_ADF
#
# WHY THIS EXISTS
#   In the PHASE_13_76 B3.2b three-audit round, three independent auditors
#   audited three different byte sets, because the source archive carried no
#   commit, no dirty flag and no manifest.  Not one audit question was
#   answered against a common snapshot.  (Aggregation finding D-1.)
#
# v03 — panel revision, PHASE_13_78 Official Review Summary [X]
#   R1  tracked files DELETED from disk are no longer counted or claimed as
#       included; they are recorded in _AUDIT_DELETED_TRACKED.txt.
#   R2  ref mode no longer labels the generation-time repository HEAD as the
#       snapshot commit.  SOURCE_COMMIT and REPO_HEAD are separate fields.
#   R3  fail closed: manifest count == archived source count, or the zip is
#       removed and the exit status is non-zero.  No valid-looking partial
#       archive can be produced.
#   P2-A  key-fingerprint match is now basename-anchored, so dfdraw/gb report
#         their own key files instead of an empty block.
#   P2-C  the script fingerprints itself into _AUDIT_HEAD.txt.
#
# USAGE
#   make_audit_snapshot.sh <src_dir> <name> [out_dir] [git_ref]
#
#     src_dir   a git working tree      e.g. ../../AliasDataFrame
#     name      short package label     e.g. adf | dfdraw | gb
#     out_dir   destination             default: $PWD
#     git_ref   OPTIONAL commit/tag.    When given, the snapshot is taken
#               from that ref via `git archive` — the working tree is NEVER
#               touched, nothing is checked out, stashed or modified.
#
# ENVIRONMENT (optional)
#   ADF_SNAP_DROP_EXT   extensions to exclude, regex alternation
#                       default: pkl|root|npy|npz|so|pyc|o|a|whl
#   ADF_SNAP_KEYFILES   basename regex of "key files" echoed in the report
#                       default: AliasDataFrame\.py|dfdraw\.py|.*characterization\.py
#
# EXAMPLES
#   make_audit_snapshot.sh ../../AliasDataFrame adf    .
#   make_audit_snapshot.sh ../../dfdraw         dfdraw .
#   make_audit_snapshot.sh ../../AliasDataFrame adf    .  92e2098b
#
# Exit 0 = a COMPLETE, self-consistent snapshot was written.
# Non-zero = nothing usable was left behind.
# ============================================================================

set -u

SCRIPT_PATH="${BASH_SOURCE[0]}"
SRC="${1:-}"
NAME="${2:-}"
OUT="${3:-$PWD}"
REF="${4:-}"

DROP_EXT="${ADF_SNAP_DROP_EXT:-pkl|root|npy|npz|so|pyc|o|a|whl}"
KEYFILES="${ADF_SNAP_KEYFILES:-AliasDataFrame\.py|dfdraw\.py|.*characterization\.py}"

ZIP=""   # set later; referenced by die()

die() {
    echo "ERROR: $*" >&2
    # R3: never leave a valid-looking partial archive behind.
    if [ -n "$ZIP" ] && [ -f "$ZIP" ]; then
        rm -f "$ZIP"
        echo "ERROR: removed incomplete archive $ZIP" >&2
    fi
    exit 1
}

if [ -z "$SRC" ] || [ -z "$NAME" ]; then
    echo "usage: make_audit_snapshot.sh <src_dir> <name> [out_dir] [git_ref]" >&2
    exit 2
fi

# --- preflight -------------------------------------------------------------
for _cmd in git zip unzip md5sum sha256sum tar xargs; do
    command -v "$_cmd" >/dev/null 2>&1 || { echo "missing command: $_cmd" >&2; exit 2; }
done

SRC=$(cd "$SRC" 2>/dev/null && pwd) || { echo "no such dir: $1" >&2; exit 2; }
mkdir -p "$OUT" 2>/dev/null || { echo "cannot create out_dir: $OUT" >&2; exit 2; }
OUT=$(cd "$OUT" && pwd)

cd "$SRC" || exit 2
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "not a git work tree: $SRC" >&2; exit 2; }

TOPLEVEL=$(git rev-parse --show-toplevel)
PREFIX=$(git rev-parse --show-prefix)
PREFIX_LABEL="${PREFIX:-<repo root>}"
# `<rev>:<path>` resolves relative to the CURRENT directory (git 2.43),
# so the archive below is always run from the repo toplevel.
ARCHIVE_TREEISH_PATH="${PREFIX%/}"

REPO_HEAD=$(git rev-parse HEAD)

if [ -n "$REF" ]; then
    git rev-parse --verify --quiet "${REF}^{commit}" >/dev/null || {
        echo "not a commit: $REF" >&2; exit 2; }
    REF_FULL=$(git rev-parse "${REF}^{commit}")
    MODE="ref"
    SOURCE_COMMIT="$REF_FULL"
    STATE="clean@${REF_FULL}"
else
    REF_FULL=""
    MODE="worktree"
    SOURCE_COMMIT=""   # decided below, after dirtiness is known
    STATE=""
fi

STAMP=$(date -u +%Y%m%d_%H%M%S)
if [ "$MODE" = "ref" ]; then
    ZIP="$OUT/sources_${NAME}_$(git rev-parse --short "$REF_FULL")_${STAMP}.zip"
else
    ZIP="$OUT/sources_${NAME}_${STAMP}.zip"
fi

WORK=$(mktemp -d) || exit 2
trap 'rm -rf "$WORK"' EXIT
META="$WORK/meta"; TREE="$WORK/tree"
mkdir -p "$META" "$TREE" || exit 2

# ---------------------------------------------------------------------------
# 1. materialise the file set
# ---------------------------------------------------------------------------
if [ "$MODE" = "ref" ]; then
    git -C "$TOPLEVEL" archive "${REF_FULL}:${ARCHIVE_TREEISH_PATH}" \
        | tar -x -C "$TREE" \
        || die "git archive failed for ${REF_FULL}:${ARCHIVE_TREEISH_PATH}"
    ( cd "$TREE" && find . -type f -printf '%P\n' ) | sort > "$WORK/all.txt"
    : > "$META/_AUDIT_DELETED_TRACKED.txt"
    NDELETED=0
    PAYLOAD_ROOT="$TREE"
else
    # R1: git ls-files lists tracked files that no longer exist on disk.
    # v02 counted and claimed them, then shipped a zip without them, exit 0.
    git ls-files --deleted | sort > "$META/_AUDIT_DELETED_TRACKED.txt"
    NDELETED=$(wc -l < "$META/_AUDIT_DELETED_TRACKED.txt")
    if [ "$NDELETED" -gt 0 ]; then
        git ls-files | sort \
          | grep -vxF -f "$META/_AUDIT_DELETED_TRACKED.txt" > "$WORK/all.txt"
    else
        git ls-files | sort > "$WORK/all.txt"
        echo "(no tracked files are missing from disk)" \
            > "$META/_AUDIT_DELETED_TRACKED.txt"
    fi
    PAYLOAD_ROOT="$SRC"
fi

grep -vE "\.(${DROP_EXT})\$" "$WORK/all.txt" > "$META/_AUDIT_FILES.txt" || true
grep -E  "\.(${DROP_EXT})\$" "$WORK/all.txt" > "$META/_AUDIT_EXCLUDED.txt" || true
NFILES=$(wc -l < "$META/_AUDIT_FILES.txt")

# R3
[ "$NFILES" -gt 0 ] || die "no files to archive after filtering (drop_ext=$DROP_EXT)"

# R1: every listed file must actually exist before we claim it.
MISSING=0
while IFS= read -r f; do
    [ -f "$PAYLOAD_ROOT/$f" ] || { echo "$f" >> "$WORK/missing.txt"; MISSING=1; }
done < "$META/_AUDIT_FILES.txt"
[ "$MISSING" -eq 0 ] || {
    echo "listed but not present on disk:" >&2
    sed 's/^/    /' "$WORK/missing.txt" >&2
    die "file list disagrees with the filesystem"
}

# ---------------------------------------------------------------------------
# 2. anchor
# ---------------------------------------------------------------------------
if [ "$MODE" = "worktree" ]; then
    # scoped to THIS package, not the whole repository
    git status --porcelain -- . > "$META/_AUDIT_DIRTY.txt" || true
    if [ -s "$META/_AUDIT_DIRTY.txt" ]; then
        STATE="DIRTY"
        SOURCE_COMMIT="${REPO_HEAD} + UNCOMMITTED WORKING-TREE CHANGES"
    else
        STATE="clean"
        SOURCE_COMMIT="$REPO_HEAD"
        echo "(this package is clean at HEAD)" > "$META/_AUDIT_DIRTY.txt"
    fi

    git ls-files --others --exclude-standard > "$META/_AUDIT_UNTRACKED.txt" || true
    NUNTRACKED=$(wc -l < "$META/_AUDIT_UNTRACKED.txt")
    grep -E '\.py$' "$META/_AUDIT_UNTRACKED.txt" \
        > "$META/_AUDIT_UNTRACKED_PY.txt" || true
    NUNTRACKED_PY=$(wc -l < "$META/_AUDIT_UNTRACKED_PY.txt")
else
    echo "(snapshot taken from ${REF_FULL} via git archive; working tree not consulted)" \
        > "$META/_AUDIT_DIRTY.txt"
    : > "$META/_AUDIT_UNTRACKED.txt"
    : > "$META/_AUDIT_UNTRACKED_PY.txt"
    NUNTRACKED=0
    NUNTRACKED_PY=0
fi

TOOL_MD5=$(md5sum "$SCRIPT_PATH" 2>/dev/null | cut -d' ' -f1)

{
    echo "package               : $NAME"
    echo "source_dir            : $SRC"
    echo "repo_prefix           : $PREFIX_LABEL"
    echo "snapshot_utc          : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host                  : $(hostname)"
    echo "snapshot_mode         : $MODE"
    echo
    echo "# R2: SOURCE_COMMIT is where the ARCHIVED BYTES come from."
    echo "#     REPO_HEAD is only where the repository happened to be when"
    echo "#     this snapshot was generated.  In ref mode they differ."
    echo "SOURCE_COMMIT         : $SOURCE_COMMIT"
    echo "repo_head_at_generation: $REPO_HEAD"
    if [ "$MODE" = "ref" ]; then
        echo "snapshot_ref_requested: $REF"
    fi
    echo "state                 : $STATE"
    echo "branch_at_generation  : $(git rev-parse --abbrev-ref HEAD)"
    echo "files_included        : $NFILES"
    echo "deleted_tracked       : $NDELETED   (excluded, see _AUDIT_DELETED_TRACKED.txt)"
    echo "drop_ext              : $DROP_EXT"
    echo "generator_md5         : $TOOL_MD5   (make_audit_snapshot.sh)"
    echo
    if [ "$MODE" = "ref" ]; then
        echo "--- git log -5 of SOURCE_COMMIT ($REF_FULL) ---"
        git log --oneline -5 "$REF_FULL"
        echo
        echo "--- git log -5 of REPO_HEAD at generation (context only) ---"
    else
        echo "--- git log -5 of SOURCE_COMMIT / REPO_HEAD ---"
    fi
    git log --oneline -5
} > "$META/_AUDIT_HEAD.txt"

# ---------------------------------------------------------------------------
# 3. per-file md5 manifest  (R3: failure is fatal)
# ---------------------------------------------------------------------------
( cd "$PAYLOAD_ROOT" \
    && tr '\n' '\0' < "$META/_AUDIT_FILES.txt" | xargs -0 md5sum ) \
    > "$META/_AUDIT_MD5.txt" 2>"$WORK/md5.err" \
    || { sed 's/^/    /' "$WORK/md5.err" >&2; die "md5sum failed on one or more files"; }

NMD5=$(wc -l < "$META/_AUDIT_MD5.txt")
[ "$NMD5" -eq "$NFILES" ] \
    || die "manifest mismatch: _AUDIT_FILES.txt=$NFILES but _AUDIT_MD5.txt=$NMD5"

# ---------------------------------------------------------------------------
# 4. environment
# ---------------------------------------------------------------------------
{
    echo "python         : $(python3 -V 2>&1)"
    echo "python_path    : $(command -v python3)"
    echo "VIRTUAL_ENV    : ${VIRTUAL_ENV:-<none>}"
    echo "uname          : $(uname -srm)"
    echo
    echo "--- pip freeze ---"
    python3 -m pip freeze 2>/dev/null
} > "$META/_AUDIT_ENV.txt"

# ---------------------------------------------------------------------------
# 5. build the zip  (sources first, verify, then metadata)
# ---------------------------------------------------------------------------
rm -f "$ZIP"
( cd "$PAYLOAD_ROOT" \
    && tr '\n' '\0' < "$META/_AUDIT_FILES.txt" | xargs -0 zip -q "$ZIP" ) \
    || die "zip failed while adding source files"
[ -f "$ZIP" ] || die "zip produced no archive"

# R3: what is actually IN the archive must equal what we claim.
NZIP=$(unzip -Z1 "$ZIP" | grep -cv '/$')
[ "$NZIP" -eq "$NFILES" ] \
    || die "archive mismatch: claimed $NFILES source files, archive holds $NZIP"

( cd "$META" && zip -q "$ZIP" _AUDIT_*.txt ) || die "zip failed while adding metadata"

# ---------------------------------------------------------------------------
# 6. report
# ---------------------------------------------------------------------------
ZMD5=$(md5sum "$ZIP" | cut -d' ' -f1)
ZSHA=$(sha256sum "$ZIP" | cut -d' ' -f1)

echo "==================================================================="
echo "SNAPSHOT   $ZIP"
echo "  package        $NAME          mode: $MODE"
echo "  SOURCE_COMMIT  $SOURCE_COMMIT"
echo "  REPO_HEAD      $REPO_HEAD   (generation context only)"
echo "  state          $STATE"
echo "  files          $NFILES included, verified present in the archive"
if [ "$NDELETED" -gt 0 ]; then
    echo "  deleted        $NDELETED tracked file(s) missing from disk, EXCLUDED:"
    sed 's/^/                   /' "$META/_AUDIT_DELETED_TRACKED.txt"
fi
echo "  md5            $ZMD5"
echo "  sha256         $ZSHA"
echo "  generator      $TOOL_MD5"
echo
echo "  key file fingerprints:"
grep -E "( |/)(${KEYFILES})\$" "$META/_AUDIT_MD5.txt" | sed 's/^/    /' \
    || echo "    (none matched ADF_SNAP_KEYFILES — set it per package)"
echo
if [ "$STATE" = "DIRTY" ]; then
    echo "  *** THIS PACKAGE IS DIRTY — the zip contains UNCOMMITTED bytes."
    echo "  *** Tell reviewers exactly what these bytes are."
    echo "  *** For a banked commit instead, pass a git_ref as argument 4."
    echo
fi
if [ "$NUNTRACKED_PY" -gt 0 ]; then
    echo "  *** $NUNTRACKED_PY UNTRACKED .py FILE(S) ARE MISSING FROM THE SNAPSHOT:"
    sed 's/^/    /' "$META/_AUDIT_UNTRACKED_PY.txt"
    echo "  *** If any is a new test, git add it and re-run."
    echo
elif [ "$MODE" = "worktree" ]; then
    echo "  no untracked .py files — nothing source-relevant is missing"
    echo "  ($NUNTRACKED untracked non-source files ignored: .bak, .log, docs, …)"
    echo
fi
echo "  Paste into the audit prompt §1:"
echo "      $NAME   $(basename "$ZIP")"
echo "      md5              $ZMD5"
echo "      SOURCE_COMMIT    $SOURCE_COMMIT"
echo "      state            $STATE"
echo "      REPO_HEAD        $REPO_HEAD  (generation context, NOT the source)"
echo "      env              $(python3 -V 2>&1)"
echo "==================================================================="
