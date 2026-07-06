#!/bin/bash
# =============================================================================
# scripts/phase_tag.sh — AliasDataFrame Phase Boundary Management (v3, hardened)
# Ref: BUG_AliasDataFrame_20260706_phase_tagging
# =============================================================================
# Authoritative moving pointer: PHASE_BEGIN_AliasDataFrame (SOLE source of truth
# for the harvest / diff_to_phase anchor). The stray pointers PHASE_BEGIN_ADF and
# bare PHASE_BEGIN are HISTORICAL/DEPRECATED — this script never reads or updates
# them; do not resurrect them. (Naming convention otherwise unchanged.)
#
# v2 hardening (9-reviewer panel, 2026-07-06):
#   - phase_end REFUSES on missing / diverged / empty (zero-commit) BEGIN; --force override.
#   - phase_begin: optional commit-ish (retroactive begin); open-phase warning.
# v3 hardening (7-reviewer panel, 2026-07-06 — all found by RUNNING, not reading):
#   - (Sonnet22) phase_end REFUSES an OUT-OF-ORDER close: if a newer PHASE_*_ADF_BEGIN
#     lies strictly after this BEGIN and is reachable from HEAD, closing here would
#     silently absorb the later phase's commits. --force override.
#   - (doc48) phase_end's "already exists" messages dereference to the COMMIT (^{commit}),
#     matching phase_begin — so pasted hashes are usable in `git log`.
#   - (Sonnet23) phase_label / phase_begin print the anchor via `git tag --points-at`
#     (exact) instead of `git describe` (ambiguous when tags coincide on one commit).
#
# Usage:
#   source scripts/phase_tag.sh
#   phase_begin 13_71_ADF [commit-ish]     # BEGIN (run BEFORE first commit)
#   phase_end   13_71_ADF [--force]        # close; refuses on missing/diverged/empty/out-of-order
#   phase_label                            # show current phase anchor
# =============================================================================

SUBPROJECT="AliasDataFrame"
TAGSUFFIX="ADF"

# short COMMIT hash a tag points at (deref annotated tag object -> commit)
_pt_commit() { git rev-parse --short "$1^{commit}" 2>/dev/null; }

phase_begin() {
    local phase="$1" at="${2:-HEAD}"
    if [ -z "$phase" ]; then
        echo "Usage: phase_begin <phase_id> [commit-ish]  (e.g. 13_71_ADF)"; return 1
    fi
    if git rev-parse -q --verify "refs/tags/PHASE_${phase}_BEGIN" >/dev/null; then
        echo "refusing: PHASE_${phase}_BEGIN already exists at $(_pt_commit "PHASE_${phase}_BEGIN")"
        return 1
    fi
    if ! git rev-parse -q --verify "${at}^{commit}" >/dev/null; then
        echo "refusing: commit-ish '${at}' does not resolve to a commit."; return 1
    fi
    local prev_open
    prev_open=$(comm -23 \
        <(git tag --list "PHASE_*_${TAGSUFFIX}_BEGIN" 2>/dev/null | sed 's/_BEGIN$//' | sort) \
        <(git tag --list "PHASE_*_${TAGSUFFIX}_END"   2>/dev/null | sed 's/_END$//'   | sort))
    [ -n "$prev_open" ] && echo "WARNING: open ${TAGSUFFIX} phase(s) without END: $(echo $prev_open | tr '\n' ' ')"

    git tag -a "PHASE_${phase}_BEGIN" -m "Phase ${phase//_/.} begin" "$at"
    git tag -f "PHASE_BEGIN_${SUBPROJECT}" "PHASE_${phase}_BEGIN"
    echo "Created PHASE_${phase}_BEGIN at $(_pt_commit "PHASE_${phase}_BEGIN")"
    echo "Moving pointer PHASE_BEGIN_${SUBPROJECT} -> PHASE_${phase}_BEGIN ($(_pt_commit "PHASE_${phase}_BEGIN"))"
    if [ "$at" = "HEAD" ]; then
        echo "NOTE: run this BEFORE the first commit of the phase."
    else
        echo "NOTE: retroactive begin at ${at}."
    fi
}

phase_end() {
    local phase="$1" force="${2:-}"
    if [ -z "$phase" ]; then
        echo "Usage: phase_end <phase_id> [--force]  (e.g. 13_71_ADF)"; return 1
    fi
    if git rev-parse -q --verify "refs/tags/PHASE_${phase}_END" >/dev/null; then
        echo "refusing: PHASE_${phase}_END already exists at $(_pt_commit "PHASE_${phase}_END")"
        return 1
    fi

    local ok=1
    if ! git rev-parse -q --verify "refs/tags/PHASE_${phase}_BEGIN" >/dev/null; then
        echo "REFUSE: PHASE_${phase}_BEGIN missing — cannot close a phase that was never opened."
        ok=0
    elif ! git merge-base --is-ancestor "PHASE_${phase}_BEGIN" HEAD; then
        echo "REFUSE: PHASE_${phase}_BEGIN is not an ancestor of HEAD (diverged history / L-2)."
        ok=0
    else
        local n
        n=$(git rev-list --count "PHASE_${phase}_BEGIN"..HEAD)
        if [ "$n" -eq 0 ]; then
            echo "REFUSE: zero commits since PHASE_${phase}_BEGIN — nothing implemented to close."
            echo "        (Guard that catches closing the wrong / unimplemented phase.)"
            ok=0
        fi
    fi

    # v3 (Sonnet22): out-of-order close — a newer BEGIN inside (this_BEGIN, HEAD]
    # means this range would swallow a later phase's commits.
    if [ "$ok" -eq 1 ]; then
        local bc later t tc
        bc=$(git rev-parse "PHASE_${phase}_BEGIN^{commit}")
        later=""
        for t in $(git tag --list "PHASE_*_${TAGSUFFIX}_BEGIN"); do
            [ "$t" = "PHASE_${phase}_BEGIN" ] && continue
            tc=$(git rev-parse "$t^{commit}")
            [ "$tc" = "$bc" ] && continue
            if git merge-base --is-ancestor "$t" HEAD && git merge-base --is-ancestor "$bc" "$t"; then
                later="${later}${t} ($(git rev-parse --short "$tc"))\n"
            fi
        done
        if [ -n "$later" ]; then
            echo "REFUSE: closing PHASE_${phase} would absorb commits from a LATER phase:"
            printf "        %b" "$later"
            echo "        (out-of-order close — ${phase}_BEGIN..HEAD spans a newer BEGIN.)"
            ok=0
        fi
    fi

    if [ "$ok" -ne 1 ]; then
        if [ "$force" = "--force" ]; then
            echo "OVERRIDE (--force): tagging PHASE_${phase}_END despite the refusal above."
        else
            echo "Aborting. If this is genuinely intended, re-run:  phase_end ${phase} --force"
            return 1
        fi
    fi

    git tag -a "PHASE_${phase}_END" -m "Phase ${phase//_/.} approved"
    echo "Created PHASE_${phase}_END at $(_pt_commit "PHASE_${phase}_END")"
    echo "Phase diff:  git log --oneline PHASE_${phase}_BEGIN..PHASE_${phase}_END"
}

phase_label() {
    if git rev-parse -q --verify "refs/tags/PHASE_BEGIN_${SUBPROJECT}" >/dev/null; then
        local ac tags
        ac=$(_pt_commit "PHASE_BEGIN_${SUBPROJECT}")
        # exact tags on that commit (no `git describe` ambiguity)
        tags=$(git tag --points-at "PHASE_BEGIN_${SUBPROJECT}^{commit}" \
               --list "PHASE_*_${TAGSUFFIX}_BEGIN" 2>/dev/null | tr '\n' ' ')
        echo "Anchor PHASE_BEGIN_${SUBPROJECT} -> ${ac} (${tags:-<no ADF begin tag on this commit>})"
    fi
    local latest
    latest=$(git tag --list "PHASE_*_${TAGSUFFIX}_*BEGIN" --sort=-version:refname 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo "Newest ${TAGSUFFIX} begin: $(echo "$latest" | sed 's/PHASE_//; s/_BEGIN//; s/_/./g')"
    else
        echo "Phase unknown (no ${TAGSUFFIX} begin tags)"
    fi
}

echo "Phase tag functions loaded for $SUBPROJECT (v3, hardened)"
echo "  phase_begin <id> [commit-ish]  — start phase (run BEFORE first commit)"
echo "  phase_end <id> [--force]       — close phase (refuses on missing/diverged/empty/out-of-order)"
echo "  phase_label                    — show current phase anchor"
