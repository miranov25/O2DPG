#!/bin/bash
# =============================================================================
# dfextensions/scripts/phase_tag_common.sh — Phase Boundary Management (v6.1-generic)\n# Localization (SUBPROJECT/TAGSUFFIX) is injected by per-subproject wrappers.
# Ref: BUG_AliasDataFrame_20260706_phase_tagging
# =============================================================================
# Authoritative moving pointer: PHASE_BEGIN_${SUBPROJECT} (SOLE source of truth
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

# --- GENERIC: localization comes from the per-subproject wrapper ------------
# This file is dfextensions/scripts/phase_tag_common.sh (single source of truth,
# shared by ADF / dfdraw / GB / RDF). It REFUSES to load standalone: SUBPROJECT
# and TAGSUFFIX must be set by the 3-line wrapper scripts/phase_tag.sh in each
# subproject. Deliberately NO auto-detection from $PWD -- silent wrong-project
# localization is the root cause of BUG_AliasDataFrame_20260706_phase_tagging.
if [ -z "$SUBPROJECT" ] || [ -z "$TAGSUFFIX" ]; then
    echo "REFUSE: phase_tag_common.sh must be sourced via a subproject wrapper"
    echo "        that sets SUBPROJECT and TAGSUFFIX (see <subproject>/scripts/phase_tag.sh)."
    return 1 2>/dev/null || exit 1
fi

# short COMMIT hash a tag points at (deref annotated tag object -> commit)
_pt_commit() { git rev-parse --short "$1^{commit}" 2>/dev/null; }

phase_begin() {
    if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
        echo "Usage: phase_begin <phase_id> [commit-ish]   (id EXACT, e.g. 13_71_${TAGSUFFIX}; run BEFORE first commit)"; return 0
    fi
    local phase="${1:-}" at="${2:-HEAD}"
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
    if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
        echo "Usage: phase_end <phase_id> [--force]   (id EXACT, e.g. 13_71_${TAGSUFFIX}; refuses on missing/diverged/empty/out-of-order)"; return 0
    fi
    local phase="${1:-}" force="${2:-}"
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
    if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then echo "Usage: phase_label   (no args; prints the current phase anchor)"; return 0; fi
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

# -----------------------------------------------------------------------------
# _pt_resolve <input> — normalize a user-typed phase id to the canonical form and
# confirm it has a BEGIN tag. Accepts dots (13.69.ADF), and a bare prefix (13_69)
# that maps unambiguously to exactly one PHASE_<...>_BEGIN. Echoes the canonical id
# on success; empty on failure (0 or >1 matches).
_pt_resolve() {
    local x="${1//./_}"
    if git rev-parse -q --verify "refs/tags/PHASE_${x}_BEGIN" >/dev/null; then
        echo "$x"; return 0
    fi
    local matches count
    matches=$(git tag --list "PHASE_${x}_*_BEGIN" 2>/dev/null | sed -E 's/^PHASE_//; s/_BEGIN$//' | sort -u)
    count=$(printf '%s' "$matches" | grep -c .)
    if [ "$count" = "1" ]; then echo "$matches"; return 0; fi
    if [ "$count" -gt "1" ]; then
        echo "AMBIGUOUS" >&2; printf '%s\n' "$matches" >&2
    fi
    return 1
}

# _phase_id_help — the "glue": expected form + example + the available phase ids.
_phase_id_help() {
    echo "  Expected phase id:  <major>_<minor>_${TAGSUFFIX}     e.g.  13_69_${TAGSUFFIX}"
    echo "  (underscores, not dots; the _${TAGSUFFIX} suffix is required — '13_69' and"
    echo "   '13.69.${TAGSUFFIX}' are auto-resolved when they match exactly one phase.)"
    echo "  Available ${TAGSUFFIX} phases with a BEGIN tag (newest first):"
    git tag --list "PHASE_*_${TAGSUFFIX}_BEGIN" --sort=-version:refname 2>/dev/null \
        | sed -E 's/^PHASE_//; s/_BEGIN$//' | sed 's/^/    /' | head -20
}

# phase_help — standalone help/example for the whole toolset.
phase_help() {
    echo "phase_tag.sh — usage & expected phase-id form"
    echo "  phase_begin <id> [commit-ish]   e.g. phase_begin 13_71_${TAGSUFFIX}   (id is EXACT, canonical)"
    echo "  phase_end   <id> [--force]      e.g. phase_end   13_71_${TAGSUFFIX}"
    echo "  phase_label                     current anchor"
    echo "  phase_info  <id> [--commits --dates --lines --files --all]   (id may be 13_71 / 13.71.${TAGSUFFIX})"
    echo
    _phase_id_help
}

# _phase_info_help — full help for phase_info WITH a worked example.
_phase_info_help() {
    echo "phase_info <id> [--commits --dates --lines --files --all]   (read-only)"
    echo "  id may be 13_69_${TAGSUFFIX}, 13_69, or 13.69.${TAGSUFFIX} (auto-resolved when unambiguous)"
    echo "  flags (combine freely; with none, only the summary prints):"
    echo "    --commits   related commits: hash, date, subject"
    echo "    --dates     per-commit date + author"
    echo "    --lines     lines changed per commit (+ phase total)"
    echo "    --files     files changed across the phase"
    echo "    --all       all of the above"
    echo
    echo "  Example:"
    echo "    \$ phase_info 13_69_${TAGSUFFIX} --commits --lines"
    echo "    Phase 13.69.${TAGSUFFIX}  [CLOSED]"
    echo "      BEGIN PHASE_13_69_${TAGSUFFIX}_BEGIN -> 3bb0811  2026-07-05"
    echo "      END   PHASE_13_69_${TAGSUFFIX}_END   -> 9186387  2026-07-06"
    echo "      commits: 7     lines: +512 -18"
    echo "      --- related commits ---"
    echo "        9186387  2026-07-06  ADF: draw tests use the lazy switch"
    echo "        553ada8   2026-07-06  ADF: external persistence (D5b) + G-1 expand"
    echo "      --- lines per commit ---"
    echo "        9186387  ADF: draw tests ...        2 files changed, 6 insertions(+), 2 deletions(-)"
    echo
    _phase_id_help
}

# -----------------------------------------------------------------------------
# phase_info <id> [flags] — READ-ONLY phase inspection (never mutates tags).
#   default (no flags): one-line-per-field summary (BEGIN/END commit+date, #commits, +/- lines)
#   --commits   list related commits (hash, date, subject)
#   --dates     BEGIN/END + per-commit dates
#   --lines     lines changed per commit (+ phase total)
#   --files     files changed across the phase
#   --all       everything above
# id may be given as 13_69_ADF, 13_69, or 13.69.ADF (auto-resolved when unambiguous).
# Range uses merge-base(BEGIN, END|HEAD) so a diverged BEGIN (L-2) still yields a
# sane phase-only range. If END is absent the phase is reported OPEN (range to HEAD).
# -----------------------------------------------------------------------------
phase_info() {
    # Order-independent arg parse: the id and the flags may appear in ANY order.
    local raw="" c_commits=0 c_dates=0 c_lines=0 c_files=0 a
    for a in "$@"; do
        case "$a" in
            -h|--help) _phase_info_help; return 0 ;;
            --commits) c_commits=1 ;;
            --dates)   c_dates=1 ;;
            --lines)   c_lines=1 ;;
            --files)   c_files=1 ;;
            --all)     c_commits=1; c_dates=1; c_lines=1; c_files=1 ;;
            --*|-*)    echo "phase_info: unknown flag '$a'"; _phase_info_help; return 1 ;;
            *)         if [ -z "$raw" ]; then raw="$a"
                       else echo "phase_info: give exactly one phase id (got '$raw' and '$a')"; return 1; fi ;;
        esac
    done
    if [ -z "$raw" ]; then _phase_info_help; return 0; fi
    local phase; phase=$(_pt_resolve "$raw")
    if [ -z "$phase" ]; then
        echo "phase_info: could not resolve '${raw}' to a phase with a BEGIN tag."
        _phase_id_help; return 1
    fi
    local beg="PHASE_${phase}_BEGIN" end state
    if git rev-parse -q --verify "refs/tags/PHASE_${phase}_END" >/dev/null; then
        end="PHASE_${phase}_END"; state="CLOSED"
    else
        end="HEAD"; state="OPEN (no END tag; range to HEAD)"
    fi
    local base; base=$(git merge-base "$beg" "$end")
    local range="${base}..${end}"

    # v6.1 CONCURRENT-PHASE FIX: when several phases are OPEN on one branch,
    # the bare BEGIN..END range mixes their commits and both phases report
    # wrong numbers. If any commit in range carries this phase's message
    # prefix (unbracketed convention 'PHASE_<id>:', legacy '[PHASE_<id>]'
    # tolerated), statistics are computed from the prefix-matched commits
    # only. Zero matches (pre-convention phases) -> unchanged range behavior.
    local pgrep="^\\[?PHASE_${phase}\\]?[: ]"
    local n_all n_match
    n_all=$(git rev-list --count "$range")
    n_match=$(git log -E --grep="$pgrep" --format=%h "$range" | wc -l | tr -d ' ')
    local filt=""
    if [ "$n_match" -gt 0 ]; then filt=1; fi

    local n add del
    if [ -n "$filt" ]; then
        n=$n_match
        add=$(git log -E --grep="$pgrep" --numstat --format= "$range" | awk '{a+=$1} END{print a+0}')
        del=$(git log -E --grep="$pgrep" --numstat --format= "$range" | awk '{d+=$2} END{print d+0}')
    else
        n=$n_all
        add=$(git diff --numstat "$base" "$end" | awk '{a+=$1} END{print a+0}')
        del=$(git diff --numstat "$base" "$end" | awk '{d+=$2} END{print d+0}')
    fi
    echo "Phase ${phase//_/.}  [$state]"
    echo "  BEGIN $beg -> $(git rev-parse --short "$beg^{commit}")  $(git log -1 --date=short --format=%ad "$beg^{commit}")"
    if [ "$end" != "HEAD" ]; then
        echo "  END   $end -> $(git rev-parse --short "$end^{commit}")  $(git log -1 --date=short --format=%ad "$end^{commit}")"
    fi
    echo "  commits: $n     lines: +${add} -${del}"
    if [ -n "$filt" ] && [ "$n_match" -lt "$n_all" ]; then
        echo "  (prefix-filtered: $n_match of $n_all commits in range belong to this phase; concurrent phases share this branch)"
    fi

    if [ "$c_commits" = 1 ]; then
        echo "  --- related commits ---"
        if [ -n "$filt" ]; then git log -E --grep="$pgrep" --date=short --format='    %h  %ad  %s' "$range"
        else git log --date=short --format='    %h  %ad  %s' "$range"; fi
    fi
    if [ "$c_dates" = 1 ]; then
        echo "  --- commit dates ---"
        if [ -n "$filt" ]; then git log -E --grep="$pgrep" --format='    %h  %ci  %an' "$range"
        else git log --format='    %h  %ci  %an' "$range"; fi
    fi
    if [ "$c_lines" = 1 ]; then
        echo "  --- lines per commit ---"
        local h subj st
        { if [ -n "$filt" ]; then git log -E --grep="$pgrep" --format='%h%x09%s' "$range"
          else git log --format='%h%x09%s' "$range"; fi; } | while IFS=$'\t' read -r h subj; do
            st=$(git show --shortstat --format= "$h" | tr -s ' ' | sed '/^$/d' | tail -1 | sed 's/^ *//')
            printf '    %s  %-44.44s  %s\n' "$h" "$subj" "${st:-no file changes}"
        done
    fi
    if [ "$c_files" = 1 ]; then
        echo "  --- files changed ---"
        if [ -n "$filt" ]; then
            git log -E --grep="$pgrep" --numstat --format= "$range" \
              | awk '{a[$3]+=$1; d[$3]+=$2} END{t_a=0;t_d=0;for(f in a){printf "    %-55s | +%d -%d\n", f, a[f], d[f]; t_a+=a[f]; t_d+=d[f]} printf "    %d files changed, %d insertions(+), %d deletions(-)\n", length(a), t_a, t_d}' \
              | sort
        else
            git diff --stat "$base" "$end" | sed 's/^/    /'
        fi
    fi
}

# =============================================================================
# Dual mode:
#   - SOURCED (source scripts/phase_tag.sh): loads the functions, one quiet line
#     (set PHASE_TAG_QUIET=1 to silence entirely).
#   - EXECUTED (bash scripts/phase_tag.sh <cmd> ...): runs one command and prints
#     the result — no banner, exit code preserved. This is the reviewer-runnable,
#     loggable form: a reviewer names one command, you run it, the output is the log.
#
# Reviewer-loggable examples (run and paste the output):
#   bash scripts/phase_tag.sh info  13_69_ADF --all
#   bash scripts/phase_tag.sh report 13_69_ADF          # stamped full report (date + HEAD)
#   bash scripts/phase_tag.sh label
# =============================================================================
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    _cmd="${1:-help}"; shift 2>/dev/null
    case "$_cmd" in
        begin)  phase_begin "$@" ;;
        end)    phase_end   "$@" ;;
        label)  phase_label "$@" ;;
        info)   phase_info  "$@" ;;
        help|-h|--help) phase_help ;;
        report)
            # self-identifying transcript for review logs (matches the gate-transcript precedent)
            echo "# phase_tag.sh report | $(date -u +%Y-%m-%dT%H:%M:%SZ) | repo HEAD $(git rev-parse --short HEAD 2>/dev/null)"
            phase_info "$@" --all ;;
        *) echo "unknown command: $_cmd"; echo; phase_help; exit 2 ;;
    esac
    exit $?
else
    [ -z "${PHASE_TAG_QUIET:-}" ] && echo "phase_tag_common.sh loaded ($SUBPROJECT/$TAGSUFFIX, v6.1-generic) — run 'phase_help' for usage."
fi
