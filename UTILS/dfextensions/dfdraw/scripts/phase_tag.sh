#!/bin/bash
# =============================================================================
# phase_tag.sh — dfdraw Phase Boundary Management
# =============================================================================
#
# Usage:
#   source scripts/phase_tag.sh
#   phase_begin 13_15_DF      # Creates PHASE_13_15_DF_BEGIN + updates PHASE_BEGIN_dfdraw
#   phase_end 13_15_DF        # Creates PHASE_13_15_DF_END
#   phase_label               # Prints current phase label
#
# Adapted from GroupByRegression/scripts/phase_tag.sh

SUBPROJECT="${SUBPROJECT:-dfdraw}"

phase_begin() {
    local phase="$1"
    if [ -z "$phase" ]; then
        echo "Usage: phase_begin <phase_id>  (e.g. 13_15_DF)"
        return 1
    fi
    git tag -a "PHASE_${phase}_BEGIN" -m "Phase ${phase//_/.} begin"
    git tag -f "PHASE_BEGIN_${SUBPROJECT}" "PHASE_${phase}_BEGIN"
    echo "Created PHASE_${phase}_BEGIN, updated PHASE_BEGIN_${SUBPROJECT}"
}

phase_end() {
    local phase="$1"
    if [ -z "$phase" ]; then
        echo "Usage: phase_end <phase_id>  (e.g. 13_15_DF)"
        return 1
    fi
    git tag -a "PHASE_${phase}_END" -m "Phase ${phase//_/.} approved"
    echo "Created PHASE_${phase}_END"
}

phase_label() {
    local latest
    latest=$(git tag --list 'PHASE_*_DF_*_BEGIN' --sort=-version:refname 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        local phase_id
        phase_id=$(echo "$latest" | sed 's/PHASE_//; s/_BEGIN//; s/_/./g')
        echo "Phase $phase_id"
    else
        echo "Phase unknown"
    fi
}

echo "Phase tag functions loaded for $SUBPROJECT"
echo "  phase_begin <id>  — start new phase"
echo "  phase_end <id>    — close phase"
echo "  phase_label       — show current phase"
