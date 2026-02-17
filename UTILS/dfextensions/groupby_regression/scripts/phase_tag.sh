#!/bin/bash
# scripts/phase_tag.sh — Phase tagging utilities for GroupBy regression
# Usage: source scripts/phase_tag.sh

SUBPROJECT="GroupByRegression"

phase_begin() {
    # Create permanent BEGIN tag + update moving tag
    # Usage: phase_begin 13_10_GB
    local phase="$1"
    if [ -z "$phase" ]; then
        echo "Usage: phase_begin <phase_id>  (e.g. 13_10_GB)"
        return 1
    fi
    git tag -a "PHASE_${phase}_BEGIN" -m "Phase ${phase//_/.} begin"
    git tag -f "PHASE_BEGIN_${SUBPROJECT}" "PHASE_${phase}_BEGIN"
    echo "Created PHASE_${phase}_BEGIN, updated PHASE_BEGIN_${SUBPROJECT}"
}

phase_end() {
    # Create permanent END tag at approval
    # Usage: phase_end 13_10_GB
    local phase="$1"
    if [ -z "$phase" ]; then
        echo "Usage: phase_end <phase_id>  (e.g. 13_10_GB)"
        return 1
    fi
    git tag -a "PHASE_${phase}_END" -m "Phase ${phase//_/.} approved"
    echo "Created PHASE_${phase}_END"
}

phase_current() {
    # Get current phase from moving tag
    local target=$(git describe --tags --match "PHASE_*_BEGIN" --abbrev=0 \
                    $(git rev-list -1 "PHASE_BEGIN_${SUBPROJECT}" 2>/dev/null) 2>/dev/null)
    if [ -z "$target" ]; then
        echo "unknown"
    else
        echo "$target" | sed 's/PHASE_//; s/_BEGIN//; s/_/./g'
    fi
}

phase_label() {
    # Full label for scripts (e.g. "Phase 13.10.GB")
    echo "Phase $(phase_current)"
}
