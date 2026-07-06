#!/bin/bash
# dfextensions/scripts/phase_tag.sh — ORG wrapper (cross-subproject tooling phases).
# Lives BESIDE phase_tag_common.sh — no Org directory needed.
SUBPROJECT="ORG"; TAGSUFFIX="ORG"
_PT_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/phase_tag_common.sh"
if [ ! -f "$_PT_COMMON" ]; then echo "REFUSE: $_PT_COMMON missing"; return 1 2>/dev/null || exit 1; fi
source "$_PT_COMMON"
