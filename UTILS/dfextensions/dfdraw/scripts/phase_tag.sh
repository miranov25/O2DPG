#!/bin/bash
# <dfdraw>/scripts/phase_tag.sh — wrapper; all logic in dfextensions/scripts/phase_tag_common.sh
SUBPROJECT="dfdraw"; TAGSUFFIX="DF"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/phase_tag_common.sh"
