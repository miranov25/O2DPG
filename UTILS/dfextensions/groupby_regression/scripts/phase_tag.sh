#!/bin/bash
# <GBregression>/scripts/phase_tag.sh — wrapper; all logic in dfextensions/scripts/phase_tag_common.sh
SUBPROJECT="groupby_regression"; TAGSUFFIX="GB"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/phase_tag_common.sh"
