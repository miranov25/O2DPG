#!/bin/bash
# <RDF>/scripts/phase_tag.sh — wrapper; all logic in dfextensions/scripts/phase_tag_common.sh
SUBPROJECT="RDF"; TAGSUFFIX="RDF"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/phase_tag_common.sh"
