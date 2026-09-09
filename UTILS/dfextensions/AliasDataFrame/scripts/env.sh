#!/usr/bin/env bash
# AliasDataFrame environment bootstrap.
# Commit this file. Machine/user-specific values belong in env.local.sh.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${_SCRIPT_DIR}/env.local.sh" ]; then
    # shellcheck source=/dev/null
    source "${_SCRIPT_DIR}/env.local.sh"
fi

echo "AliasDataFrame environment"
echo "  ADF_CODE = ${ADF_CODE:-<unset>}"
echo "  ADF_DOCS = ${ADF_DOCS:-<unset>}"
