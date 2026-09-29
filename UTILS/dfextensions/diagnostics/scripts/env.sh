# shellcheck shell=bash
#
# PHASE_13_74_ADF diagnostics generic environment.
# Source this file; do not execute it.
#
# Canonical project variables:
#   DFEXT_CODE              shared dfextensions root
#   DIAGNOSTICS_CODE        diagnostics source/code root
#   DIAGNOSTICS_DOCS        PHASE/CRR/review-document root
#   DIAGNOSTICS_DATA        generated monitoring data / large evidence
#   DIAGNOSTICS_FILES       temporary reviewer-distribution workspace
#   DIAGNOSTICS_INSTALL_BY_MD5
#
# The caller may override any variable before sourcing this file.

if [ -z "${O2DPG:-}" ]; then
    echo "diagnostics env: \$O2DPG is not set" >&2
    return 1
fi

if [ -z "${NOTES:-}" ]; then
    echo "diagnostics env: \$NOTES is not set" >&2
    return 1
fi

if [ -z "${NOTESData:-}" ]; then
    echo "diagnostics env: \$NOTESData is not set" >&2
    return 1
fi

export DFEXT_CODE="${DFEXT_CODE:-$O2DPG/UTILS/dfextensions}"
export DIAGNOSTICS_CODE="${DIAGNOSTICS_CODE:-$DFEXT_CODE/diagnostics}"
export DIAGNOSTICS_DOCS="${DIAGNOSTICS_DOCS:-$NOTES/JIRA/O2-6532/docs/diagnostics}"
export DIAGNOSTICS_DATA="${DIAGNOSTICS_DATA:-$NOTESData/JIRA/O2-6532/diagnostics}"
export DIAGNOSTICS_FILES="${DIAGNOSTICS_FILES:-$DIAGNOSTICS_CODE/files_diagnostics}"
export DIAGNOSTICS_INSTALL_BY_MD5="${DIAGNOSTICS_INSTALL_BY_MD5:-$DFEXT_CODE/scripts/install_by_md5.sh}"

# Make the dfextensions package importable from any working directory.
# Re-sourcing this file must not duplicate the entry.
_DIAGNOSTICS_PYROOT="$O2DPG/UTILS"
case ":${PYTHONPATH:-}:" in
    *":$_DIAGNOSTICS_PYROOT:"*)
        ;;
    *)
        export PYTHONPATH="$_DIAGNOSTICS_PYROOT${PYTHONPATH:+:$PYTHONPATH}"
        ;;
esac
unset _DIAGNOSTICS_PYROOT

# Respect an explicitly configured Downloads directory.
if [ -z "${Downloads:-}" ] && [ -d "$HOME/Downloads" ]; then
    Downloads="$HOME/Downloads"
    export Downloads
fi

if [ ! -d "$DIAGNOSTICS_CODE" ]; then
    echo "diagnostics env: missing diagnostics source tree: $DIAGNOSTICS_CODE" >&2
    return 1
fi

if [ ! -f "$DIAGNOSTICS_INSTALL_BY_MD5" ]; then
    echo "diagnostics env: missing shared installer: $DIAGNOSTICS_INSTALL_BY_MD5" >&2
    return 1
fi
