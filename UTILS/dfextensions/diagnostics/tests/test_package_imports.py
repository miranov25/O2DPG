"""PHASE_13_74_ADF Increment 1 - package and import infrastructure (D0 / S-20).

Acceptance tests T-P1..T-P6 plus AC-COMPAT, P2-A and P2-B.

Every test here runs the thing it claims to test in a SEPARATE INTERPRETER with
an explicit working directory and an explicit PYTHONPATH.  Testing import
behaviour inside the already-imported pytest process would prove nothing: the
modules are on sys.path by then, which is exactly the condition under test.

FM#12: each test fails against the pre-increment tree.
  - T-P1/T-P2 fail: no __init__.py existed, so the canonical import did not resolve.
  - T-P3 fails: six production sys.path mutations existed.
  - T-P4 fails: no interpreter floor was enforced anywhere.
  - T-P5 fails: importing the package pulled in the render layer.
  - P2-B fails: plotted_columns() did not exist; the plotted set came from a
    second, independently written filter.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent            # .../diagnostics/tests
DIAG = HERE.parent                                # .../diagnostics
PKG_PARENT = DIAG.parent.parent                   # parent of dfextensions

PRODUCTION_MODULES = [
    "report_diagnostics.py", "run_metrics.py", "explain_bundle.py",
    "dfx_run_with_diagnostics.py", "schema.py", "collector.py",
    "audit.py", "job_host_analysis.py", "conclusion_model.py",
]


def _run(code, cwd, extra_env=None):
    """Execute `code` in a fresh interpreter from `cwd` with PYTHONPATH set."""
    env = dict(os.environ)
    # PREPEND, never replace: on a production host PYTHONPATH already carries
    # the analysis stack (pandas, numpy, ROOT).  Overwriting it strips those
    # from the child interpreter and every subprocess test fails for a reason
    # that has nothing to do with what is under test.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (str(PKG_PARENT) + os.pathsep + existing) if existing \
        else str(PKG_PARENT)
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, "-c", code], cwd=str(cwd),
                          capture_output=True, text=True, env=env)


# --------------------------------------------------------------------------
# T-P1 - canonical package imports resolve from the reviewed checkout
# --------------------------------------------------------------------------
def test_TP1_canonical_imports_resolve(tmp_path):
    r = _run(
        "from dfextensions.diagnostics import RunMetrics\n"
        "from dfextensions.diagnostics import report_diagnostics\n"
        "print(RunMetrics.__module__)\n"
        "print(report_diagnostics.__name__)\n",
        cwd=tmp_path)
    assert r.returncode == 0, f"canonical import failed:\n{r.stderr}"
    assert "dfextensions.diagnostics.run_metrics" in r.stdout
    assert "dfextensions.diagnostics.report_diagnostics" in r.stdout


# --------------------------------------------------------------------------
# T-P2 - imports work from a working directory outside diagnostics/
# --------------------------------------------------------------------------
def test_TP2_imports_from_external_working_directory(tmp_path):
    r = _run(
        "import os\n"
        "from dfextensions.diagnostics import RunMetrics\n"
        "print('CWD', os.getcwd())\n",
        cwd=tmp_path)
    assert r.returncode == 0, f"import from external cwd failed:\n{r.stderr}"
    assert str(tmp_path) in r.stdout
    assert str(DIAG) not in r.stdout


# --------------------------------------------------------------------------
# T-P3 - production code contains no sys.path mutation
# --------------------------------------------------------------------------
def test_TP3_no_production_syspath_mutation():
    """The single controlled optional-dependency fallback is the ONLY place a
    path may be added, and it must restore sys.path afterwards."""
    offenders = []
    for name in PRODUCTION_MODULES:
        p = DIAG / name
        if not p.is_file():
            continue
        for n, line in enumerate(p.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            if "sys.path.insert" in s or "sys.path.append" in s:
                # permitted only inside _optional_dependency_path()
                if name == "report_diagnostics.py" and s == "sys.path.insert(0, s)":
                    continue
                offenders.append(f"{name}:{n}: {s}")
    assert not offenders, "production sys.path mutation:\n" + "\n".join(offenders)


def test_TP3_controlled_fallback_restores_syspath():
    """The one permitted fallback must leave sys.path exactly as it found it."""
    r = _run(
        "import sys\n"
        "from dfextensions.diagnostics import report_diagnostics as rd\n"
        "before = list(sys.path)\n"
        "with rd._optional_dependency_path(*rd._ADF_CANDIDATES):\n"
        "    pass\n"
        "print('RESTORED', sys.path == before)\n",
        cwd=DIAG.parent)
    assert r.returncode == 0, r.stderr
    assert "RESTORED True" in r.stdout, f"sys.path not restored:\n{r.stdout}{r.stderr}"


# --------------------------------------------------------------------------
# T-P4 - supported interpreter floor is enforced with a clear message
# --------------------------------------------------------------------------
def test_TP4_python_floor_declared():
    r = _run("import dfextensions.diagnostics as d; print('FLOOR', d.MIN_PYTHON)",
             cwd=DIAG.parent)
    assert r.returncode == 0, r.stderr
    assert "FLOOR (3, 10)" in r.stdout


def test_TP4_python_floor_rejects_older_interpreter(tmp_path):
    """Simulate an older interpreter: the failure must be a named, actionable
    error naming the requirement - not an obscure downstream SyntaxError."""
    r = _run(
        "import sys\n"
        "sys.version_info = (3, 9, 0)\n"
        "try:\n"
        "    import dfextensions.diagnostics\n"
        "    print('NO_ERROR')\n"
        "except RuntimeError as e:\n"
        "    print('RAISED', 'requires Python' in str(e), '3.10' in str(e))\n",
        cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "RAISED True True" in r.stdout, f"floor not enforced clearly:\n{r.stdout}"


# --------------------------------------------------------------------------
# T-P5 - collection core imports without the optional analysis stack
# --------------------------------------------------------------------------
def test_TP5_core_imports_without_adf_or_dfdraw(tmp_path):
    r = _run(
        "import sys\n"
        "class _Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name.split('.')[0] in ('AliasDataFrame',): return self\n"
        "    def load_module(self, name):\n"
        "        raise ImportError('blocked for test: ' + name)\n"
        "sys.meta_path.insert(0, _Block())\n"
        "from dfextensions.diagnostics import RunMetrics\n"
        "from dfextensions.diagnostics import schema, collector\n"
        "print('CORE_OK')\n",
        cwd=tmp_path)
    assert r.returncode == 0, f"core import needs the optional stack:\n{r.stderr}"
    assert "CORE_OK" in r.stdout


def test_TP5_package_import_does_not_pull_in_render_layer(tmp_path):
    """Importing the package must not import the render layer, or a
    collection-only host cannot use the package at all."""
    r = _run(
        "import sys\n"
        "import dfextensions.diagnostics\n"
        "print('RENDER_IMPORTED',\n"
        "      'dfextensions.diagnostics.report_diagnostics' in sys.modules)\n",
        cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "RENDER_IMPORTED False" in r.stdout


# --------------------------------------------------------------------------
# T-P6 - clear dependency failure when the render layer needs what is absent
# --------------------------------------------------------------------------
def test_TP6_missing_optional_dependency_reports_clearly(tmp_path):
    r = _run(
        "import sys\n"
        "class _Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name.split('.')[0] == 'AliasDataFrame': return self\n"
        "    def load_module(self, name):\n"
        "        raise ImportError('blocked for test: ' + name)\n"
        "sys.meta_path.insert(0, _Block())\n"
        "from dfextensions.diagnostics import report_diagnostics as rd\n"
        "import pandas as pd\n"
        "try:\n"
        "    _adf, _reg = rd._build_adf(pd.DataFrame({'t_rel': [0.0, 1.0]}))\n"
        "    print('NO_ERROR')\n"
        "except ImportError as e:\n"
        "    print('IMPORTERROR', 'AliasDataFrame' in str(e))\n"
        "except Exception as e:\n"
        "    print('OTHER', type(e).__name__)\n",
        cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "IMPORTERROR True" in r.stdout, (
        "a missing optional dependency must fail as a named ImportError "
        f"identifying the component; got:\n{r.stdout}")


# --------------------------------------------------------------------------
# AC-COMPAT - the architect's direct script invocation is preserved
# --------------------------------------------------------------------------
def test_ACCOMPAT_direct_script_invocation_still_works():
    """`python report_diagnostics.py <bundle> -o <out>` must keep working from
    inside the diagnostics directory, with NO PYTHONPATH set at all - this is
    the invocation in daily use for the GSI investigation."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, "report_diagnostics.py", "--help"],
                       cwd=str(DIAG), capture_output=True, text=True, env=env)
    assert r.returncode == 0, (
        "direct script invocation broke - this is the daily-use command:\n"
        f"{r.stderr}")
    assert "bundles" in r.stdout and "--out-dir" in r.stdout


def test_ACCOMPAT_script_mode_resolves_sibling_modules():
    """Script mode must resolve schema/audit/job_host_analysis/conclusion_model
    without any path insertion, purely through Python's own script-directory
    rule."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    code = ("import report_diagnostics as r\n"
            "print('SIBLINGS', all(hasattr(r, n) for n in "
            "('schema', 'audit_mod', 'jha_mod', 'cm_mod')))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(DIAG),
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "SIBLINGS True" in r.stdout


# --------------------------------------------------------------------------
# P2-B - classify_availability() is the single owner of the plotted set
# --------------------------------------------------------------------------
def _rd():
    sys.path.insert(0, str(DIAG.parent.parent))
    from dfextensions.diagnostics import report_diagnostics as rd
    return rd


def test_P2B_measured_zero_is_plotted_never_sampled_is_not():
    pd = pytest.importorskip("pandas")
    rd = _rd()
    wide = pd.DataFrame({"cpu_target_job": [0.0, 0.0, 0.0],
                         "cpu_current_user_non_job": [1.0, 2.0, 1.5]})
    states = rd.classify_availability(wide)
    assert states["target_job"] == "zero"
    assert states["current_user_non_job"] == "active"
    assert states["other_visible_workloads"] == "no_data"

    plotted = rd.plotted_columns(states)
    # measured idleness is a real result and MUST be drawn
    assert "cpu_target_job" in plotted
    assert "cpu_current_user_non_job" in plotted
    # never-sampled MUST NOT be drawn: drawing it asserts a zero nobody measured
    assert "cpu_other_visible_workloads" not in plotted


def test_P2B_single_owner_cannot_drift_from_the_caption():
    """The plotted set and the availability caption must be derived from ONE
    classification.  This is the actual content of P2-B: not new behaviour, but
    the removal of a second, independently written filter that could disagree
    with the caption beside the figure."""
    pd = pytest.importorskip("pandas")
    rd = _rd()
    for frame in (
        pd.DataFrame({"cpu_target_job": [0.0], "cpu_current_user_non_job": [1.0]}),
        pd.DataFrame({"cpu_target_job": [float("nan")],
                      "cpu_other_visible_workloads": [3.0]}),
        pd.DataFrame({"cpu_current_user_non_job": [0.0, 0.0]}),
    ):
        states = rd.classify_availability(frame)
        plotted = rd.plotted_columns(states)
        for scope, state in states.items():
            col = f"cpu_{scope}"
            assert (col in plotted) == (state != "no_data"), (
                f"plotted set disagrees with classification for {scope}: "
                f"state={state} plotted={col in plotted}")


def test_P2B_no_data_note_states_absence_is_not_idleness():
    rd = _rd()
    note = rd._availability_note({"target_job": "no_data",
                                  "current_user_non_job": "active",
                                  "other_visible_workloads": "active"})
    low = note.lower()
    assert "no data" in low
    assert "idle" in low, "the note must say absence is not an idle job"


# --------------------------------------------------------------------------
# P2-A - production-path regression, full render tier
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not (DIAG.parent / "AliasDataFrame").is_dir(),
    reason="full-render tier: AliasDataFrame not available on this host")
def test_P2A_never_sampled_series_absent_from_rendered_report(tmp_path):
    """End-to-end: render a bundle in which the target job was never sampled.
    The rendered HTML must (a) not draw the unsampled series, and (b) say in
    words that absence of a job line means no data, not an idle job."""
    pd = pytest.importorskip("pandas")
    rd = _rd()
    wide = pd.DataFrame({"t_rel": [0.0, 1.0, 2.0],
                         "cpu_current_user_non_job": [1.0, 2.0, 1.0]})
    states = rd.classify_availability(wide)
    assert states["target_job"] == "no_data"
    plotted = rd.plotted_columns(states)
    assert "cpu_target_job" not in plotted

    adf, _registered = rd._build_adf(wide)      # returns (adf, alias names)
    figs = rd._draw_expr(adf, "[" + ",".join(plotted) + "]",
                         "P2-A regression", "cores", tmp_path / "figures",
                         "p2a_background_vs_job.png")
    assert figs, "the render path produced no figure"
    note = rd._availability_note(states)
    assert "no data" in note.lower() and "idle" in note.lower()


# --------------------------------------------------------------------------
# LEDGER-PKT - the completion ledger rides in every reviewer packet
# --------------------------------------------------------------------------
def test_LEDGERPKT_builder_declares_the_ledger():
    src = (DIAG / "reviewer_bundle.py").read_text()
    assert "LEDGER_NAMES" in src
    assert "Completion_Ledger" in src


def test_LEDGERPKT_identity_gate_removed_and_substitution_retained_unused():
    """[Decision 3, 2026-07-23] information propagates in full: the build-time
    identity gate is gone.  The substitution helpers stay defined but must not
    be called, so re-enabling them later is a switch rather than new work."""
    src = (DIAG / "reviewer_bundle.py").read_text()
    assert "def _scrub(" in src and "def _leak_scan(" in src, \
        "substitution helpers must be retained for the deferred option"
    live = [n for n, line in enumerate(src.splitlines(), 1)
            if ("_scrub(" in line or "_leak_scan(" in line)
            and not line.strip().startswith(("#", "def ", '"""'))
            and "remain defined" not in line]
    assert not live, f"substitution must not be invoked; live calls at {live}"
