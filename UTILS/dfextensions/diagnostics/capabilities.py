"""dfextensions.diagnostics - committed capability registry [PHASE_13_74_ADF, D0].

This module is the SINGLE committed source of truth for what the diagnostics
subproject can do.  The human-readable Capability Matrix (docs/CAPABILITY_MATRIX.md)
is GENERATED from it by generate_matrix() and is committed alongside it
[architect Decision 1, 2026-07-23: commit the generated matrix so capability is
tracked over time - this supersedes v8 A-2, which said generate-only].

Acceptance test T-P9 regenerates the matrix from this registry and fails if the
committed matrix has drifted, so the committed copy can never go silently stale.

Each capability lists the tests that actually PROVE it.  Every referenced path
must exist in the suite - a mapping-integrity test enforces this, so the registry
can never claim a capability is covered by a test that is not there.  The
`planned_oracles` field records the named oracle tests the v8 proposal (§13.2)
intends but which are not yet written; they are declared, never counted as if
present.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Count lock [v8 §13.2: 0 -> 3].  A test asserts len(CAPABILITIES) == COUNT_LOCK,
# so adding or dropping a capability without updating the lock fails the gate.
COUNT_LOCK = 3

# The complete set of named oracle tests from v8 §13.2. These are the tests the
# proposal intends; several are not yet written and live in `planned_oracles`.
# A test asserts the union of every capability's planned_oracles equals this set
# exactly, so the registry can neither silently drop a named oracle (the P1-1
# finding: 4 were missing) nor invent one outside the ratified list.
V8_SECTION_13_2_ORACLES = frozenset({
    "test_host_health_no_external_writes",
    "test_host_health_rate_oracle",
    "test_run_metrics_wrapper_invariance",
    "test_report_summary_matches_pandas_oracle",
    "test_background_influence_oracle",
    "test_process_top_union_oracle",
    "test_user_aggregate_oracle",
    "test_workload_rollup_oracle",
    "test_target_job_scope_oracle",
    "test_statistical_audit_replay",
})


@dataclass(frozen=True)
class Capability:
    id: str
    title: str
    question: str
    status: str                         # operational | partial | planned
    proving_tests: List[str]            # real paths, existence-checked
    planned_oracles: List[str] = field(default_factory=list)  # v8 §13.2 intent


CAPABILITIES: List[Capability] = [
    Capability(
        id="DIAGNOSTICS.host_health",
        title="Host health",
        question="Is this machine healthy?",
        status="operational",
        proving_tests=[
            "tests/test_dfx_host_diagnostics.sh",
            "tests/test_collector.py",
            "tests/test_diagnostics_py.py",
        ],
        planned_oracles=[
            "test_host_health_no_external_writes",
            "test_host_health_rate_oracle",
            "test_process_top_union_oracle",
            "test_user_aggregate_oracle",
            "test_workload_rollup_oracle",
        ],
    ),
    Capability(
        id="DIAGNOSTICS.run_metrics",
        title="Run metrics",
        question="Did the machine's load affect my job?",
        status="operational",
        proving_tests=[
            "tests/test_run_metrics.py",
            "tests/test_wrapper_vertical.py",
            "tests/test_orchestration.py",
        ],
        planned_oracles=[
            "test_run_metrics_wrapper_invariance",
            "test_target_job_scope_oracle",
        ],
    ),
    Capability(
        id="DIAGNOSTICS.analytics_report",
        title="Analytics report",
        question="What do the host and job series say together?",
        status="operational",
        proving_tests=[
            "tests/test_job_host_analysis.py",
            "tests/test_conclusion_model.py",
            "tests/test_audit.py",
            "tests/test_integration_contracts.py",
            "tests/test_e2e_presence.py",
        ],
        planned_oracles=[
            "test_report_summary_matches_pandas_oracle",
            "test_background_influence_oracle",
            "test_statistical_audit_replay",
        ],
    ),
]

_HEADER = "# Diagnostics Capability Matrix\n"
_NOTE = (
    "_Generated from `capabilities.py` by `generate_matrix()` "
    "[PHASE_13_74_ADF, D0]. Do not edit by hand; edit the registry and "
    "regenerate. Reproducibility is enforced by test T-P9._\n"
)


def generate_matrix() -> str:
    """Render the committed Capability Matrix deterministically.

    Deterministic by construction: fixed registry order, no timestamp, no host
    or path detail.  T-P9 compares this output byte-for-byte against the
    committed docs/CAPABILITY_MATRIX.md, so any non-determinism would break the
    gate rather than ship silently.
    """
    lines = [_HEADER, "", _NOTE, "",
             f"Capabilities: **{len(CAPABILITIES)}** (count lock {COUNT_LOCK}).",
             "",
             "| Capability | Answers | Status | Proving tests | Planned oracles (v8 §13.2) |",
             "|---|---|---|---|---|"]
    for c in CAPABILITIES:
        proving = "<br>".join(f"`{p}`" for p in c.proving_tests)
        planned = "<br>".join(f"`{o}`" for o in c.planned_oracles) or "—"
        lines.append(
            f"| `{c.id}` | {c.question} | {c.status} | {proving} | {planned} |"
        )
    lines.append("")
    return "\n".join(lines)


def referenced_test_paths() -> List[str]:
    """Every real proving-test path referenced by the registry (for the
    mapping-integrity check)."""
    return [p for c in CAPABILITIES for p in c.proving_tests]


if __name__ == "__main__":          # regenerate the committed matrix
    out = Path(__file__).resolve().parent / "docs" / "CAPABILITY_MATRIX.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(generate_matrix())
    print(f"[capabilities] wrote {out} ({len(CAPABILITIES)} capabilities)")
