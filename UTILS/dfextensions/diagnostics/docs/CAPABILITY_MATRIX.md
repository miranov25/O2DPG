# Diagnostics Capability Matrix


_Generated from `capabilities.py` by `generate_matrix()` [PHASE_13_74_ADF, D0]. Do not edit by hand; edit the registry and regenerate. Reproducibility is enforced by test T-P9._


Capabilities: **3** (count lock 3).

| Capability | Answers | Status | Proving tests | Planned oracles (v8 §13.2) |
|---|---|---|---|---|
| `DIAGNOSTICS.host_health` | Is this machine healthy? | operational | `tests/test_dfx_host_diagnostics.sh`<br>`tests/test_collector.py`<br>`tests/test_diagnostics_py.py` | `test_host_health_no_external_writes`<br>`test_host_health_rate_oracle`<br>`test_process_top_union_oracle`<br>`test_user_aggregate_oracle`<br>`test_workload_rollup_oracle` |
| `DIAGNOSTICS.run_metrics` | Did the machine's load affect my job? | operational | `tests/test_run_metrics.py`<br>`tests/test_wrapper_vertical.py`<br>`tests/test_orchestration.py` | `test_run_metrics_wrapper_invariance`<br>`test_target_job_scope_oracle` |
| `DIAGNOSTICS.analytics_report` | What do the host and job series say together? | operational | `tests/test_job_host_analysis.py`<br>`tests/test_conclusion_model.py`<br>`tests/test_audit.py`<br>`tests/test_integration_contracts.py`<br>`tests/test_e2e_presence.py` | `test_report_summary_matches_pandas_oracle`<br>`test_background_influence_oracle`<br>`test_statistical_audit_replay` |
