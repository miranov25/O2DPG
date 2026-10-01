#!/usr/bin/env python3
"""
query_diagnostics.py - PHASE_13_75_DIAG DuckDB commissioning query layer.

This is a small, inspectable bridge:

    canonical/current diagnostics TSV bundle
        -> explicit-schema DuckDB raw tables
        -> ordinary SQL / reviewed SQL templates

It deliberately does NOT touch collection, time alignment, chain mode, or report
rendering.  The CLI follows the DataGSIAI D5 pattern: ordinary SQL remains the
query language; templates are convenience/example SQL and can always be shown
with --show-sql.

Current commissioning note
--------------------------
The PHASE_13_75 final TSV NULL/quoting contract is not yet fully banked.  The
Increment-1A transport currently uses empty fields for missing numeric values.
Therefore this commissioning loader uses nullstr='' for schema_version=1 input.
Before Acceptance-A, this must be tightened to the final canonical '\\N' dialect
with quoted-literal semantics required by the proposal.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional


DUCKDB_MIN = (1, 4, 5)
DUCKDB_MAX = (2, 0, 0)

# Explicit current raw schemas.  Keep column order identical to the TSV headers.
TABLE_SCHEMAS = {
    "host_samples": [
        ("ts", "DOUBLE"),
        ("elapsed_s", "DOUBLE"),
        ("compact_stall_total", "BIGINT"),
        ("compact_stall_per_s", "DOUBLE"),
        ("compact_fail_total", "BIGINT"),
        ("compact_fail_per_s", "DOUBLE"),
        ("thp_fault_alloc_total", "BIGINT"),
        ("thp_fault_alloc_per_s", "DOUBLE"),
        ("thp_fault_fallback_total", "BIGINT"),
        ("thp_fault_fallback_per_s", "DOUBLE"),
        ("thp_collapse_alloc_total", "BIGINT"),
        ("pgscan_direct_total", "BIGINT"),
        ("allocstall_total", "BIGINT"),
        ("allocstall_per_s", "DOUBLE"),
        ("kcompactd_cpu_s_total", "DOUBLE"),
        ("kcompactd_cpu_per_s", "DOUBLE"),
        ("khugepaged_cpu_s_total", "DOUBLE"),
        ("khugepaged_cpu_per_s", "DOUBLE"),
        ("psi_mem_some_avg10", "DOUBLE"),
        ("psi_mem_full_avg10", "DOUBLE"),
        ("pswpin_total", "BIGINT"),
        ("pswpout_total", "BIGINT"),
        # Current shell writer can render this cumulative counter in exponent form.
        ("disk_read_sectors_total", "DOUBLE"),
        ("sample_wall_s", "DOUBLE"),
        ("overrun", "INTEGER"),
        ("loadavg1", "DOUBLE"),
        ("cpu_busy_pct", "DOUBLE"),
        ("mem_available_kb", "BIGINT"),
        ("ctxt_per_s", "DOUBLE"),
        ("procs_running", "INTEGER"),
    ],
    "disk_samples": [
        ("ts", "DOUBLE"),
        ("device", "VARCHAR"),
        ("reads_completed", "BIGINT"),
        ("sectors_read", "BIGINT"),
        ("writes_completed", "BIGINT"),
        ("sectors_written", "BIGINT"),
        ("io_in_progress", "BIGINT"),
        ("io_time_ms", "BIGINT"),
    ],
    "network_samples": [
        ("ts", "DOUBLE"),
        ("iface", "VARCHAR"),
        ("rx_bytes", "BIGINT"),
        ("rx_packets", "BIGINT"),
        ("rx_errs", "BIGINT"),
        ("rx_drop", "BIGINT"),
        ("tx_bytes", "BIGINT"),
        ("tx_packets", "BIGINT"),
        ("tx_errs", "BIGINT"),
        ("tx_drop", "BIGINT"),
    ],
    "process_samples": [
        ("ts", "DOUBLE"),
        ("run_id", "VARCHAR"),
        ("tenant", "VARCHAR"),
        ("proc", "VARCHAR"),
        ("pid", "BIGINT"),
        ("starttime", "BIGINT"),
        ("is_target_job", "INTEGER"),
        ("state", "VARCHAR"),
        ("cpu_pct", "DOUBLE"),
        ("rss_kb", "BIGINT"),
        ("virt_kb", "BIGINT"),
        ("shr_kb", "BIGINT"),
        ("cpu_time_s", "DOUBLE"),
        ("rank_cpu_all", "INTEGER"),
        ("rank_mem_all", "INTEGER"),
        ("rank_cpu_user", "INTEGER"),
        ("rank_mem_user", "INTEGER"),
        ("selected_reason", "VARCHAR"),
    ],
    "process_catalog": [
        ("run_id", "VARCHAR"),
        ("process_id", "VARCHAR"),
        ("host_boot_id", "VARCHAR"),
        ("user_name", "VARCHAR"),
        ("uid", "BIGINT"),
        ("pid", "BIGINT"),
        ("starttime", "BIGINT"),
        ("ppid", "BIGINT"),
        ("parent_process_id", "VARCHAR"),
        ("command", "VARCHAR"),
        ("exe", "VARCHAR"),
        ("exe_state", "VARCHAR"),
        ("cmdline", "VARCHAR"),
        ("cmdline_state", "VARCHAR"),
        ("first_seen_ts", "DOUBLE"),
        ("master_pid", "BIGINT"),
        ("master_process_id", "VARCHAR"),
        ("is_master", "INTEGER"),
        ("is_target_job", "INTEGER"),
    ],
    "user_samples": [
        ("ts", "DOUBLE"),
        ("run_id", "VARCHAR"),
        ("tenant", "VARCHAR"),
        ("is_current_user", "INTEGER"),
        ("process_count", "INTEGER"),
        ("running_count", "INTEGER"),
        ("blocked_count", "INTEGER"),
        ("cpu_cores", "DOUBLE"),
        ("cpu_time_s", "DOUBLE"),
        ("rss_kb", "BIGINT"),
        ("virt_kb", "BIGINT"),
        ("io_read_bytes", "BIGINT"),
        ("io_write_bytes", "BIGINT"),
        ("io_coverage_state", "VARCHAR"),
    ],
    "workload_rollup": [
        ("ts", "DOUBLE"),
        ("run_id", "VARCHAR"),
        ("scope", "VARCHAR"),
        ("process_count", "INTEGER"),
        ("running_count", "INTEGER"),
        ("blocked_count", "INTEGER"),
        ("cpu_cores", "DOUBLE"),
        ("rss_kb", "BIGINT"),
        ("io_read_bytes", "BIGINT"),
        ("io_write_bytes", "BIGINT"),
        ("visible_process_count", "INTEGER"),
        ("readable_process_count", "INTEGER"),
        ("permission_denied_count", "INTEGER"),
        ("vanished_during_scan_count", "INTEGER"),
        ("new_process_count", "INTEGER"),
        ("accounting_lower_bound", "INTEGER"),
    ],
}

QUERY_TEMPLATES = (
    "bundle_summary",
    "host_summary",
    "top_processes",
    "top_average",
    "process_tree",
    "top_users",
    "disk_activity",
    "network_activity",
)

TEMPLATE_PARAM_SPECS = {
    "bundle_summary": {},
    "host_summary": {},
    "top_processes": {
        "limit": {"type": "int", "default": 20, "min": 1, "max": 10000},
    },
    "top_average": {
        "limit": {"type": "int", "default": 30, "min": 1, "max": 10000},
    },
    "process_tree": {},
    "top_users": {
        "limit": {"type": "int", "default": 20, "min": 1, "max": 10000},
    },
    "disk_activity": {
        "limit": {"type": "int", "default": 50, "min": 1, "max": 10000},
    },
    "network_activity": {
        "limit": {"type": "int", "default": 50, "min": 1, "max": 10000},
    },
}


class ConfigError(RuntimeError):
    pass


def _import_duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise ConfigError(
            "DuckDB is required for build/query commands; expected "
            "PHASE_13_75 target DuckDB >=1.4.5,<2.0"
        ) from exc

    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", duckdb.__version__)
    if not m:
        raise ConfigError(f"cannot parse DuckDB version {duckdb.__version__!r}")
    version = tuple(int(x) for x in m.groups())
    if version < DUCKDB_MIN or version >= DUCKDB_MAX:
        raise ConfigError(
            f"DuckDB {duckdb.__version__} outside validated target "
            f">={'.'.join(map(str, DUCKDB_MIN))},<2.0"
        )
    return duckdb


def _read_manifest(bundle: Path) -> dict:
    out = {}
    p = bundle / "manifest.kv"
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def _expected_header(table: str) -> list[str]:
    return [name for name, _ in TABLE_SCHEMAS[table]]


def _validate_bundle(bundle: Path) -> dict:
    bundle = bundle.resolve()
    if not bundle.is_dir():
        raise ConfigError(f"bundle directory does not exist: {bundle}")

    row_counts = {}
    for table in TABLE_SCHEMAS:
        path = bundle / f"{table}.tsv"
        if not path.is_file():
            raise ConfigError(f"required TSV missing: {path}")
        with path.open("r", encoding="utf-8", newline="") as f:
            header_line = f.readline().rstrip("\n")
        got = header_line.split("\t")
        want = _expected_header(table)
        if got != want:
            raise ConfigError(
                f"{path.name}: header mismatch\n"
                f"expected={want}\n"
                f"got={got}"
            )
        # Independent TSV-vs-DuckDB row-count oracle. Canonical free-text
        # fields in process/sample tables may contain quoted newlines, so count
        # logical TSV records for every table rather than physical lines.
        with path.open("r", encoding="utf-8", newline="") as f:
            row_counts[table] = max(
                sum(1 for _ in csv.reader(f, delimiter="\t", quotechar='"')) - 1,
                0,
            )

    return {
        "bundle": str(bundle),
        "manifest": _read_manifest(bundle),
        "tsv_row_counts": row_counts,
    }


def _columns_sql(schema) -> str:
    # Identifiers/types are fixed internal constants, never user input.
    pairs = ", ".join(f"'{name}': '{dtype}'" for name, dtype in schema)
    return "{" + pairs + "}"


def _build_db(bundle_path: str, db_path: str, replace: bool = False) -> dict:
    duckdb = _import_duckdb()
    bundle = Path(bundle_path).resolve()
    db = Path(db_path).resolve()
    validation = _validate_bundle(bundle)

    if db.exists() and not replace:
        raise ConfigError(f"database already exists: {db} (use --replace)")
    db.parent.mkdir(parents=True, exist_ok=True)

    tmp = db.with_name(db.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    con = duckdb.connect(str(tmp))
    try:
        loaded_counts = {}
        for table, schema in TABLE_SCHEMAS.items():
            path = str(bundle / f"{table}.tsv")
            read_schema = schema
            select_sql = "*"
            if table == "host_samples":
                # The inherited collector can emit elapsed_s=INVALID for a
                # valid first/delta row when elapsed time cannot be derived.
                # Read only that source field as text and map the documented
                # sentinel to SQL NULL; unexpected non-numeric text must still
                # fail rather than being silently coerced.
                read_schema = [
                    (name, "VARCHAR" if name == "elapsed_s" else dtype)
                    for name, dtype in schema
                ]
                select_sql = ", ".join(
                    (
                        "CASE WHEN elapsed_s IS NULL OR elapsed_s = 'INVALID' "
                        "THEN NULL ELSE CAST(elapsed_s AS DOUBLE) END AS elapsed_s"
                        if name == "elapsed_s" else name
                    )
                    for name, _dtype in schema
                )
            columns = _columns_sql(read_schema)
            # process_catalog uses the PHASE_13_75 canonical NULL token.
            # Inject the SQL literal explicitly so DuckDB sees exactly one
            # backslash followed by N, not Python repr escaping.
            nullstr_sql = r"'\N'" if table == "process_catalog" else "''"
            sql = f"""
CREATE TABLE {table} AS
SELECT {select_sql}
FROM read_csv(
    ?,
    delim='\\t',
    header=true,
    columns={columns},
    nullstr={nullstr_sql},
    quote='"',
    escape='"',
    auto_detect=false,
    allow_quoted_nulls=false
)
"""
            con.execute(sql, [path])
            loaded_counts[table] = con.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

        for table, expected in validation["tsv_row_counts"].items():
            if loaded_counts[table] != expected:
                raise ConfigError(
                    f"{table}: TSV row count {expected} != DuckDB {loaded_counts[table]}"
                )

        con.execute("""
CREATE VIEW process_samples_named AS
SELECT
    ps.*,
    pc.process_id,
    pc.host_boot_id,
    ps.tenant AS user_name,
    pc.user_name AS catalog_user_name,
    pc.uid AS catalog_uid,
    pc.ppid,
    pc.parent_process_id,
    ps.proc AS command,
    pc.command AS catalog_command,
    pc.exe,
    pc.exe_state,
    pc.cmdline,
    pc.cmdline_state,
    pc.first_seen_ts,
    pc.master_pid,
    pc.master_process_id,
    pc.is_master
FROM process_samples AS ps
LEFT JOIN process_catalog AS pc
  ON pc.run_id = ps.run_id
 AND pc.pid = ps.pid
 AND pc.starttime = ps.starttime
""")

        con.execute(
            "CREATE TABLE _diagnostics_meta "
            "(key VARCHAR PRIMARY KEY, value VARCHAR)"
        )
        manifest = validation["manifest"]
        meta = {
            "source_bundle": str(bundle),
            "bundle_name": bundle.name,
            "run_id": manifest.get("run_id", ""),
            "host": manifest.get("host", ""),
            "bundle_utc": manifest.get("utc", ""),
            "schema_version": manifest.get("schema_version", ""),
            "bundle_verdict": manifest.get("verdict", ""),
            "bundle_finalized": "1" if manifest.get("verdict") else "0",
            "duckdb_version": duckdb.__version__,
            "loader_null_contract": "legacy_tables_empty_null;process_catalog_unquoted_backslashN_null",
        }
        con.executemany(
            "INSERT INTO _diagnostics_meta VALUES (?, ?)",
            list(meta.items()),
        )
        con.execute("CHECKPOINT")
    finally:
        con.close()

    os.replace(tmp, db)
    return {
        "result": "OK",
        "db": str(db),
        "bundle": str(bundle),
        "duckdb_version": duckdb.__version__,
        "row_counts": loaded_counts,
        "bundle_finalized": bool(validation["manifest"].get("verdict")),
    }


def _parse_template_params(name: str, raw: Optional[str]) -> dict:
    if name not in TEMPLATE_PARAM_SPECS:
        raise ConfigError(f"unknown query template {name!r}")
    specs = TEMPLATE_PARAM_SPECS[name]

    supplied = {}
    if raw:
        for item in raw.split(";"):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ConfigError(
                    "--template-params entries must use key=value separated by ';'"
                )
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                raise ConfigError("template parameters require non-empty key=value")
            if key in supplied:
                raise ConfigError(f"duplicate template parameter {key!r}")
            if key not in specs:
                allowed = ", ".join(sorted(specs)) or "none"
                raise ConfigError(
                    f"template {name!r} does not accept {key!r}; allowed: {allowed}"
                )
            supplied[key] = value

    cooked = {}
    for key, spec in specs.items():
        value = supplied.get(key, spec["default"])
        if spec["type"] != "int":
            raise ConfigError(f"unsupported parameter type for {key!r}")
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"template parameter {key!r} must be an integer") from exc
        if value < spec["min"] or value > spec["max"]:
            raise ConfigError(
                f"template parameter {key!r} must be between "
                f"{spec['min']} and {spec['max']}"
            )
        cooked[key] = value
    return cooked


def _template_sql(name: str, params: Optional[dict] = None) -> str:
    if name not in TEMPLATE_PARAM_SPECS:
        raise ConfigError(f"unknown query template {name!r}")
    values = _parse_template_params(name, None)
    if params:
        unknown = sorted(set(params) - set(TEMPLATE_PARAM_SPECS[name]))
        if unknown:
            raise ConfigError(
                f"template {name!r} does not accept parameter {unknown[0]!r}"
            )
        values.update(params)

    if name == "bundle_summary":
        parts = []
        for table in TABLE_SCHEMAS:
            tcol = "first_seen_ts" if table == "process_catalog" else "ts"
            parts.append(
                f"SELECT '{table}' AS relation, COUNT(*) AS rows, "
                f"MIN({tcol}) AS ts_min, MAX({tcol}) AS ts_max FROM {table}"
            )
        return ("\nUNION ALL\n".join(parts) + "\nORDER BY relation").strip()

    if name == "host_summary":
        return """
SELECT
    COUNT(*) AS samples,
    MIN(ts) AS ts_min,
    MAX(ts) AS ts_max,
    MAX(ts) - MIN(ts) AS duration_s,
    ROUND(AVG(cpu_busy_pct), 3) AS avg_cpu_busy_pct,
    MAX(cpu_busy_pct) AS max_cpu_busy_pct,
    ROUND(AVG(loadavg1), 3) AS avg_loadavg1,
    MAX(loadavg1) AS max_loadavg1,
    MIN(mem_available_kb) AS min_mem_available_kb,
    ROUND(AVG(ctxt_per_s), 3) AS avg_ctxt_per_s,
    MAX(ctxt_per_s) AS max_ctxt_per_s,
    MAX(procs_running) AS max_procs_running,
    ROUND(AVG(sample_wall_s), 4) AS avg_sample_wall_s,
    MAX(sample_wall_s) AS max_sample_wall_s,
    SUM(COALESCE(overrun, 0)) AS overruns,
    MAX(compact_stall_per_s) AS max_compact_stall_per_s,
    MAX(allocstall_per_s) AS max_allocstall_per_s,
    MAX(kcompactd_cpu_per_s) AS max_kcompactd_cpu_per_s
FROM host_samples
""".strip()

    limit = values.get("limit")

    if name == "top_processes":
        return f"""
SELECT
    tenant,
    proc,
    MAX(is_target_job) AS is_target_job,
    COUNT(*) AS selected_samples,
    ROUND(AVG(cpu_pct), 3) AS avg_cpu_pct,
    MAX(cpu_pct) AS max_cpu_pct,
    MAX(rss_kb) AS max_rss_kb,
    MAX(virt_kb) AS max_virt_kb,
    MAX(cpu_time_s) AS max_cpu_time_s
FROM process_samples
GROUP BY tenant, proc
ORDER BY max_rss_kb DESC, max_cpu_pct DESC NULLS LAST, tenant ASC, proc ASC
LIMIT {limit}
""".strip()

    if name == "top_average":
        return f"""
WITH nt AS (
    SELECT COUNT(DISTINCT ts) AS n FROM user_samples
),
process_at_time AS (
    SELECT
        ts,
        COALESCE(user_name, tenant) AS user_name,
        COALESCE(command, proc) AS command,
        COUNT(*) AS n_processes,
        SUM(cpu_pct) / 100.0 AS cpu_cores,
        SUM(rss_kb) / 1048576.0 AS rss_gb
    FROM process_samples_named
    GROUP BY ts, COALESCE(user_name, tenant), COALESCE(command, proc)
)
SELECT
    user_name AS user,
    command,
    COUNT(*) AS sampled_times,
    ROUND(100.0 * COUNT(*) / NULLIF((SELECT n FROM nt), 0), 1) AS coverage_pct,
    ROUND(AVG(n_processes), 2) AS avg_processes,
    MAX(n_processes) AS max_processes,
    ROUND(AVG(cpu_cores), 2) AS avg_cpu_cores_when_captured,
    ROUND(MAX(cpu_cores), 2) AS max_cpu_cores,
    ROUND(AVG(rss_gb), 2) AS avg_rss_gb_when_captured,
    ROUND(MAX(rss_gb), 2) AS max_rss_gb
FROM process_at_time
GROUP BY user_name, command
ORDER BY avg_cpu_cores_when_captured DESC, avg_rss_gb_when_captured DESC, user ASC, command ASC
LIMIT {limit}
""".strip()

    if name == "process_tree":
        return """
WITH RECURSIVE tree AS (
    SELECT
        process_id, parent_process_id, master_process_id, pid, ppid,
        user_name, command, cmdline, is_master, is_target_job, 0 AS depth,
        printf('%020d', pid) AS tree_path
    FROM process_catalog
    WHERE is_master = 1
    UNION ALL
    SELECT
        c.process_id, c.parent_process_id, c.master_process_id, c.pid, c.ppid,
        c.user_name, c.command, c.cmdline, c.is_master, c.is_target_job, t.depth + 1,
        t.tree_path || '/' || printf('%020d', c.pid) AS tree_path
    FROM process_catalog AS c
    JOIN tree AS t ON c.parent_process_id = t.process_id
    WHERE c.is_target_job = 1
)
SELECT
    depth,
    repeat('  ', depth) || command AS process_tree,
    user_name AS user,
    pid, ppid, cmdline
FROM tree
ORDER BY tree_path
""".strip()

    if name == "top_users":
        return f"""
SELECT
    tenant,
    MAX(is_current_user) AS is_current_user,
    COUNT(DISTINCT ts) AS sampled_timestamps,
    MAX(process_count) AS max_process_count,
    ROUND(AVG(cpu_cores), 3) AS avg_cpu_cores,
    MAX(cpu_cores) AS max_cpu_cores,
    MAX(rss_kb) AS max_rss_kb,
    MAX(virt_kb) AS max_virt_kb,
    MAX(io_read_bytes) AS max_io_read_bytes,
    MAX(io_write_bytes) AS max_io_write_bytes
FROM user_samples
GROUP BY tenant
ORDER BY max_rss_kb DESC, max_cpu_cores DESC, tenant ASC
LIMIT {limit}
""".strip()

    if name == "disk_activity":
        return f"""
SELECT
    device,
    COUNT(*) AS samples,
    MAX(ts) - MIN(ts) AS duration_s,
    MAX(sectors_read) - MIN(sectors_read) AS delta_sectors_read,
    MAX(sectors_written) - MIN(sectors_written) AS delta_sectors_written,
    ROUND(
        (MAX(sectors_read) - MIN(sectors_read))
        / NULLIF(MAX(ts) - MIN(ts), 0), 3
    ) AS sectors_read_per_s,
    ROUND(
        (MAX(sectors_written) - MIN(sectors_written))
        / NULLIF(MAX(ts) - MIN(ts), 0), 3
    ) AS sectors_written_per_s,
    MAX(io_in_progress) AS max_io_in_progress,
    MAX(io_time_ms) - MIN(io_time_ms) AS delta_io_time_ms
FROM disk_samples
GROUP BY device
ORDER BY
    (MAX(sectors_read) - MIN(sectors_read))
      + (MAX(sectors_written) - MIN(sectors_written)) DESC,
    device ASC
LIMIT {limit}
""".strip()

    if name == "network_activity":
        return f"""
SELECT
    iface,
    COUNT(*) AS samples,
    MAX(ts) - MIN(ts) AS duration_s,
    MAX(rx_bytes) - MIN(rx_bytes) AS delta_rx_bytes,
    MAX(tx_bytes) - MIN(tx_bytes) AS delta_tx_bytes,
    ROUND(
        (MAX(rx_bytes) - MIN(rx_bytes))
        / NULLIF(MAX(ts) - MIN(ts), 0), 3
    ) AS rx_bytes_per_s,
    ROUND(
        (MAX(tx_bytes) - MIN(tx_bytes))
        / NULLIF(MAX(ts) - MIN(ts), 0), 3
    ) AS tx_bytes_per_s,
    MAX(rx_errs) - MIN(rx_errs) AS delta_rx_errs,
    MAX(tx_errs) - MIN(tx_errs) AS delta_tx_errs,
    MAX(rx_drop) - MIN(rx_drop) AS delta_rx_drop,
    MAX(tx_drop) - MIN(tx_drop) AS delta_tx_drop
FROM network_samples
GROUP BY iface
ORDER BY
    (MAX(rx_bytes) - MIN(rx_bytes))
      + (MAX(tx_bytes) - MIN(tx_bytes)) DESC,
    iface ASC
LIMIT {limit}
""".strip()

    raise ConfigError(f"template implementation missing for {name!r}")


def _query_sql(db_path: str, sql: str) -> dict:
    duckdb = _import_duckdb()
    db = Path(db_path).resolve()
    if not db.is_file():
        raise ConfigError(f"DuckDB database does not exist: {db}")

    query = sql.strip()
    while query.endswith(";"):
        query = query[:-1].rstrip()
    if not query:
        raise ConfigError("SQL query is empty")

    con = duckdb.connect(str(db), read_only=True)
    try:
        # Same principle as DataGSIAI D5: user input must be query-shaped.
        cur = con.execute(
            f"SELECT * FROM ({query}) AS diagnostics_user_query"
        )
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
    finally:
        con.close()

    return {
        "result": "OK",
        "db": str(db),
        "columns": columns,
        "rows": [[_json_safe(v) for v in row] for row in rows],
        "result_row_count": len(rows),
    }


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _format_result(result: dict, output_format: str) -> str:
    columns = result["columns"]
    rows = result["rows"]

    if output_format == "json":
        return json.dumps(
            {
                "result": result["result"],
                "result_row_count": result["result_row_count"],
                "columns": columns,
                "rows": rows,
            },
            indent=2,
        )

    if output_format in ("csv", "tsv"):
        s = io.StringIO()
        w = csv.writer(
            s,
            delimiter="," if output_format == "csv" else "\t",
            lineterminator="\n",
        )
        w.writerow(columns)
        w.writerows(rows)
        return s.getvalue().rstrip("\n")

    if output_format == "table":
        text_rows = [
            ["" if v is None else str(v) for v in row]
            for row in rows
        ]
        widths = [len(str(c)) for c in columns]
        for row in text_rows:
            for i, v in enumerate(row):
                widths[i] = max(widths[i], len(v))
        header = "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(columns))
        rule = "  ".join("-" * widths[i] for i in range(len(columns)))
        body = [
            "  ".join(v.ljust(widths[i]) for i, v in enumerate(row))
            for row in text_rows
        ]
        return "\n".join([header, rule, *body])

    raise ConfigError(f"unknown output format {output_format!r}")


def _print_build(result: dict):
    print(f"db={result['db']}")
    print(f"bundle={result['bundle']}")
    print(f"duckdb={result['duckdb_version']}")
    print(f"bundle_finalized={int(result['bundle_finalized'])}")
    for table in TABLE_SCHEMAS:
        print(f"{table:20s} rows={result['row_counts'][table]}")


def _build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-db", help="Build one derived DuckDB database from one TSV bundle.")
    p.add_argument("--bundle", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--replace", action="store_true")

    sub.add_parser("list-templates", help="List built-in SQL templates.")

    p = sub.add_parser("query-template", help="Show or execute a built-in ordinary-SQL template.")
    p.add_argument("--db", help="DuckDB database; required unless --show-sql.")
    p.add_argument("--template", required=True, choices=QUERY_TEMPLATES)
    p.add_argument(
        "--template-params",
        help="Template-local parameters as 'key=value;key=value'.",
    )
    p.add_argument("--show-sql", action="store_true")
    p.add_argument(
        "--format",
        choices=("json", "table", "csv", "tsv"),
        default="table",
    )

    p = sub.add_parser("query-sql", help="Execute ordinary read-only query-shaped DuckDB SQL.")
    p.add_argument("--db", required=True)
    q = p.add_mutually_exclusive_group(required=True)
    q.add_argument("--sql")
    q.add_argument("--sql-file")
    p.add_argument(
        "--format",
        choices=("json", "table", "csv", "tsv"),
        default="table",
    )
    return ap


def main(argv=None):
    ap = _build_parser()
    args = ap.parse_args(argv)

    try:
        if args.command == "build-db":
            _print_build(_build_db(args.bundle, args.db, replace=args.replace))
            return 0

        if args.command == "list-templates":
            for name in QUERY_TEMPLATES:
                params = TEMPLATE_PARAM_SPECS[name]
                if params:
                    desc = ";".join(
                        f"{k}={v['default']}[{v['min']}..{v['max']}]"
                        for k, v in params.items()
                    )
                else:
                    desc = "no parameters"
                print(f"{name:20s} {desc}")
            return 0

        if args.command == "query-template":
            params = _parse_template_params(args.template, args.template_params)
            sql = _template_sql(args.template, params)
            if args.show_sql:
                print(sql)
                return 0
            if not args.db:
                raise ConfigError("query-template execution requires --db")
            result = _query_sql(args.db, sql)
            print(_format_result(result, args.format))
            return 0

        if args.command == "query-sql":
            sql = args.sql
            if args.sql_file:
                sql = Path(args.sql_file).read_text(encoding="utf-8")
            result = _query_sql(args.db, sql)
            print(_format_result(result, args.format))
            return 0

        raise ConfigError(f"unknown command {args.command!r}")
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
