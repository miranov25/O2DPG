from pathlib import Path
import importlib.util
import sys

import pytest


HERE = Path(__file__).resolve().parents[1]
MODULE = HERE / "query_diagnostics.py"


def _load():
    spec = importlib.util.spec_from_file_location("query_diagnostics", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_template_parameter_validation():
    q = _load()
    assert q._parse_template_params("top_processes", None) == {"limit": 20}
    assert q._parse_template_params("top_processes", "limit=7") == {"limit": 7}
    assert q._parse_template_params("top_average", "limit=7") == {"limit": 7}
    assert q._parse_template_params("process_tree", None) == {}
    with pytest.raises(q.ConfigError):
        q._parse_template_params("top_processes", "bogus=1")
    with pytest.raises(q.ConfigError):
        q._parse_template_params("top_processes", "limit=0")
    with pytest.raises(q.ConfigError):
        q._parse_template_params("top_processes", "limit=2;limit=3")


def test_templates_are_plain_inspectable_sql():
    q = _load()
    for name in q.QUERY_TEMPLATES:
        sql = q._template_sql(name)
        assert "SELECT" in sql.upper()
        assert "{{" not in sql
        assert "}}" not in sql


def _row_for(schema, table):
    vals = []
    for name, dtype in schema:
        if table == "process_catalog" and name in (
            "parent_process_id", "exe", "master_pid", "master_process_id"
        ):
            vals.append(r"\N")
        elif dtype == "VARCHAR":
            if name == "run_id":
                vals.append("r1")
            elif name == "tenant":
                vals.append("u1")
            elif name == "proc":
                vals.append("p1")
            elif name == "state":
                vals.append("S")
            elif name == "selected_reason":
                vals.append("rank_mem_all")
            elif name == "io_coverage_state":
                vals.append("complete")
            elif name == "scope":
                vals.append("current_user_non_job")
            elif name == "device":
                vals.append("nvme0")
            elif name == "iface":
                vals.append("eth0")
            else:
                vals.append("x")
        elif name == "ts":
            vals.append("100")
        elif dtype == "DOUBLE":
            vals.append("1.0")
        else:
            vals.append("1")
    return vals


def _write_bundle(q, root):
    root.mkdir()
    (root / "manifest.kv").write_text(
        "schema_version=1\nrun_id=r1\nhost=testhost\nutc=20260930T000000Z\nverdict=PASS\n"
    )
    for table, schema in q.TABLE_SCHEMAS.items():
        header = "\t".join(name for name, _ in schema)
        row = "\t".join(_row_for(schema, table))
        (root / f"{table}.tsv").write_text(header + "\n" + row + "\n")



def _set_first_tsv_value(path, column, value):
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    row = lines[1].split("\t")
    row[header.index(column)] = value
    path.write_text("\t".join(header) + "\n" + "\t".join(row) + "\n")


def _write_catalog_rows(q, path, overrides):
    schema = q.TABLE_SCHEMAS["process_catalog"]
    names = [name for name, _dtype in schema]
    base = dict(zip(names, _row_for(schema, "process_catalog")))
    lines = ["\t".join(names)]
    for change in overrides:
        row = dict(base)
        row.update(change)
        lines.append("\t".join(str(row[name]) for name in names))
    path.write_text("\n".join(lines) + "\n")


def test_real_duckdb_build_and_templates(tmp_path):
    pytest.importorskip("duckdb")
    q = _load()
    bundle = tmp_path / "bundle"
    _write_bundle(q, bundle)
    db = tmp_path / "diag.duckdb"

    result = q._build_db(str(bundle), str(db))
    assert db.is_file()
    assert all(n == 1 for n in result["row_counts"].values())

    bundle_summary = q._query_sql(str(db), q._template_sql("bundle_summary"))
    assert bundle_summary["result_row_count"] == len(q.TABLE_SCHEMAS)
    assert {r[0] for r in bundle_summary["rows"]} == set(q.TABLE_SCHEMAS)

    host_summary = q._query_sql(str(db), q._template_sql("host_summary"))
    assert host_summary["rows"][0][0] == 1

    # Canonical unquoted \N must load as SQL NULL, including BIGINT master_pid.
    nulls = q._query_sql(
        str(db),
        "SELECT parent_process_id, exe, master_pid, master_process_id "
        "FROM process_catalog",
    )
    assert nulls["rows"] == [[None, None, None, None]]


def test_header_mismatch_fails_before_duckdb(tmp_path):
    q = _load()
    bundle = tmp_path / "bundle"
    _write_bundle(q, bundle)
    p = bundle / "host_samples.tsv"
    p.write_text("wrong\theader\n1\t2\n")
    with pytest.raises(q.ConfigError):
        q._validate_bundle(bundle)



def test_process_sample_command_overrides_stale_catalog_command(tmp_path):
    pytest.importorskip("duckdb")
    q = _load()
    bundle = tmp_path / "bundle"
    _write_bundle(q, bundle)
    _set_first_tsv_value(bundle / "process_samples.tsv", "proc", "realjob")
    _set_first_tsv_value(bundle / "process_catalog.tsv", "command", "wrapper")
    db = tmp_path / "diag.duckdb"

    q._build_db(str(bundle), str(db))
    named = q._query_sql(
        str(db),
        "SELECT command, catalog_command FROM process_samples_named",
    )
    assert named["rows"] == [["realjob", "wrapper"]]

    top = q._query_sql(str(db), q._template_sql("top_average", {"limit": 10}))
    assert top["rows"][0][1] == "realjob"


def test_invalid_elapsed_s_is_loadable_and_becomes_null(tmp_path):
    pytest.importorskip("duckdb")
    q = _load()
    bundle = tmp_path / "bundle"
    _write_bundle(q, bundle)
    _set_first_tsv_value(bundle / "host_samples.tsv", "elapsed_s", "INVALID")
    db = tmp_path / "diag.duckdb"

    q._build_db(str(bundle), str(db))
    result = q._query_sql(str(db), "SELECT elapsed_s FROM host_samples")
    assert result["rows"] == [[None]]


def test_process_tree_orders_descendants_under_real_parent(tmp_path):
    pytest.importorskip("duckdb")
    q = _load()
    bundle = tmp_path / "bundle"
    _write_bundle(q, bundle)
    _write_catalog_rows(q, bundle / "process_catalog.tsv", [
        {
            "process_id": "root", "pid": "10", "starttime": "100", "ppid": "1",
            "parent_process_id": r"\N", "command": "root", "master_pid": "10",
            "master_process_id": "root", "is_master": "1", "is_target_job": "1",
        },
        {
            "process_id": "child-a", "pid": "20", "starttime": "101", "ppid": "10",
            "parent_process_id": "root", "command": "childA", "master_pid": "10",
            "master_process_id": "root", "is_master": "0", "is_target_job": "1",
        },
        {
            "process_id": "grand", "pid": "30", "starttime": "102", "ppid": "20",
            "parent_process_id": "child-a", "command": "grand", "master_pid": "10",
            "master_process_id": "root", "is_master": "0", "is_target_job": "1",
        },
        {
            "process_id": "child-b", "pid": "40", "starttime": "103", "ppid": "10",
            "parent_process_id": "root", "command": "childB", "master_pid": "10",
            "master_process_id": "root", "is_master": "0", "is_target_job": "1",
        },
    ])
    db = tmp_path / "diag.duckdb"

    q._build_db(str(bundle), str(db))
    tree = q._query_sql(str(db), q._template_sql("process_tree"))
    rendered = [row[1] for row in tree["rows"]]
    assert rendered == ["root", "  childA", "    grand", "  childB"]


def test_process_sample_user_overrides_stale_catalog_user(tmp_path):
    pytest.importorskip("duckdb")
    q = _load()
    bundle = tmp_path / "bundle"
    _write_bundle(q, bundle)
    _set_first_tsv_value(bundle / "process_samples.tsv", "tenant", "newuser")
    _set_first_tsv_value(bundle / "process_catalog.tsv", "user_name", "olduser")
    db = tmp_path / "diag.duckdb"

    q._build_db(str(bundle), str(db))
    named = q._query_sql(
        str(db),
        "SELECT user_name, catalog_user_name FROM process_samples_named",
    )
    assert named["rows"] == [["newuser", "olduser"]]

    top = q._query_sql(str(db), q._template_sql("top_average", {"limit": 10}))
    assert top["rows"][0][0] == "newuser"


def test_hostile_process_command_roundtrips_through_duckdb(tmp_path):
    pytest.importorskip("duckdb")
    q = _load()
    c_spec = importlib.util.spec_from_file_location("collector_for_tsv_test", HERE / "collector.py")
    c = importlib.util.module_from_spec(c_spec)
    c_spec.loader.exec_module(c)

    bundle = tmp_path / "bundle"
    _write_bundle(q, bundle)
    schema = q.TABLE_SCHEMAS["process_samples"]
    names = [name for name, _dtype in schema]
    values = _row_for(schema, "process_samples")
    hostile = 'bad\t"quote"\nline\\N'
    values[names.index("proc")] = hostile
    sample_path = bundle / "process_samples.tsv"
    sample_path.write_text("\t".join(names) + "\n", encoding="utf-8")
    c._write_tsv_row(sample_path, values)
    db = tmp_path / "diag.duckdb"

    q._build_db(str(bundle), str(db))
    result = q._query_sql(str(db), "SELECT proc FROM process_samples")
    assert result["rows"] == [[hostile]]
