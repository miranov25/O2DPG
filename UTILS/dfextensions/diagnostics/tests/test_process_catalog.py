import csv
import importlib.util
import os
import shutil
from pathlib import Path
from types import SimpleNamespace


def _load_collector(path):
    spec = importlib.util.spec_from_file_location("collector_candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_proc(root, pid, comm, ppid, starttime, uid=0, ticks=100, rss_pages=10, cmdline=None, exe="/usr/bin/python3"):
    d = root / str(pid)
    d.mkdir(parents=True)
    after = (
        f"S {ppid} 0 0 0 -1 0 0 0 0 0 {ticks} 0 0 0 20 0 1 0 "
        f"{starttime} {1 << 20} {rss_pages} " + " ".join(["0"] * 30)
    )
    (d / "stat").write_text(f"{pid} ({comm}) {after}\n")
    (d / "statm").write_text(f"20 {rss_pages} 2 1 0 1 0\n")
    (d / "status").write_text(f"Name:\t{comm}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    (d / "cmdline").write_bytes(cmdline or (comm.encode() + b"\0"))
    (d / "exe").symlink_to(exe)


def test_process_catalog_identity_hierarchy_and_free_text(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("boot-test\n")

    _make_proc(proc, 50, "o2-dpl-run", 1, 111, cmdline=b"o2-dpl-run\0--workflow\0main\0")
    _make_proc(
        proc, 51, "root.exe", 50, 112,
        cmdline=b"root.exe\0-b\0-q\0macro.C(\"a\\tb\")\0literal\\N\0",
    )

    monkeypatch.setenv("PROC_ROOT", str(proc))
    monkeypatch.setenv("CLK_TCK_OVERRIDE", "100")
    monkeypatch.setenv("DFX_UID_OVERRIDE", "0")
    monkeypatch.setenv("DFX_BOOTID_OVERRIDE", "boot-test")
    monkeypatch.setenv("DFX_TABLE_FORMAT", "tsv")

    candidate = Path(__file__).resolve().parents[1] / "collector.py"
    c = _load_collector(candidate)

    pidfile = tmp_path / "master.pid"
    pidfile.write_text("50\n")
    out = tmp_path / "bundle"

    rc = c.main([
        "-o", str(out), "--run-id", "run-test", "--interval", "0.01",
        "--nsamples", "1", "--target-pid-file", str(pidfile),
    ])
    assert rc == 0

    with (out / "process_catalog.tsv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quotechar='"'))
    by_pid = {int(r["pid"]): r for r in rows}
    assert set(by_pid) == {50, 51}

    master = by_pid[50]
    child = by_pid[51]
    assert master["command"] == "o2-dpl-run"
    assert master["cmdline"] == "o2-dpl-run --workflow main"
    assert master["is_master"] == "1"
    assert master["is_target_job"] == "1"

    assert child["command"] == "root.exe"
    assert child["ppid"] == "50"
    assert child["parent_process_id"] == master["process_id"]
    assert child["master_process_id"] == master["process_id"]
    assert child["is_master"] == "0"
    assert child["is_target_job"] == "1"
    assert "literal\\N" in child["cmdline"]
    assert "\\tb" in child["cmdline"]

    # Stable process-instance identity and no per-sample duplication.
    assert master["process_id"] == c.process_id("boot-test", 50, 111)
    assert child["process_id"] == c.process_id("boot-test", 51, 112)



def _init_sample_outputs(c, out):
    out.mkdir()
    for name, hdr in (
        ("process_samples.tsv", c.PROC_HEADER),
        ("user_samples.tsv", c.USER_HEADER),
        ("workload_rollup.tsv", c.ROLL_HEADER),
    ):
        (out / name).write_text(hdr.replace(",", "\t") + "\n")
    (out / "process_catalog.tsv").write_text(c.CATALOG_HEADER.replace(",", "\t") + "\n")


def test_master_pid_reuse_does_not_rebind_target_root(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("boot-reuse\n")
    _make_proc(proc, 50, "master", 1, 111, cmdline=b"master\0")
    _make_proc(proc, 51, "child", 50, 112, cmdline=b"child\0")

    monkeypatch.setenv("PROC_ROOT", str(proc))
    monkeypatch.setenv("CLK_TCK_OVERRIDE", "100")
    monkeypatch.setenv("DFX_UID_OVERRIDE", "0")
    monkeypatch.setenv("DFX_BOOTID_OVERRIDE", "boot-reuse")
    monkeypatch.setenv("DFX_TABLE_FORMAT", "tsv")
    c = _load_collector(Path(__file__).resolve().parents[1] / "collector.py")

    pidfile = tmp_path / "master.pid"
    pidfile.write_text("50\n")
    out = tmp_path / "bundle"
    _init_sample_outputs(c, out)
    args = SimpleNamespace(
        nall=20, nuser=20, max_d_state_rows=20,
        target_pid_file=str(pidfile), job_cgroup=None,
    )
    red = c.Redactor(c.uid_name(0), raw=True)
    seen = set()
    master_state = {}

    prev = c.sample_once(
        None, 1.0, args, red, 0, "r1", 1.0, "boot-reuse", out,
        seen, master_state,
    )

    # Original root exits; Linux reuses the SAME numeric PID for unrelated work.
    shutil.rmtree(proc / "50")
    shutil.rmtree(proc / "51")
    _make_proc(proc, 50, "unrelated", 1, 999, cmdline=b"unrelated\0")
    _make_proc(proc, 52, "unrelated-child", 50, 1000, cmdline=b"unrelated-child\0")

    c.sample_once(
        prev, 1.0, args, red, 0, "r1", 2.0, "boot-reuse", out,
        seen, master_state,
    )

    with (out / "process_samples.tsv").open(newline="", encoding="utf-8") as f:
        samples = list(csv.DictReader(f, delimiter="\t"))
    second = [r for r in samples if r["ts"] == "2.0"]
    by_pid = {int(r["pid"]): r for r in second}
    assert by_pid[50]["is_target_job"] == "0"
    assert by_pid[52]["is_target_job"] == "0"

    with (out / "process_catalog.tsv").open(newline="", encoding="utf-8") as f:
        catalog = list(csv.DictReader(f, delimiter="\t", quotechar='"'))
    masters = [r for r in catalog if r["is_master"] == "1"]
    assert len(masters) == 1
    assert masters[0]["pid"] == "50"
    assert masters[0]["starttime"] == "111"


def test_fractional_sampler_timestamps_are_unique(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("boot-ts\n")
    _make_proc(proc, 60, "worker", 1, 211, cmdline=b"worker\0")

    monkeypatch.setenv("PROC_ROOT", str(proc))
    monkeypatch.setenv("CLK_TCK_OVERRIDE", "100")
    monkeypatch.setenv("DFX_UID_OVERRIDE", "0")
    monkeypatch.setenv("DFX_BOOTID_OVERRIDE", "boot-ts")
    monkeypatch.setenv("DFX_TABLE_FORMAT", "tsv")
    c = _load_collector(Path(__file__).resolve().parents[1] / "collector.py")

    out = tmp_path / "bundle"
    assert c.main([
        "-o", str(out), "--run-id", "r-ts",
        "--interval", "0.01", "--nsamples", "3",
    ]) == 0

    with (out / "user_samples.tsv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    timestamps = [r["ts"] for r in rows]
    assert len(timestamps) == 3
    assert len(set(timestamps)) == 3
    assert any("." in ts for ts in timestamps)



def test_exec_refreshes_metadata_for_same_comm_and_preserves_identity(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("boot-exec\n")
    _make_proc(
        proc, 70, "worker", 1, 311, uid=111111,
        cmdline=b"wrapper\0--launch\0", exe="/opt/wrapper",
    )

    monkeypatch.setenv("PROC_ROOT", str(proc))
    monkeypatch.setenv("CLK_TCK_OVERRIDE", "100")
    monkeypatch.setenv("DFX_UID_OVERRIDE", "111111")
    monkeypatch.setenv("DFX_BOOTID_OVERRIDE", "boot-exec")
    monkeypatch.setenv("DFX_TABLE_FORMAT", "tsv")
    c = _load_collector(Path(__file__).resolve().parents[1] / "collector.py")

    out = tmp_path / "bundle"
    _init_sample_outputs(c, out)
    args = SimpleNamespace(
        nall=20, nuser=20, max_d_state_rows=20,
        target_pid_file=None, job_cgroup=None,
    )
    red = c.Redactor(c.uid_name(111111), raw=True)
    catalog = {}
    master_state = {}

    prev = c.sample_once(
        None, 1.0, args, red, 111111, "r1", 1.0, "boot-exec", out,
        catalog, master_state,
    )
    process_key = c.process_id("boot-exec", 70, 311)

    # exec()/credential transition keeps pid/starttime/comm but changes the
    # mutable process description and current user identity.
    shutil.rmtree(proc / "70")
    _make_proc(
        proc, 70, "worker", 1, 311, uid=222222, ticks=200,
        cmdline=b"realjob\0--work\0", exe="/opt/realjob",
    )
    c.sample_once(
        prev, 1.0, args, red, 111111, "r1", 2.0, "boot-exec", out,
        catalog, master_state,
    )

    with (out / "process_catalog.tsv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quotechar='"'))
    assert len(rows) == 1
    assert rows[0]["process_id"] == process_key
    assert rows[0]["first_seen_ts"] == "1.0"
    assert rows[0]["command"] == "worker"
    assert rows[0]["user_name"] == "uid222222"
    assert rows[0]["uid"] == "222222"
    assert rows[0]["exe"] == "/opt/realjob"
    assert rows[0]["cmdline"] == "realjob --work"

    with (out / "process_samples.tsv").open(newline="", encoding="utf-8") as f:
        samples = list(csv.DictReader(f, delimiter="\t", quotechar='"'))
    second = [r for r in samples if r["ts"] == "2.0"]
    assert len(second) == 1
    assert second[0]["tenant"] == "uid222222"
    assert second[0]["proc"] == "worker"


def test_process_samples_hostile_command_uses_canonical_tsv(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("boot-hostile\n")
    hostile = 'bad\t"quote"\nline\\N'
    _make_proc(
        proc, 80, hostile, 1, 411, uid=333333,
        cmdline=b"hostile\0--arg\0", exe="/opt/hostile",
    )
    _make_proc(
        proc, 81, r"\N", 1, 412, uid=333333,
        cmdline=b"literal-null-token\0", exe="/opt/literal-null-token",
    )

    monkeypatch.setenv("PROC_ROOT", str(proc))
    monkeypatch.setenv("CLK_TCK_OVERRIDE", "100")
    monkeypatch.setenv("DFX_UID_OVERRIDE", "333333")
    monkeypatch.setenv("DFX_BOOTID_OVERRIDE", "boot-hostile")
    monkeypatch.setenv("DFX_TABLE_FORMAT", "tsv")
    c = _load_collector(Path(__file__).resolve().parents[1] / "collector.py")

    out = tmp_path / "bundle"
    assert c.main([
        "-o", str(out), "--run-id", "r-hostile",
        "--interval", "0.01", "--nsamples", "1",
    ]) == 0

    raw = (out / "process_samples.tsv").read_text(encoding="utf-8")
    assert '"bad\t""quote""\nline\\N"' in raw

    with (out / "process_samples.tsv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quotechar='"'))
    by_proc = {row["proc"]: row for row in rows}
    assert hostile in by_proc
    assert r"\N" in by_proc
    assert len(by_proc[hostile]) == len(c.PROC_HEADER.split(","))
    assert len(by_proc[r"\N"]) == len(c.PROC_HEADER.split(","))
