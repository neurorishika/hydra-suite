import pytest

from hydra_suite.runtime import cpu_budget as cb


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_usable_cores_prefers_affinity_over_cpu_count(tmp_path):
    assert (
        cb.usable_cores(
            affinity={0, 1, 2},
            cpu_count=64,
            cgroup_root=str(tmp_path / "none"),
            proc_cgroup=str(tmp_path / "none"),
        )
        == 3
    )


def test_usable_cores_falls_back_to_cpu_count(tmp_path):
    assert (
        cb.usable_cores(
            affinity=None,
            cpu_count=8,
            cgroup_root=str(tmp_path / "x"),
            proc_cgroup=str(tmp_path / "x"),
        )
        == 8
    )


def test_usable_cores_cgroup_quota_on_ancestor(tmp_path):
    root = tmp_path / "cg"
    proc = tmp_path / "proc_cgroup"
    _write(proc, "0::/user.slice/user-1.slice/session-3.scope\n")
    _write(root / "user.slice" / "cpu.max", "max 100000\n")
    _write(root / "user.slice" / "user-1.slice" / "cpu.max", "250000 100000\n")
    # session scope has no cpu.max file; ancestor quota 2.5 -> ceil = 3
    assert (
        cb.usable_cores(
            affinity=set(range(16)),
            cpu_count=16,
            cgroup_root=str(root),
            proc_cgroup=str(proc),
        )
        == 3
    )


def test_usable_cores_ignores_garbage_cgroup(tmp_path):
    root = tmp_path / "cg"
    proc = tmp_path / "proc_cgroup"
    _write(proc, "0::/a\n")
    _write(root / "a" / "cpu.max", "garbage\n")
    assert (
        cb.usable_cores(
            affinity=set(range(4)),
            cpu_count=4,
            cgroup_root=str(root),
            proc_cgroup=str(proc),
        )
        == 4
    )


def test_usable_cores_never_below_one(tmp_path):
    assert (
        cb.usable_cores(
            affinity=set(),
            cpu_count=None,
            cgroup_root=str(tmp_path),
            proc_cgroup=str(tmp_path / "x"),
        )
        == 1
    )


def test_resolve_budget_auto_env_flag_and_clamp():
    assert cb.resolve_cpu_budget(None, env={}, usable=12) == cb.CpuBudget(
        12, 12, "auto"
    )
    b = cb.resolve_cpu_budget(None, env={"HYDRA_CPU_BUDGET": "6"}, usable=12)
    assert (b.cores, b.source) == (6, "env")
    b = cb.resolve_cpu_budget(4, env={"HYDRA_CPU_BUDGET": "6"}, usable=12)
    assert (b.cores, b.source) == (4, "flag")
    b = cb.resolve_cpu_budget(40, env={}, usable=12)
    assert b.cores == 12 and b.warnings  # can only lower


@pytest.mark.parametrize("bad", ["abc", "0", "-3", ""])
def test_resolve_budget_invalid_env_is_ignored_with_warning(bad):
    b = cb.resolve_cpu_budget(None, env={"HYDRA_CPU_BUDGET": bad}, usable=8)
    assert b.cores == 8 and b.source == "auto"
    assert bool(b.warnings) == (bad != "")


@pytest.mark.parametrize(
    "jobs,n_videos,cores,gpu_bound,tpj,expected",
    [
        (4, 10, 16, False, None, (4, 4, False)),
        (16, 20, 12, False, None, (6, 2, True)),  # CPU floor 2 -> max 6 jobs
        (16, 20, 12, True, None, (12, 1, True)),  # GPU floor 1 -> max 12 jobs
        (8, 2, 32, True, None, (8, 16, False)),  # divide by active videos
        (1, 1, 12, False, None, (1, 12, False)),
        (16, 20, 12, False, 3, (16, 3, False)),  # explicit override: no clamp
    ],
)
def test_plan_cpu_table(jobs, n_videos, cores, gpu_bound, tpj, expected):
    plan = cb.plan_cpu(
        jobs=jobs,
        n_videos=n_videos,
        budget=cb.CpuBudget(cores, cores, "auto"),
        gpu_bound=gpu_bound,
        threads_per_job=tpj,
    )
    assert (plan.jobs, plan.threads_per_job, plan.clamped) == expected
    assert bool(plan.warnings) == (
        plan.clamped or (tpj is not None and jobs * tpj > cores)
    )


def test_single_run_cpu_cap_precedence():
    env = {"HYDRA_CPU_THREADS": "3", "HYDRA_CPU_BUDGET": "8"}
    assert cb.single_run_cpu_cap(6, env=env, usable=16) == 3
    assert cb.single_run_cpu_cap(6, env={"HYDRA_CPU_BUDGET": "8"}, usable=16) == 6
    assert cb.single_run_cpu_cap(None, env={"HYDRA_CPU_BUDGET": "8"}, usable=16) == 8
    assert cb.single_run_cpu_cap(None, env={}, usable=16) is None
    assert (
        cb.single_run_cpu_cap(None, env={"HYDRA_CPU_THREADS": "junk"}, usable=16)
        is None
    )


def test_single_run_explicit_budget_can_only_lower_inherited_threads():
    """An explicit budget is a ceiling under HYDRA_CPU_THREADS, never ignored."""
    inherited = {"HYDRA_CPU_THREADS": "16"}
    assert cb.single_run_cpu_cap(4, env=inherited, usable=32) == 4
    assert cb.single_run_cpu_cap(24, env=inherited, usable=32) == 16
    # Invalid / absent explicit budget -> the inherited cap stands.
    assert cb.single_run_cpu_cap(None, env=inherited, usable=32) == 16
    assert cb.single_run_cpu_cap(-3, env=inherited, usable=32) == 16
    # HYDRA_CPU_BUDGET never overrides the inherited per-child cap.
    env = {"HYDRA_CPU_THREADS": "16", "HYDRA_CPU_BUDGET": "4"}
    assert cb.single_run_cpu_cap(None, env=env, usable=32) == 16


def test_negative_request_is_ignored_with_warning():
    b = cb.resolve_cpu_budget(-3, env={}, usable=8)
    assert (b.cores, b.source) == (8, "auto") and b.warnings
    assert cb.single_run_cpu_cap(-3, env={}, usable=16) is None
    assert cb.single_run_cpu_cap(-3, env={"HYDRA_CPU_BUDGET": "4"}, usable=16) == 4


def test_usable_cores_container_root_cgroup(tmp_path):
    root = tmp_path / "cg"
    proc = tmp_path / "proc_cgroup"
    _write(proc, "0::/\n")
    _write(root / "cpu.max", "300000 100000\n")
    assert (
        cb.usable_cores(
            affinity=set(range(16)),
            cpu_count=16,
            cgroup_root=str(root),
            proc_cgroup=str(proc),
        )
        == 3
    )


def test_capped_child_env_caps_thread_vars_and_strips_budget():
    base = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "16", "HYDRA_CPU_BUDGET": "32"}
    env = cb.capped_child_env(base, 4)
    assert env["HYDRA_CPU_THREADS"] == "4" and "HYDRA_CPU_BUDGET" not in env
    assert env["OMP_NUM_THREADS"] == "1" and env["MKL_NUM_THREADS"] == "4"
    assert all(env[v] for v in cb.THREAD_ENV_VARS)
    assert base["MKL_NUM_THREADS"] == "16"  # input untouched
