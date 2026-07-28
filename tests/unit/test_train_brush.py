import os
import subprocess
import sys

import pytest

import train_brush as tb

DEFAULT_TOTAL_STEPS = 75000            # matches train_brush.py's own --total_steps default
INTERMEDIATE_STEPS = 10000             # an in-progress checkpoint, well below DEFAULT_TOTAL_STEPS
FINAL_EXPORT_NAME = f"splat_{DEFAULT_TOTAL_STEPS}.ply"
INTERMEDIATE_EXPORT_NAME = f"splat_{INTERMEDIATE_STEPS}.ply"

START_TIME = 1000.0                    # fixed "now" -- see fixed_time()
FRESH_MTIME = START_TIME + 1.0         # newer than START_TIME -- counts as this run's own export
STALE_MTIME = START_TIME - 500.0       # older than START_TIME -- a leftover from a prior run


class FakeProcess:
    """Stand-in for subprocess.Popen. poll_results is consumed in order;
    once exhausted, further poll() calls repeat the last value."""
    def __init__(self, poll_results=(None,), returncode=0):
        self._poll_results = list(poll_results)
        self._last = None
        self.returncode = returncode
        self.terminate_called = False
        self.kill_called = False
        self.wait_calls = []
        self._timeout_on_next_wait = False

    def poll(self):
        if self._poll_results:
            self._last = self._poll_results.pop(0)
        if self._last is not None:
            self.returncode = self._last
        return self._last

    def terminate(self):
        self.terminate_called = True

    def kill(self):
        self.kill_called = True

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self._timeout_on_next_wait:
            self._timeout_on_next_wait = False
            raise subprocess.TimeoutExpired(cmd="brush_app", timeout=timeout)
        return self.returncode


def base_argv(data, brush_app, export_path, extra=None):
    argv = [
        "prog",
        "--data", str(data),
        "--brush_app", str(brush_app),
        "--export_path", str(export_path),
    ]
    return argv + (extra or [])


def no_sleep(monkeypatch):
    monkeypatch.setattr(tb.time, "sleep", lambda s: None)


def fixed_time(monkeypatch, t):
    monkeypatch.setattr(tb.time, "time", lambda: t)


# --------------------------------------------------------------------------
# main() -- argument validation, cmd/env construction (subprocess.Popen
# mocked throughout; brush_app itself is never runnable in this environment)
# --------------------------------------------------------------------------

def test_main_errors_when_export_name_missing_iter_placeholder(tmp_path, monkeypatch, capsys):
    # Regression test: --export_name without a literal "{iter}" placeholder
    # produced a regex with no capturing group, and the first time a
    # matching file appeared, final_export_done's m.group(1) raised a raw
    # IndexError -- crashing training right at the finish line. Now
    # rejected up front with a clean error instead.
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    monkeypatch.setattr(sys, "argv", base_argv(
        tmp_path / "data", brush_app, tmp_path / "export", ["--export_name", "my_export.ply"],
    ))
    with pytest.raises(SystemExit) as exc_info:
        tb.main()
    assert exc_info.value.code == 1
    assert "must contain the literal" in capsys.readouterr().out


def test_main_errors_when_brush_app_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", base_argv(tmp_path / "data", tmp_path / "nope", tmp_path / "export"))
    with pytest.raises(SystemExit) as exc_info:
        tb.main()
    assert exc_info.value.code == 1
    assert "brush_app not found" in capsys.readouterr().out


def test_main_creates_export_path(tmp_path, monkeypatch):
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    proc = FakeProcess(poll_results=[0])
    monkeypatch.setattr(tb.subprocess, "Popen", lambda cmd, env=None: proc)
    export_path = tmp_path / "nested" / "export"
    monkeypatch.setattr(sys, "argv", base_argv(tmp_path / "data", brush_app, export_path))
    with pytest.raises(SystemExit):
        tb.main()
    assert export_path.is_dir()


def test_main_default_cmd_construction_includes_with_viewer_and_excludes_alpha_weight(tmp_path, monkeypatch):
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    calls = []
    proc = FakeProcess(poll_results=[0])
    def fake_popen(cmd, env=None):
        calls.append((cmd, env))
        return proc
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)
    data, export_path = tmp_path / "data", tmp_path / "export"
    monkeypatch.setattr(sys, "argv", base_argv(data, brush_app, export_path))
    with pytest.raises(SystemExit) as exc_info:
        tb.main()
    assert exc_info.value.code == 0

    cmd, env = calls[0]
    assert cmd[0] == str(brush_app)
    assert cmd[1] == str(data)
    assert cmd[2] == "--with-viewer"  # inserted at index 2 by default
    assert "--total-steps" in cmd and cmd[cmd.index("--total-steps") + 1] == str(DEFAULT_TOTAL_STEPS)
    assert "--lr-coeffs-sh-scale" in cmd and cmd[cmd.index("--lr-coeffs-sh-scale") + 1] == "80"
    assert "--export-path" in cmd and cmd[cmd.index("--export-path") + 1] == str(export_path)
    assert "--export-name" in cmd and cmd[cmd.index("--export-name") + 1] == "splat_{iter}.ply"
    assert "--match-alpha-weight" not in cmd  # default None -- must not be forced onto the cmd


def test_main_no_viewer_omits_with_viewer_flag_and_does_not_force_display(tmp_path, monkeypatch):
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    calls = []
    proc = FakeProcess(poll_results=[0])
    def fake_popen(cmd, env=None):
        calls.append((cmd, env))
        return proc
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(sys, "argv", base_argv(
        tmp_path / "data", brush_app, tmp_path / "export", ["--no_viewer"],
    ))
    with pytest.raises(SystemExit):
        tb.main()
    cmd, env = calls[0]
    assert "--with-viewer" not in cmd
    assert "DISPLAY" not in env  # with_viewer False -- must not force a DISPLAY value


def test_main_with_viewer_sets_display_env_from_flag(tmp_path, monkeypatch):
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    calls = []
    proc = FakeProcess(poll_results=[0])
    def fake_popen(cmd, env=None):
        calls.append(env)
        return proc
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sys, "argv", base_argv(
        tmp_path / "data", brush_app, tmp_path / "export", ["--display", ":7"],
    ))
    with pytest.raises(SystemExit):
        tb.main()
    assert calls[0]["DISPLAY"] == ":7"


def test_main_match_alpha_weight_added_when_given(tmp_path, monkeypatch):
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    calls = []
    proc = FakeProcess(poll_results=[0])
    def fake_popen(cmd, env=None):
        calls.append(cmd)
        return proc
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sys, "argv", base_argv(
        tmp_path / "data", brush_app, tmp_path / "export", ["--match_alpha_weight", "0.25"],
    ))
    with pytest.raises(SystemExit):
        tb.main()
    cmd = calls[0]
    assert cmd[cmd.index("--match-alpha-weight") + 1] == "0.25"


def test_main_eval_flags_omitted_by_default(tmp_path, monkeypatch):
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    calls = []
    proc = FakeProcess(poll_results=[0])
    def fake_popen(cmd, env=None):
        calls.append(cmd)
        return proc
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sys, "argv", base_argv(tmp_path / "data", brush_app, tmp_path / "export", []))
    with pytest.raises(SystemExit):
        tb.main()
    cmd = calls[0]
    assert "--eval-split-every" not in cmd
    assert "--eval-save-to-disk" not in cmd


def test_main_eval_split_every_and_save_to_disk_added_when_given(tmp_path, monkeypatch):
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    calls = []
    proc = FakeProcess(poll_results=[0])
    def fake_popen(cmd, env=None):
        calls.append(cmd)
        return proc
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sys, "argv", base_argv(
        tmp_path / "data", brush_app, tmp_path / "export",
        ["--eval_split_every", "8", "--eval_save_to_disk"],
    ))
    with pytest.raises(SystemExit):
        tb.main()
    cmd = calls[0]
    assert cmd[cmd.index("--eval-split-every") + 1] == "8"
    assert "--eval-save-to-disk" in cmd


def test_main_explicit_total_steps_overrides_default(tmp_path, monkeypatch):
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    calls = []
    proc = FakeProcess(poll_results=[0])
    def fake_popen(cmd, env=None):
        calls.append(cmd)
        return proc
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sys, "argv", base_argv(
        tmp_path / "data", brush_app, tmp_path / "export", ["--total_steps", "500"],
    ))
    with pytest.raises(SystemExit):
        tb.main()
    cmd = calls[0]
    assert cmd[cmd.index("--total-steps") + 1] == "500"


def test_main_always_sets_gpu_env_vars_and_default_xdg_runtime_dir(tmp_path, monkeypatch):
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    calls = []
    proc = FakeProcess(poll_results=[0])
    def fake_popen(cmd, env=None):
        calls.append(env)
        return proc
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", base_argv(tmp_path / "data", brush_app, tmp_path / "export"))
    with pytest.raises(SystemExit):
        tb.main()
    env = calls[0]
    assert env["__NV_PRIME_RENDER_OFFLOAD"] == "1"
    assert env["__GLX_VENDOR_LIBRARY_NAME"] == "nvidia"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["XDG_RUNTIME_DIR"] == f"/run/user/{os.getuid()}"


def test_main_preserves_existing_xdg_runtime_dir(tmp_path, monkeypatch):
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    calls = []
    proc = FakeProcess(poll_results=[0])
    def fake_popen(cmd, env=None):
        calls.append(env)
        return proc
    monkeypatch.setattr(tb.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/custom/runtime")
    monkeypatch.setattr(sys, "argv", base_argv(tmp_path / "data", brush_app, tmp_path / "export"))
    with pytest.raises(SystemExit):
        tb.main()
    assert calls[0]["XDG_RUNTIME_DIR"] == "/custom/runtime"


@pytest.mark.parametrize("missing_flag", ["--data", "--brush_app", "--export_path"])
def test_main_errors_when_required_arg_missing(tmp_path, monkeypatch, missing_flag):
    argv = base_argv(tmp_path / "data", tmp_path / "brush_app", tmp_path / "export")
    idx = argv.index(missing_flag)
    argv = argv[:idx] + argv[idx + 2:]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        tb.main()
    assert exc_info.value.code == 2


# --------------------------------------------------------------------------
# Polling/completion-detection logic: --with-viewer never exits on its own,
# so main() must detect the final export on disk and close the process
# itself, rather than blocking forever on process exit.
# --------------------------------------------------------------------------

def run_main(monkeypatch, tmp_path, brush_app, export_path, total_steps=DEFAULT_TOTAL_STEPS, export_name="splat_{iter}.ply"):
    monkeypatch.setattr(sys, "argv", base_argv(
        tmp_path / "data", brush_app, export_path,
        ["--total_steps", str(total_steps), "--export_name", export_name],
    ))


def test_main_terminates_process_when_final_export_appears_while_running(tmp_path, monkeypatch):
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    fixed_time(monkeypatch, START_TIME)
    export_path = tmp_path / "export"
    export_path.mkdir()
    final_file = export_path / FINAL_EXPORT_NAME
    final_file.write_text("done")
    os.utime(final_file, (FRESH_MTIME, FRESH_MTIME))

    proc = FakeProcess(poll_results=[None])  # still "running" when checked
    monkeypatch.setattr(tb.subprocess, "Popen", lambda cmd, env=None: proc)
    run_main(monkeypatch, tmp_path, brush_app, export_path)

    with pytest.raises(SystemExit) as exc_info:
        tb.main()
    assert exc_info.value.code == 0
    assert proc.terminate_called
    assert proc.wait_calls == [15]


def test_main_falls_back_to_kill_when_terminate_wait_times_out(tmp_path, monkeypatch):
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    fixed_time(monkeypatch, START_TIME)
    export_path = tmp_path / "export"
    export_path.mkdir()
    final_file = export_path / FINAL_EXPORT_NAME
    final_file.write_text("done")
    os.utime(final_file, (FRESH_MTIME, FRESH_MTIME))

    proc = FakeProcess(poll_results=[None])
    proc._timeout_on_next_wait = True
    monkeypatch.setattr(tb.subprocess, "Popen", lambda cmd, env=None: proc)
    run_main(monkeypatch, tmp_path, brush_app, export_path)

    with pytest.raises(SystemExit) as exc_info:
        tb.main()
    assert exc_info.value.code == 0
    assert proc.terminate_called
    assert proc.kill_called
    assert len(proc.wait_calls) == 2  # first (timed out), then the plain fallback wait()


def test_main_propagates_real_crash_code_when_no_export_ever_appears(tmp_path, monkeypatch):
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    fixed_time(monkeypatch, START_TIME)
    export_path = tmp_path / "export"
    export_path.mkdir()  # no export file at all

    proc = FakeProcess(poll_results=[7])  # exits immediately with a real crash code
    monkeypatch.setattr(tb.subprocess, "Popen", lambda cmd, env=None: proc)
    run_main(monkeypatch, tmp_path, brush_app, export_path)

    with pytest.raises(SystemExit) as exc_info:
        tb.main()
    assert exc_info.value.code == 7


def test_main_trusts_artifact_over_nonzero_exit_code(tmp_path, monkeypatch):
    # If the final export is on disk, treat the run as successful even if
    # the process happened to exit with a nonzero code afterward (e.g. a
    # viewer-close race) -- the artifact is the source of truth.
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    fixed_time(monkeypatch, START_TIME)
    export_path = tmp_path / "export"
    export_path.mkdir()
    final_file = export_path / FINAL_EXPORT_NAME
    final_file.write_text("done")
    os.utime(final_file, (FRESH_MTIME, FRESH_MTIME))

    proc = FakeProcess(poll_results=[3])  # exits immediately, nonzero, but export already exists
    monkeypatch.setattr(tb.subprocess, "Popen", lambda cmd, env=None: proc)
    run_main(monkeypatch, tmp_path, brush_app, export_path)

    with pytest.raises(SystemExit) as exc_info:
        tb.main()
    assert exc_info.value.code == 0


def test_main_stale_leftover_export_from_prior_run_does_not_count(tmp_path, monkeypatch):
    # Regression-style guard on the after_ts logic: a completed export
    # already sitting in --export_path from a PRIOR run must not make this
    # run look successful before it has actually trained anything.
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    fixed_time(monkeypatch, START_TIME)
    export_path = tmp_path / "export"
    export_path.mkdir()
    stale_file = export_path / FINAL_EXPORT_NAME
    stale_file.write_text("old")
    os.utime(stale_file, (STALE_MTIME, STALE_MTIME))

    proc = FakeProcess(poll_results=[9])  # exits immediately with a real crash code
    monkeypatch.setattr(tb.subprocess, "Popen", lambda cmd, env=None: proc)
    run_main(monkeypatch, tmp_path, brush_app, export_path)

    with pytest.raises(SystemExit) as exc_info:
        tb.main()
    assert exc_info.value.code == 9  # stale file ignored -- real exit code propagated


def test_main_intermediate_checkpoint_does_not_count_as_final(tmp_path, monkeypatch):
    # An intermediate export (step number < total_steps) matching the same
    # filename pattern must not be mistaken for the final checkpoint.
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    no_sleep(monkeypatch)
    fixed_time(monkeypatch, START_TIME)
    export_path = tmp_path / "export"
    export_path.mkdir()
    intermediate_file = export_path / INTERMEDIATE_EXPORT_NAME  # step < DEFAULT_TOTAL_STEPS
    intermediate_file.write_text("checkpoint")
    os.utime(intermediate_file, (FRESH_MTIME, FRESH_MTIME))

    proc = FakeProcess(poll_results=[4])  # exits immediately with a real crash code
    monkeypatch.setattr(tb.subprocess, "Popen", lambda cmd, env=None: proc)
    run_main(monkeypatch, tmp_path, brush_app, export_path)

    with pytest.raises(SystemExit) as exc_info:
        tb.main()
    assert exc_info.value.code == 4  # intermediate checkpoint ignored


def test_main_keyboard_interrupt_terminates_process_and_reraises(tmp_path, monkeypatch):
    brush_app = tmp_path / "brush_app"
    brush_app.write_text("")
    fixed_time(monkeypatch, START_TIME)

    def raising_sleep(s):
        raise KeyboardInterrupt
    monkeypatch.setattr(tb.time, "sleep", raising_sleep)

    export_path = tmp_path / "export"
    export_path.mkdir()  # no export -- loop must actually reach time.sleep()
    proc = FakeProcess(poll_results=[None])  # stays "running" so the loop body executes
    monkeypatch.setattr(tb.subprocess, "Popen", lambda cmd, env=None: proc)
    run_main(monkeypatch, tmp_path, brush_app, export_path)

    with pytest.raises(KeyboardInterrupt):
        tb.main()
    assert proc.terminate_called
    assert proc.wait_calls == [None]
