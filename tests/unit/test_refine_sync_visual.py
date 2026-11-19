# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""refine_sync_visual.py: a camera's sub-frame capture offset from how far moving points sit off their
epipolar lines, on synthetic cameras whose true offsets are known."""

import math

import numpy as np
import pytest

import refine_sync_visual as rsv

FPS = 30.0


def look_at(eye, target=(0.0, 0.0, 0.0), w=1600, h=1200, f=1400.0, label="x"):
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, -1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return rsv.Camera(label, K, R, -R @ eye, w, h)


def project(cam, X):
    p = (cam.R @ X.T).T + cam.t
    uv = (cam.K @ p.T).T
    return uv[:, :2] / uv[:, 2:3]


def scene(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.25, size=(n, 3))
    V = rng.normal(scale=1.5, size=(n, 3))  # metres per second: a moving body
    A = rng.normal(scale=2.0, size=(n, 3))
    return X, V, A


def observe(cam, X, V, A, t):
    """Image position, velocity (px/frame) and acceleration (px/frame^2) of the points at time t (seconds)."""
    pos = lambda s: X + V * s + 0.5 * A * s * s  # noqa: E731
    dt = 1.0 / FPS
    p0, pm, pp = project(cam, pos(t)), project(cam, pos(t - dt)), project(cam, pos(t + dt))
    return p0, (pp - pm) / 2.0, pp - 2 * p0 + pm


def measure(ca, cb, off_a, off_b, prior_frames=0.0, seed=0):
    X, V, A = scene(seed=seed)
    ua, va, aa = observe(ca, X, V, A, off_a)
    ub, vb, ab = observe(cb, X, V, A, off_b)
    return rsv.pair_phase(ca, cb, ua, ub, va, vb, aa, ab, np.ones(len(ua)), size_b=(cb.width, cb.height),
                          prior_frames=prior_frames)


def test_the_phase_of_a_pair_is_how_much_later_a_fired_than_b():
    ca = look_at([3.0, 0.2, 0.0], label="00")
    cb = look_at([3.0 * math.cos(0.7), 0.1, 3.0 * math.sin(0.7)], label="01")
    for off_a, off_b in ((0.0, 0.0), (0.004, -0.003), (-0.006, 0.005)):
        row = measure(ca, cb, off_a, off_b)
        assert row is not None
        assert row["phase_frames"] == pytest.approx((off_a - off_b) * FPS, abs=0.02), (off_a, off_b, row)
        assert row["se_frames"] < 0.05


def test_a_phase_beyond_half_a_frame_comes_back_around_the_audio_prior():
    """Each camera can be up to half a frame off its export, so a pair up to a frame; the phase alone
    wraps at half a frame, so the audio residuals pick the right turn."""
    ca = look_at([3.0, 0.2, 0.0], label="00")
    cb = look_at([3.0 * math.cos(0.7), 0.1, 3.0 * math.sin(0.7)], label="01")
    off_a, off_b = 0.4 / FPS, -0.35 / FPS  # 0.75 frame apart
    prior = (0.38 - (-0.33))  # what audio said, in frames
    row = measure(ca, cb, off_a, off_b, prior_frames=prior)
    assert row["phase_frames"] == pytest.approx(0.75, abs=0.03)


def test_the_graph_puts_every_camera_at_its_offset_with_the_reference_at_zero():
    true = {"00": 0.0, "01": 0.004, "02": -0.0055, "03": 0.002}
    edges = {}
    for a, b in (("00", "01"), ("01", "02"), ("02", "03"), ("00", "03"), ("00", "02")):
        edges[(a, b)] = {"phase_frames": (true[a] - true[b]) * FPS, "sigma_frames": 0.02}
    edges[("01", "03")] = {"phase_frames": 0.45, "sigma_frames": 0.02}  # a bad edge
    solved = rsv.solve_graph(edges, sorted(true), "00", FPS)
    for label, off in true.items():
        assert solved["cameras"][label]["offset_s"] == pytest.approx(off, abs=2e-4)
    assert solved["dropped_pairs"] == ["01-03"]
    assert solved["cameras"]["00"]["status"] == "reference"
    assert all(solved["cameras"][l]["status"] == "ok" for l in ("01", "02", "03"))
    assert rsv.verdict(solved["cameras"], FPS) == "good"


def test_a_camera_no_pair_reaches_is_not_placed_and_one_pair_is_weak():
    edges = {("00", "01"): {"phase_frames": 0.1, "sigma_frames": 0.02}}
    solved = rsv.solve_graph(edges, ["00", "01", "02"], "00", FPS)
    assert solved["cameras"]["02"] == {"offset_s": None, "sigma_s": None, "pairs": 0, "status": "not placed"}
    assert solved["cameras"]["01"]["status"] == "weak"
    assert rsv.verdict(solved["cameras"], FPS) == "failed"


def test_frames_that_disagree_give_no_phase_for_the_pair():
    agree = [{"phase_frames": 0.2 + d, "se_frames": 0.02} for d in (-0.01, 0.0, 0.01)]
    assert rsv.combine_frames(agree, 0.0)["phase_frames"] == pytest.approx(0.2, abs=0.01)
    disagree = [{"phase_frames": p, "se_frames": 0.02} for p in (0.2, -0.2, 0.2, -0.2)]
    assert rsv.combine_frames(disagree, 0.0) is None
    # a phase near the wrap is combined on the circle, not averaged to the middle
    edge = [{"phase_frames": p, "se_frames": 0.02} for p in (0.49, -0.49, 0.48)]
    assert abs(rsv.combine_frames(edge, 0.0)["phase_frames"]) > 0.45


def rotated(cam, degrees, axis=(0.0, 1.0, 0.0)):
    """The same camera turned a little about its centre: a pose error."""
    import cv2

    R_err, _ = cv2.Rodrigues(np.asarray(axis, float) * math.radians(degrees))
    centre = cam.center
    R = R_err @ cam.R
    return rsv.Camera(cam.label, cam.K, R, -R @ centre, cam.width, cam.height)


def test_a_pose_error_is_taken_out_by_the_fit_over_frames():
    """With camera B's pose 0.3 degrees off (several pixels of epipolar error), one frame's phase is wrong;
    fitting every frame together with a per-cell pose term recovers it, because the motion changes from frame
    to frame and the pose error does not."""
    ca = look_at([3.0, 0.2, 0.0], label="00")
    cb = look_at([3.0 * math.cos(0.7), 0.1, 3.0 * math.sin(0.7)], label="01")
    cb_seen = rotated(cb, 1.0)
    off_a, off_b = 0.003, -0.004
    want = (off_a - off_b) * FPS
    rows = []
    rng = np.random.default_rng(5)
    for frame in range(8):
        X, V, A = scene(seed=10 + frame)
        # a body moving one way in each frame (like a limb), a different way the next
        direction = rng.normal(size=3)
        V = 2.0 * direction / np.linalg.norm(direction) + 0.2 * V
        ua, va, aa = observe(ca, X, V, A, off_a)
        ub, vb, ab = observe(cb, X, V, A, off_b)
        row = rsv.pair_phase(ca, cb_seen, ua, ub, va, vb, aa, ab, np.ones(len(ua)), size_b=(cb.width, cb.height))
        assert row is not None
        rows.append(row)
    single = [r["phase_frames"] for r in rows if r["phase_frames"] is not None]
    assert not single or max(abs(p - want) for p in single) > 0.05  # one frame alone is thrown by the pose error
    fit = rsv.fit_pair(rows)
    assert fit["phase_frames"] == pytest.approx(want, abs=0.03), (fit, want)
    assert fit["sigma_frames"] < 0.05 and fit["frames"] == 8


def test_the_fit_needs_several_frames():
    ca = look_at([3.0, 0.2, 0.0], label="00")
    cb = look_at([3.0 * math.cos(0.7), 0.1, 3.0 * math.sin(0.7)], label="01")
    X, V, A = scene()
    ua, va, aa = observe(ca, X, V, A, 0.0)
    ub, vb, ab = observe(cb, X, V, A, 0.0)
    row = rsv.pair_phase(ca, cb, ua, ub, va, vb, aa, ab, np.ones(len(ua)), size_b=(cb.width, cb.height))
    assert rsv.fit_pair([row, row]) is None


def test_tracking_gives_each_points_velocity_in_pixels_per_frame(tmp_path):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    base = (rng.random((240, 320)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    frames = []
    for i in range(5):  # the texture moves 3 px right and 1 px down per frame
        M = np.float32([[1, 0, 3.0 * i], [0, 1, 1.0 * i]])
        path = tmp_path / f"{i}.png"
        cv2.imwrite(str(path), cv2.warpAffine(base, M, (320, 240), borderMode=cv2.BORDER_REFLECT))
        frames.append(path)
    pts = np.array([[100.0, 100.0], [160.0, 120.0], [200.0, 90.0]])
    vel, acc, ok = rsv.track(frames, 2, pts, half=2)
    assert ok.all()
    assert np.allclose(vel, [[3.0, 1.0]] * 3, atol=0.15)
    assert np.allclose(acc, 0.0, atol=0.3)


def test_cameras_pair_with_their_nearest_views():
    cams = {f"{i:02d}": look_at([3 * math.cos(i * 0.5), 0.0, 3 * math.sin(i * 0.5)], label=f"{i:02d}") for i in range(6)}
    pairs = rsv.choose_pairs(cams, 2, 75.0)
    assert ("00", "01") in pairs and ("00", "05") not in pairs
    assert all(a < b for a, b in pairs)
