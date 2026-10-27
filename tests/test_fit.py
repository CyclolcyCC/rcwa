"""Tests for GIXRF fitting, data loading, divergence convolution, and geometry reconstruction."""
from pathlib import Path
import numpy as np
import pytest

import xsw
from xsw.fit import (
    AndrleBParams,
    ExperimentData,
    build_andrle_b_sample,
    compute_optimal_scale,
    compute_residuals,
    convolve_divergence,
    divergence_matrix,
    evaluate_forward,
    load_andrle_data,
)


def test_load_andrle_data():
    data = load_andrle_data()
    assert isinstance(data, ExperimentData)
    assert data.num_angles == 180
    assert data.angles_deg[0] > 0
    assert data.angles_deg[-1] < 20
    assert len(data.exp_N) == 180
    assert len(data.exp_O) == 180
    assert len(data.sigma_N) == 180
    assert len(data.sigma_O) == 180
    assert np.all(data.exp_N > 0)
    assert np.all(data.sigma_N > 0)
    assert not data.has_carbon

    # Subsampling
    sub = data.subsample(step=10)
    assert sub.num_angles == 18
    assert np.allclose(sub.angles_deg, data.angles_deg[::10])


def test_load_andrle_data_with_carbon():
    npz_path = Path("examples/output/andrle_fig4_37_experimental_data.npz")
    if npz_path.exists():
        data = load_andrle_data(npz_path)
        assert data.num_angles == 178
        assert data.has_carbon
        assert data.exp_C is not None
        assert data.sigma_C is not None
        assert len(data.exp_C) == 178
        assert np.all(data.exp_C > 0)


def test_divergence_convolution():
    angles = np.linspace(0.5, 15.0, 50)
    # 1. Zero divergence returns exact identity
    y = np.sin(angles) + 2.0
    y_conv0 = convolve_divergence(angles, y, sigma_deg=0.0)
    assert np.allclose(y, y_conv0)

    # 2. Total flux conservation on constant signal
    flat = np.full_like(angles, 5.0)
    flat_conv = convolve_divergence(angles, flat, sigma_deg=0.01)
    assert np.allclose(flat, flat_conv, rtol=1e-10)

    # 3. Smoothing property on delta spike
    spike = np.zeros_like(angles)
    spike[25] = 10.0
    spike_conv = convolve_divergence(angles, spike, sigma_deg=0.05)
    assert spike_conv[25] < 10.0
    assert spike_conv[24] > 0.0
    assert spike_conv[26] > 0.0


def test_build_andrle_sample():
    params = AndrleBParams(h=93.0, w=44.9, swa=82.5, tt=2.47, tg=3.67,
                           sigma_sw=1.5, sigma_vert=1.0, tc=2.5)
    struct, mat_n, mat_o, mat_c = build_andrle_b_sample(params, num_slices=10)

    assert isinstance(struct, xsw.Structure)
    assert len(struct.layers) >= 10
    assert mat_n.fluorescent
    assert mat_o.fluorescent
    assert mat_c.fluorescent
    assert mat_c.name == "C2O"

    # Validation errors
    with pytest.raises(ValueError, match="Groove stack tg \\+ tc"):
        build_andrle_b_sample(AndrleBParams(h=50.0, tg=45.0, tc=10.0))

    with pytest.raises(ValueError, match="Sidewall angle"):
        build_andrle_b_sample(AndrleBParams(swa=95.0))


def test_compute_optimal_scale():
    exp = np.array([10.0, 20.0, 30.0])
    model = np.array([2.0, 4.0, 6.0])
    sigma = np.array([1.0, 1.0, 1.0])

    c = compute_optimal_scale(exp, model, sigma)
    assert np.isclose(c, 5.0)

    # Check that analytical formula strictly minimizes sum-of-squares
    def loss(scale):
        return np.sum(((exp - scale * model) / sigma) ** 2)

    assert loss(c) < loss(c + 0.01)
    assert loss(c) < loss(c - 0.01)


def test_forward_evaluation():
    data = load_andrle_data().subsample(step=36)  # 5 angles for fast testing
    params = AndrleBParams(h=93.0, w=44.9, swa=82.5, tt=2.47, tg=3.67,
                           sigma_sw=1.5, sigma_vert=1.0, tc=2.5)

    result = evaluate_forward(params, data, num_slices=6, max_order=7)

    assert len(result.scaled_N) == 5
    assert len(result.scaled_O) == 5
    assert result.scale_N > 0
    assert result.scale_O > 0
    assert result.total_chi2 > 0
    assert np.isclose(result.total_chi2, result.chi2_N + result.chi2_O)

    # Check compute_residuals compatibility with scipy least_squares
    res = compute_residuals(params.as_array(), data, num_slices=6, max_order=7)
    assert len(res) == 10  # 5 angles * 2 elements
    assert np.isclose(np.sum(res ** 2), result.total_chi2)


def test_forward_evaluation_with_carbon():
    # 5 angles with carbon channel
    angles = np.linspace(1.0, 5.0, 5)
    data = ExperimentData(
        angles_deg=angles,
        exp_N=np.array([100.0, 150.0, 200.0, 220.0, 240.0]),
        exp_O=np.array([120.0, 180.0, 250.0, 270.0, 290.0]),
        sigma_N=np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
        sigma_O=np.array([6.0, 6.0, 6.0, 6.0, 6.0]),
        exp_C=np.array([80.0, 110.0, 140.0, 160.0, 180.0]),
        sigma_C=np.array([4.0, 4.0, 4.0, 4.0, 4.0]),
    )
    assert data.has_carbon

    params = AndrleBParams(h=93.0, w=44.9, swa=82.5, tt=2.47, tg=3.67,
                           sigma_sw=1.5, sigma_vert=1.0, tc=2.5)

    result = evaluate_forward(params, data, num_slices=6, max_order=7)

    assert result.scaled_C is not None
    assert result.scale_C is not None and result.scale_C > 0
    assert result.chi2_C is not None and result.chi2_C > 0
    assert np.isclose(result.total_chi2, result.chi2_N + result.chi2_O + result.chi2_C)

    res = compute_residuals(params.as_array(), data, num_slices=6, max_order=7)
    assert len(res) == 15  # 5 angles * 3 elements
    assert np.isclose(np.sum(res ** 2), result.total_chi2)


def test_fit_result_save_and_load(tmp_path):
    angles = np.linspace(1.0, 5.0, 5)
    data = ExperimentData(
        angles_deg=angles,
        exp_N=np.array([100.0, 150.0, 200.0, 220.0, 240.0]),
        exp_O=np.array([120.0, 180.0, 250.0, 270.0, 290.0]),
        sigma_N=np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
        sigma_O=np.array([6.0, 6.0, 6.0, 6.0, 6.0]),
        exp_C=np.array([80.0, 110.0, 140.0, 160.0, 180.0]),
        sigma_C=np.array([4.0, 4.0, 4.0, 4.0, 4.0]),
    )
    p0 = AndrleBParams(h=93.0, w=44.9, swa=82.5, tt=2.47, tg=3.67, tc=2.5,
                           sigma_sw=1.5, sigma_vert=1.0)
    from xsw.fit import FitResult
    fwd = evaluate_forward(p0, data, num_slices=6, max_order=7)

    fit_res = FitResult(
        params=p0,
        param_errors=np.array([1.0, 0.5, 0.2, 0.1, 0.1, 0.2, 0.3]),
        covariance=np.eye(7),
        forward=fwd,
        chi2=fwd.total_chi2,
        reduced_chi2=fwd.reduced_chi2,
        success=True,
        message="Optimization terminated successfully.",
        nfev=12,
    )

    save_path = tmp_path / "saved_fit.npz"
    fit_res.save(save_path)

    assert save_path.exists()
    assert (tmp_path / "saved_fit.txt").exists()

    loaded = FitResult.load(save_path)
    assert np.isclose(loaded.params.h, fit_res.params.h)
    assert np.isclose(loaded.params.w, fit_res.params.w)
    assert np.isclose(loaded.params.tc, fit_res.params.tc)
    assert np.allclose(loaded.param_errors, fit_res.param_errors)
    assert np.allclose(loaded.covariance, fit_res.covariance)
    assert np.isclose(loaded.chi2, fit_res.chi2)
    assert np.allclose(loaded.forward.scaled_N, fit_res.forward.scaled_N)
    assert np.allclose(loaded.forward.scaled_O, fit_res.forward.scaled_O)
    assert np.allclose(loaded.forward.scaled_C, fit_res.forward.scaled_C)
    assert loaded.success == fit_res.success
    assert loaded.message == fit_res.message

