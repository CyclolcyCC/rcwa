import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import simpson

import xsw


@pytest.mark.parametrize('pol', ['s', 'p'])
@pytest.mark.parametrize('alpha', [.08, .24])
@pytest.mark.parametrize('loss', [0., .001])
def test_single_interface_complex_nevot_croce_amplitudes(pol, alpha, loss):
    substrate = xsw.Material(-.015 + 1j*loss)
    sigma, wavelength = 1.3, 2.4
    settings = dict(wavelength=wavelength, alpha=alpha, phi=.37,
                    polarization=pol, max_order=2)
    sharp = xsw.solve(xsw.Structure(100, substrate=substrate), **settings)
    rough = xsw.solve(xsw.Structure(100, substrate=substrate,
                                    substrate_sigma=sigma), **settings)
    kz0, kz1 = 2*np.pi/wavelength * np.sqrt(
        np.array([1, substrate.epsilon])-np.cos(alpha)**2+0j)
    reflection_factor = np.exp(-2*kz0*kz1*sigma**2)
    transmission_factor = np.exp(.5*(kz0-kz1)**2*sigma**2)
    assert_allclose(rough.coefficients[0], sharp.coefficients[0]*reflection_factor,
                    rtol=2e-12, atol=2e-14)
    assert_allclose(rough.coefficients[-1], sharp.coefficients[-1]*transmission_factor,
                    rtol=2e-12, atol=2e-14)
    assert_allclose(rough.reflectance, sharp.reflectance*abs(reflection_factor)**2,
                    atol=2e-13)
    assert rough.boundary_residual < 1e-12


@pytest.mark.parametrize('pol', ['s', 'p'])
def test_rough_multilayer_against_scalar_transfer_matrices(pol):
    # Independent 2x2 Fresnel/NC calculation, including all three interfaces.
    eps = np.array([1, .995+.0007j, .992+.0003j, .99+.001j])
    thicknesses, sigmas = [17., 23.], [.8, 1.2, .5]
    wavelength, alpha = 2.4, .16
    model = xsw.Structure(100, substrate=xsw.Material(eps[-1]-1),
                          substrate_sigma=sigmas[-1])
    for epsilon, d, sigma in zip(eps[1:-1], thicknesses, sigmas):
        model.add_layer(xsw.Layer(d, background=xsw.Material(epsilon-1), sigma=sigma))
    sol = xsw.solve(model, wavelength=wavelength, alpha=alpha, phi=.43,
                    polarization=pol, max_order=2)
    kz = 2*np.pi/wavelength*np.sqrt(eps-np.cos(alpha)**2)
    admittance = kz if pol == 's' else eps/kz
    total = np.eye(2, dtype=complex)
    for j, sigma in enumerate(sigmas):
        r = (admittance[j]-admittance[j+1])/(admittance[j]+admittance[j+1])
        t = (1+r)*np.exp(.5*(kz[j]-kz[j+1])**2*sigma**2)
        r *= np.exp(-2*kz[j]*kz[j+1]*sigma**2)
        total = total @ (np.array([[1,r], [r,1]])/t)
        if j < len(thicknesses):
            phase = np.exp(1j*kz[j+1]*thicknesses[j])
            total = total @ np.diag([1/phase, phase])
    r, t = total[1,0]/total[0,0], 1/total[0,0]
    assert_allclose(sol.reflectance, abs(r)**2, atol=2e-13)
    assert_allclose(sol.transmittance, abs(t)**2*admittance[-1].real/admittance[0].real,
                    atol=2e-12)


@pytest.mark.parametrize('pol', ['s', 'p'])
def test_zero_interface_sigma_recovers_default(pol):
    material = xsw.Material(-.01+.002j, fluorescent=True)
    default = xsw.Structure(100).add_layer(xsw.Layer(20).add_box(-20,20,material,sigma=2))
    explicit = xsw.Structure(100, substrate_sigma=0).add_layer(
        xsw.Layer(20, sigma=0).add_box(-20,20,material,sigma=2))
    settings = dict(wavelength=2.4, alpha=.12, phi=.3, polarization=pol, max_order=3)
    a, b = xsw.solve(default, **settings), xsw.solve(explicit, **settings)
    assert default.substrate_sigma == default.layers[0].sigma == 0.0
    assert_array_equal(a.coefficients, b.coefficients)
    assert a.fluorescence() == b.fluorescence()


@pytest.mark.parametrize('pol', ['s', 'p'])
def test_rough_grating_fluorescence_and_subdivision(pol):
    material = xsw.Material(-.01+.002j, fluorescent=True)
    substrate = xsw.Material(-.015+.001j)
    model = xsw.Structure(100, substrate=substrate, substrate_sigma=.7)
    model.add_layer(xsw.Layer(20, sigma=1.1).add_box(-20,20,material,sigma=2))
    settings = dict(wavelength=2.4, alpha=.13, phi=.3, polarization=pol, max_order=4)
    sol = xsw.solve(model, **settings)
    x, z = np.linspace(-20+1e-10,20-1e-10,301), np.linspace(1e-10,20-1e-10,301)
    mu = .02
    numeric = simpson(simpson(sol.intensity(x,z)*np.exp(-mu*z[:,None]),x=x,axis=1),x=z)
    assert_allclose(sol.fluorescence(material,attenuation=mu), numeric, rtol=3e-8)
    assert sol.boundary_residual < 1e-12
    split = xsw.Structure(100, substrate=substrate, substrate_sigma=.7)
    split.add_layer(xsw.Layer(7, sigma=1.1).add_box(-20,20,material,sigma=2))
    split.add_layer(xsw.Layer(13).add_box(-20,20,material,sigma=2))
    divided = xsw.solve(split, **settings)
    assert_allclose(divided.intensity([-30,0,30],[3,10,19]),
                    sol.intensity([-30,0,30],[3,10,19]), rtol=2e-11, atol=2e-12)
    assert_allclose(divided.fluorescence(), sol.fluorescence(), rtol=2e-11)
    model.layers[0].sigma = model.substrate_sigma = 0
    sharp = xsw.solve(model, **settings)
    assert abs(sharp.fluorescence()-sol.fluorescence()) > .01


def test_roughness_snapshot_and_parallel_scan():
    material = xsw.Material(-.01+.002j, fluorescent=True)
    model = xsw.Structure(100, substrate_sigma=.8).add_layer(
        xsw.Layer(20, sigma=1.2).add_box(-20,20,material,sigma=2))
    prepared = xsw.prepare(model, max_order=2)
    settings = dict(wavelength=2.4, polarization='p', max_order=2, material=material)
    serial = xsw.fluorescence_map(model, [.12,.15], .3, **settings)
    parallel = xsw.fluorescence_map(model, [.12,.15], .3, workers=2, chunk_size=1, **settings)
    assert_allclose(serial, parallel, rtol=2e-12)
    model.substrate_sigma = model.layers[0].sigma = 0
    assert prepared.structure.substrate_sigma == .8
    assert prepared.structure.layers[0].sigma == 1.2
    assert_allclose(prepared.solve(wavelength=2.4, alpha=.12, phi=.3,
                                   polarization='p').fluorescence(material), serial[0])


def test_no_contrast_interface_is_unaffected():
    model = xsw.Structure(100, substrate_sigma=1e6).add_layer(xsw.Layer(20, sigma=1e6))
    sol = xsw.solve(model, wavelength=2.4, alpha=.12, phi=.3, max_order=3)
    assert_allclose(sol.intensity([0,10],[-5,10,30]), 1, atol=2e-12)
    assert_allclose([sol.reflectance,sol.transmittance], [0,1], atol=2e-12)


@pytest.mark.parametrize('pol', ['s', 'p'])
def test_rough_grating_is_invariant_to_modal_basis_scaling(monkeypatch, pol):
    from xsw import solver

    material = xsw.Material(-.01+.002j, fluorescent=True)
    model = xsw.Structure(100, substrate_sigma=.8)
    model.add_layer(xsw.Layer(12, sigma=.9).add_box(-15,15,material))
    model.add_layer(xsw.Layer(8, sigma=.7).add_box(-22,22,material))
    settings = dict(wavelength=2.4, alpha=.13, phi=.3, polarization=pol, max_order=3)
    reference = xsw.solve(model, **settings)
    original = solver._patterned

    def rescaled(*args):
        modes = original(*args)
        order = np.arange(len(modes.q))[::-1]
        scale = np.linspace(.7,1.4,len(order))*np.exp(1j*np.linspace(.2,.9,len(order)))
        return solver.Modes(modes.q[order], modes.electric[:,:,order]*scale,
                            modes.magnetic[:,:,order]*scale, modes.plus[:,order]*scale,
                            modes.minus[:,order]*scale)

    monkeypatch.setattr(solver, '_patterned', rescaled)
    changed = xsw.solve(model, **settings)
    assert_allclose(changed.fields([-30,0,30],[-1,5,15,21]),
                    reference.fields([-30,0,30],[-1,5,15,21]), atol=3e-12)
    assert_allclose(changed.fluorescence(), reference.fluorescence(), rtol=2e-12)


def test_sparse_rough_stack_and_thick_layer_stability():
    material = xsw.Material(-.01+.002j, fluorescent=True)
    single = xsw.Structure(100, substrate_sigma=.8).add_layer(
        xsw.Layer(20, background=material, sigma=1.2))
    sliced = xsw.Structure(100, substrate_sigma=.8)
    for j in range(20):
        sliced.add_layer(xsw.Layer(1, background=material, sigma=1.2 if j == 0 else 0))
    settings = dict(wavelength=2.4, alpha=.13, phi=.3, polarization='p', max_order=3)
    a, b = xsw.solve(single, **settings), xsw.solve(sliced, **settings)
    assert_allclose([a.reflectance,a.transmittance,a.fluorescence()],
                    [b.reflectance,b.transmittance,b.fluorescence()], rtol=2e-11)
    single.layers[0].thickness = 1e6
    thick = xsw.solve(single, **settings)
    assert np.isfinite(thick.intensity([0],[1,1e5,1e6])).all()
    assert thick.transmittance == 0
    assert thick.boundary_residual < 1e-12


@pytest.mark.parametrize('sigma', [-1,np.nan,np.inf,1j,[1,2],True,'2'])
def test_invalid_interface_sigma(sigma):
    with pytest.raises(ValueError, match='sigma'):
        xsw.Layer(20, sigma=sigma)
    with pytest.raises(ValueError, match='substrate_sigma'):
        xsw.Structure(100, substrate_sigma=sigma)
    model = xsw.Structure(100).add_layer(xsw.Layer(20))
    model.layers[0].sigma = sigma
    with pytest.raises(ValueError, match='sigma'):
        xsw.prepare(model)
    model.layers[0].sigma = 0
    model.substrate_sigma = sigma
    with pytest.raises(ValueError, match='substrate_sigma'):
        xsw.prepare(model)


def test_overflowing_evanescent_roughness_is_reported():
    model = xsw.Structure(100, substrate=xsw.Material(-.02), substrate_sigma=1e6)
    with pytest.raises(ValueError, match='roughness factors overflow'):
        xsw.solve(model, wavelength=2.4, alpha=.01, max_order=0)
