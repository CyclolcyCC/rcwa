import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import simpson

import xsw
from xsw.model import convolution


@pytest.mark.parametrize('inverse', [False, True])
def test_box_fourier_damping_and_mean(inverse):
    background = xsw.Material(-.002+.0001j)
    material = xsw.Material(-.01+.002j)
    orders = np.arange(-3, 4)
    sharp = xsw.Layer(20, background=background).add_box(-18, 24, material)
    rough = xsw.Layer(20, background=background).add_box(-18, 24, material, sigma=4)
    base = (1/background.epsilon if inverse else background.epsilon)*np.eye(len(orders))
    c0 = convolution(sharp, 100, orders, inverse=inverse)
    c1 = convolution(rough, 100, orders, inverse=inverse)
    g = 2*np.pi*(orders[:,None]-orders[None,:])/100
    assert_allclose(c1-base, (c0-base)*np.exp(-.5*(g*4)**2), atol=1e-16)
    assert_array_equal(c1.diagonal(), c0.diagonal())


def test_independent_boxes_and_preparation_snapshot():
    a, b = xsw.Material(-.01), xsw.Material(-.02)
    layer = xsw.Layer(20).add_box(-30,-10,a,sigma=2).add_box(10,30,b,sigma=7)
    model = xsw.Structure(100).add_layer(layer)
    prepared = xsw.prepare(model, max_order=3)
    assert [box.sigma for box in prepared.structure.layers[0].boxes] == [2,7]
    for inverse in [False, True]:
        first = xsw.Layer(20).add_box(-30,-10,a,sigma=2)
        second = xsw.Layer(20).add_box(10,30,b,sigma=7)
        expected = (convolution(first,100,prepared.orders,inverse=inverse)
                    + convolution(second,100,prepared.orders,inverse=inverse) - np.eye(7))
        assert_allclose(convolution(layer,100,prepared.orders,inverse=inverse),expected,atol=1e-16)


@pytest.mark.parametrize('pol', ['s','p'])
def test_zero_sigma_recovers_default_solution(pol):
    material = xsw.Material(-.01+.002j,fluorescent=True)
    default = xsw.Structure(100).add_layer(xsw.Layer(20).add_box(-20,20,material))
    explicit = xsw.Structure(100).add_layer(xsw.Layer(20,boxes=[xsw.Box(-20,20,material,sigma=0)]))
    args = dict(wavelength=2.4,alpha=.1,phi=.3,polarization=pol,max_order=4)
    a,b = xsw.solve(default,**args),xsw.solve(explicit,**args)
    assert default.layers[0].boxes[0].sigma == 0.0
    assert_array_equal(a.intensity([-30,0,30],[0,10,19]),b.intensity([-30,0,30],[0,10,19]))
    assert a.fluorescence(material) == b.fluorescence(material)


@pytest.mark.parametrize('pol', ['s','p'])
def test_roughness_yield_agrees_with_nominal_box_quadrature(pol):
    material = xsw.Material(-.01+.002j,fluorescent=True)
    model = xsw.Structure(100).add_layer(xsw.Layer(20).add_box(-20,20,material,sigma=6))
    args = dict(wavelength=2.4,alpha=.1,phi=.3,polarization=pol,max_order=5)
    sol = xsw.solve(model,**args)
    x,z = np.linspace(-20+1e-10,20-1e-10,301),np.linspace(1e-10,20-1e-10,301)
    for mu in [0.,.03]:
        numeric = simpson(simpson(sol.intensity(x,z)*np.exp(-mu*z[:,None]),x=x,axis=1),x=z)
        assert_allclose(sol.fluorescence(material,attenuation=mu),numeric,rtol=3e-8)
    sharp = xsw.Structure(100).add_layer(xsw.Layer(20).add_box(-20,20,material))
    assert abs(sol.fluorescence(material)-xsw.solve(sharp,**args).fluorescence(material)) > .01


@pytest.mark.parametrize('pol', ['s','p'])
def test_lossless_roughness_flux_conservation(pol):
    material = xsw.Material(-.01)
    model = xsw.Structure(100).add_layer(xsw.Layer(20).add_box(-20,20,material,sigma=5))
    sol = xsw.solve(model,wavelength=2.4,alpha=.13,phi=.3,polarization=pol,max_order=5)
    assert_allclose(sol.reflectance+sol.transmittance,1,atol=2e-10)
    assert sol.boundary_residual < 1e-12


def test_large_sigma_retains_harmonic_and_arithmetic_averages():
    material = xsw.Material(-.4)
    model = xsw.Structure(100).add_layer(xsw.Layer(20).add_box(-20,20,material,sigma=1e4))
    sol = xsw.solve(model,wavelength=2.4,alpha=.5,phi=np.pi/2,max_order=2)
    eps_t = .6 + .4*material.epsilon
    eps_x = 1/(.6 + .4/material.epsilon)
    expected = np.r_[eps_t-sol.kx**2, eps_x*(1-sol.kx**2/eps_t)]
    assert_allclose(np.sort(sol.modes[1].q**2),np.sort(expected),atol=2e-14)
    assert_allclose(sol.reflectance+sol.transmittance,1,atol=1e-12)


def test_full_period_material_unaffected():
    material = xsw.Material(-.01+.002j)
    args = dict(wavelength=2.4,alpha=.1,phi=.3,polarization='p',max_order=3)
    uniform = xsw.Structure(100).add_layer(xsw.Layer(20,background=material))
    rough = xsw.Structure(100).add_layer(xsw.Layer(20).add_box(-50,50,material,sigma=5))
    a,b = xsw.solve(uniform,**args),xsw.solve(rough,**args)
    assert_allclose([a.reflectance,a.transmittance],[b.reflectance,b.transmittance],atol=2e-12)


@pytest.mark.parametrize('sigma', [-1,np.nan,np.inf,1j,[1,2],True,'2'])
def test_invalid_sigma(sigma):
    with pytest.raises(ValueError, match='sigma'):
        xsw.Box(-10,10,xsw.VACUUM,sigma=sigma)
    with pytest.raises(ValueError, match='sigma'):
        xsw.Layer(20).add_box(-10,10,xsw.VACUUM,sigma=sigma)
