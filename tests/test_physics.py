import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import simpson

import xsw


@pytest.mark.parametrize('pol', ['s', 'p'])
@pytest.mark.parametrize('phi', [0.0, 0.37, np.pi/2])
def test_vacuum_vector_plane_wave(pol, phi):
    alpha = .2
    structure = xsw.Structure(100).add_layer(xsw.Layer(35))
    sol = xsw.solve(structure, wavelength=2.4, alpha=alpha, phi=phi,
                    polarization=pol, max_order=3)
    x, z = np.array([-63., 0, 20]), np.array([-100, 0, 20, 35, 1000])
    k = np.array([np.cos(alpha)*np.sin(phi), np.cos(alpha)*np.cos(phi), np.sin(alpha)])
    e = (np.array([np.cos(phi), -np.sin(phi), 0]) if pol == 's'
         else np.array([np.sin(alpha)*np.sin(phi), np.sin(alpha)*np.cos(phi), -np.cos(alpha)]))
    phase = np.exp(1j*sol.k0*(k[0]*x[None,:]+k[2]*z[:,None]))
    assert_allclose(sol.fields(x,z), phase[...,None]*e, atol=3e-12)
    assert_allclose(sol.fields(x,z,magnetic=True), phase[...,None]*np.cross(k,e), atol=3e-12)
    assert_allclose(sol.intensity(x,z), 1, atol=3e-12)
    assert_allclose([sol.reflectance, sol.transmittance], [0,1], atol=2e-12)


def fresnel(eps1, eps2, alpha, pol):
    tangential2 = eps1*np.cos(alpha)**2
    q1, q2 = np.sqrt(eps1-tangential2+0j), np.sqrt(eps2-tangential2+0j)
    y1, y2 = (q1, q2) if pol == 's' else (eps1/q1, eps2/q2)
    r = (y1-y2)/(y1+y2)
    t = 1+r
    return abs(r)**2, abs(t)**2*y2.real/y1.real


@pytest.mark.parametrize('pol', ['s', 'p'])
@pytest.mark.parametrize('phi', [0., .43, np.pi/2])
@pytest.mark.parametrize('eps', [.98+0j, .98+.003j, 2.25+0j])
def test_fresnel_interface(pol, phi, eps):
    structure = xsw.Structure(100, substrate=xsw.Material(eps-1))
    sol = xsw.solve(structure,wavelength=2.4,alpha=.11,phi=phi,polarization=pol,max_order=2)
    assert_allclose([sol.reflectance, sol.transmittance], fresnel(1,eps,.11,pol), atol=2e-12)


@pytest.mark.parametrize('pol', ['s', 'p'])
def test_film_against_airy_formula(pol):
    eps, eps_sub, thickness, wl, alpha = .995+.0007j, .99+.001j, 31., 2.4, .12
    material = xsw.Material(eps-1)
    structure = xsw.Structure(100,substrate=xsw.Material(eps_sub-1))
    structure.add_layer(xsw.Layer(thickness,background=material))
    sol = xsw.solve(structure,wavelength=wl,alpha=alpha,phi=.4,polarization=pol,max_order=0)
    q = np.sqrt(np.array([1,eps,eps_sub])-np.cos(alpha)**2+0j)
    y = q if pol == 's' else np.array([1,eps,eps_sub])/q
    r01, r12 = (y[0]-y[1])/(y[0]+y[1]), (y[1]-y[2])/(y[1]+y[2])
    phase = np.exp(2j*np.pi/wl*q[1]*thickness)
    r = (r01+r12*phase**2)/(1+r01*r12*phase**2)
    t = (1+r01)*(1+r12)*phase/(1+r01*r12*phase**2)
    assert_allclose(sol.reflectance, abs(r)**2, atol=2e-12)
    assert_allclose(sol.transmittance, abs(t)**2*y[-1].real/y[0].real, atol=2e-12)


def grating(loss=.002):
    material = xsw.Material(-.01+1j*loss, name='line', fluorescent=True)
    structure = xsw.Structure(100)
    structure.add_layer(xsw.Layer(12).add_box(-18,18,material))
    structure.add_layer(xsw.Layer(8).add_box(-22,22,material))
    return structure, material


@pytest.mark.parametrize('pol', ['s','p'])
@pytest.mark.parametrize('phi', [0., .3, np.pi/2])
def test_lossless_grating_energy_and_interface_equations(pol, phi):
    structure, _ = grating(loss=0)
    sol = xsw.solve(structure,wavelength=2.4,alpha=.13,phi=phi,polarization=pol,max_order=5)
    assert_allclose(sol.reflectance+sol.transmittance, 1, atol=2e-10)
    assert sol.boundary_residual < 1e-12
    for interface, z in enumerate(structure.boundaries):
        left, right = sol.modes[interface:interface+2]
        la, lb = np.split(sol._amplitudes(interface, np.array([z]))[0],2)
        ra, rb = np.split(sol._amplitudes(interface+1, np.array([z]))[0],2)
        assert_allclose(left.plus@la+left.minus@lb, right.plus@ra+right.minus@rb, atol=2e-11)


@pytest.mark.parametrize('pol', ['s','p'])
def test_semi_analytic_yield_and_absorbed_power(pol):
    structure, material = grating()
    sol = xsw.solve(structure,wavelength=2.4,alpha=.1,phi=.3,polarization=pol,max_order=5)
    for mu in [0., .025]:
        numeric = 0.
        for index, layer in enumerate(structure.layers):
            box = layer.boxes[0]
            # Sample just inside interfaces, so integration uses one material.
            x = np.linspace(box.left+1e-10,box.right-1e-10,301)
            z = np.linspace(structure.boundaries[index]+1e-10,structure.boundaries[index+1]-1e-10,301)
            numeric += simpson(simpson(sol.intensity(x,z)*np.exp(-mu*z[:,None]),x=x,axis=1),x=z)
        assert_allclose(sol.fluorescence(material,attenuation=mu), numeric, rtol=3e-8)
    # q_abs = k0*epsilon''*|E|^2/2 in Z0-normalized power units.
    absorption = sol.k0*material.chi.imag*sol.fluorescence(material)/(2*sol.incident_flux*structure.period)
    assert_allclose(sol.absorptance,absorption,atol=3e-11)


@pytest.mark.parametrize('pol',['s','p'])
def test_uniform_layer_subdivision(pol):
    material=xsw.Material(-.01+.002j,fluorescent=True)
    whole=xsw.Structure(100).add_layer(xsw.Layer(20).add_box(-20,20,material))
    sliced=xsw.Structure(100)
    for _ in range(8):
        sliced.add_layer(xsw.Layer(2.5).add_box(-20,20,material))
    args=dict(wavelength=2.4,alpha=.1,phi=.3,polarization=pol,max_order=4)
    a,b=xsw.solve(whole,**args),xsw.solve(sliced,**args)
    assert_allclose(a.intensity([-30,-10,5,40],[1,7,19]),b.intensity([-30,-10,5,40],[1,7,19]),atol=3e-11)
    assert_allclose(a.fluorescence(material),b.fluorescence(material),rtol=3e-12)


def test_thick_absorbing_layer_stability():
    material=xsw.Material(-.01+.01j,fluorescent=True)
    structure=xsw.Structure(100,substrate=material).add_layer(xsw.Layer(1e6,background=material))
    with np.errstate(over='raise',invalid='raise'):
        sol=xsw.solve(structure,wavelength=2.4,alpha=.1,polarization='p',max_order=2)
        assert np.isfinite(sol.intensity([0],[0,10,1e6,1e7])).all()
        y=sol.fluorescence(material)
    absorbed=sol.k0*material.chi.imag*y/(2*sol.incident_flux*structure.period)
    assert_allclose(absorbed,1-sol.reflectance,atol=1e-11)


def test_fourier_convergence():
    structure,material=grating()
    values=[]
    for order in [3,7,15,23]:
        sol=xsw.solve(structure,wavelength=2.4,alpha=.1,phi=.3,polarization='p',max_order=order)
        values.append(sol.fluorescence(material))
    assert abs(values[-1]-values[-2]) < abs(values[-1]-values[0])
    assert abs(values[-1]-values[-2])/values[-1] < .002


def test_conical_maxwell_equations_in_uniform_layer():
    """Independent real-space curl check, all components, off-axis illumination."""
    material=xsw.Material(-.01+.002j)
    structure=xsw.Structure(100).add_layer(xsw.Layer(20,background=material))
    sol=xsw.solve(structure,wavelength=2.4,alpha=.13,phi=.4,polarization='p',max_order=0)
    x,z,h=3.,7.,1e-4
    def derivatives(magnetic):
        f=sol.fields([x],[z],magnetic=magnetic)[0,0]
        dx=(sol.fields([x+h],[z],magnetic=magnetic)[0,0]-sol.fields([x-h],[z],magnetic=magnetic)[0,0])/(2*h)
        dz=(sol.fields([x],[z+h],magnetic=magnetic)[0,0]-sol.fields([x],[z-h],magnetic=magnetic)[0,0])/(2*h)
        dy=1j*sol.k0*sol.ky*f
        curl=np.array([dy[2]-dz[1],dz[0]-dx[2],dx[1]-dy[0]])
        return f,curl
    e,ce=derivatives(False)
    hfield,ch=derivatives(True)
    assert_allclose(ce,1j*sol.k0*hfield,rtol=1e-7,atol=1e-8)
    assert_allclose(ch,-1j*sol.k0*material.epsilon*e,rtol=1e-7,atol=1e-8)


def test_trapezoid_slice_convergence():
    material=xsw.Material(-.01+.002j,fluorescent=True)
    yields=[]
    for count in [2,4,8,16]:
        structure=xsw.Structure(100)
        for index in range(count):
            width=20+20*(index+.5)/count
            structure.add_layer(xsw.Layer(20/count).add_box(-width/2,width/2,material))
        sol=xsw.solve(structure,wavelength=2.4,alpha=.1,phi=.3,polarization='p',max_order=5)
        yields.append(sol.fluorescence(material))
    assert abs(yields[-1]-yields[-2]) < abs(yields[-1]-yields[0])
    assert abs(yields[-1]-yields[-2])/yields[-1] < 1e-4
