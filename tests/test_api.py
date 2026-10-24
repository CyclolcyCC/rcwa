import numpy as np
import pytest
from numpy.testing import assert_allclose
import xsw


def sample():
    material=xsw.Material(-.005+.001j,name='A',fluorescent=True,density=2)
    structure=xsw.Structure(80).add_layer(xsw.Layer(10).add_box(-15,15,material))
    return structure,material


def test_prepared_snapshot_and_material_identity():
    structure,material=sample()
    prepared=xsw.prepare(structure,max_order=2)
    structure.layers[0].thickness=99
    sol=prepared.solve(wavelength=2,alpha=.2)
    assert sol.structure.thickness==10
    assert sol.structure.layers[0].boxes[0].material is material
    assert_allclose(sol.fluorescence(),sol.fluorescence(material))
    assert sol.fluorescence(xsw.Material(material.chi))==0


def test_broadcast_scan_matches_serial():
    structure,material=sample()
    alphas=np.array([.1,.15])[:,None]
    phis=np.array([0,.2,.4])[None,:]
    args=dict(wavelength=2,max_order=2,material=material,polarization='p')
    result=xsw.fluorescence_map(structure,alphas,phis,chunk_size=2,**args)
    reference=np.array([[xsw.solve(structure,wavelength=2,max_order=2,alpha=a,phi=p,polarization='p').fluorescence(material)
                         for p in phis.ravel()] for a in alphas.ravel()])
    assert result.shape==(2,3)
    assert_allclose(result,reference,rtol=2e-12)


def test_parallel_scan_preserves_material_selection():
    structure,material=sample()
    # Same names/properties must not merge distinct material identities.
    other=xsw.Material(material.chi,name=material.name,fluorescent=True)
    structure.layers[0].add_box(20,30,other)
    args=dict(wavelength=2,max_order=2,material=material,chunk_size=1)
    serial=xsw.fluorescence_map(structure,[.1,.15,.2],.2,workers=1,**args)
    parallel=xsw.fluorescence_map(structure,[.1,.15,.2],.2,workers=2,**args)
    assert_allclose(serial,parallel,rtol=2e-12)


def test_energy_and_wavelength_equivalence():
    structure,_=sample()
    a=xsw.solve(structure,wavelength=2,alpha=.2,max_order=2)
    b=xsw.solve(structure,energy=xsw.HC_EV_NM/2,alpha=.2,max_order=2)
    assert_allclose(a.intensity([0],[3]),b.intensity([0],[3]))


@pytest.mark.parametrize('kwargs',[{'alpha':0},{'alpha':-1},{'alpha':2},
                                  {'polarization':'x'},{'max_order':-1},
                                  {'max_order':1.5},{'wavelength':0}, {'phi':float('nan')}])
def test_invalid_solver_inputs(kwargs):
    args=dict(wavelength=2,alpha=.1,max_order=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        xsw.solve(xsw.Structure(100),**args)


def test_invalid_geometry_and_normalization():
    with pytest.raises(ValueError):
        xsw.Layer(0)
    layer=xsw.Layer(10).add_box(-10,10,xsw.VACUUM)
    with pytest.raises(ValueError):
        layer.add_box(0,20,xsw.VACUUM)
    with pytest.raises(ValueError):
        xsw.Structure(10).add_layer(layer)
    with pytest.raises(ValueError):
        xsw.Material(-.1-.01j)
    with pytest.raises(ValueError):
        xsw.normalize([0,0])
    assert_allclose(xsw.normalize([1,2],scale=4),[.25,.5])
