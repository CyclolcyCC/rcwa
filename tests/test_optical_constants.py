from importlib.resources import files

import numpy as np
import pytest
from numpy.testing import assert_allclose

import xsw


def test_silica_from_tabulated_factors_and_cgs_units():
    # Independent cgs-unit evaluation; guards nm/cm conversion and factor of two.
    f1_si, f2_si = xsw.scattering_factors('Si', 10000)
    f1_o, f2_o = xsw.scattering_factors('O', 10000)
    wavelength_cm = 1239.8419843320026 / 10000 * 1e-7
    molecules_cm3 = 2.2 * 6.02214076e23 / (28.085 + 2*15.999)
    expected = (-2.8179403262e-13*wavelength_cm**2/np.pi*molecules_cm3
                * ((f1_si + 2*f1_o) - 1j*(f2_si + 2*f2_o)))
    actual = xsw.chi_from_formula('SiO2', 2.2, energy=10000)
    assert isinstance(actual, complex)
    assert actual.real < 0 < actual.imag
    # scipy.constants versions use different CODATA revisions at a few ppb.
    assert_allclose(actual, expected, rtol=1e-8)


def test_tabulated_values_and_interpolation():
    with files('xsw').joinpath('data', 'sf', 'si.nff').open() as stream:
        data = np.loadtxt(stream, skiprows=1)
    a, b = data[-10:-8]
    assert_allclose(xsw.scattering_factors('Si', a[0]), a[1:], rtol=0, atol=0)
    assert_allclose(xsw.scattering_factors('Si', (a[0]+b[0])/2), (a[1:]+b[1:])/2)


def test_all_bundled_elements_readable():
    tables = list(files('xsw').joinpath('data', 'sf').iterdir())
    tables = [table for table in tables if table.name.endswith('.nff')]
    assert len(tables) == 92
    for table in tables:
        element = table.name[:-4].capitalize()
        f1, f2 = xsw.scattering_factors(element, 10000)
        assert np.isfinite([f1, f2]).all()
        assert f2 >= 0


def test_repeated_and_unordered_energy_rows():
    # Pt's duplicate 6.199-eV entry: last supplied row is the documented policy.
    assert_allclose(xsw.scattering_factors('Pt', 6.199), [-.20202, 1.1279])
    # Er contains 1462.9 eV after 1463 eV in the supplied file.
    assert_allclose(xsw.scattering_factors('Er', 1462.9), [26.8455, 25.3825])
    assert_allclose(xsw.scattering_factors('Er', 1462.95), [(26.8455+26.9282)/2, (25.3825+25.3388)/2])


@pytest.mark.parametrize('formula,expected', [
    ('SiO2', 60.083), ('Si3N4', 3*28.085+4*14.007),
    ('Ca(OH)2', 40.078+2*(15.999+1.008)),
    ('Al2(SO4)3', 2*26.982+3*(32.06+4*15.999)),
    ('(Si(O)2)2', 2*60.083), ('Si0.7Ge0.3', .7*28.085+.3*72.630),
])
def test_molar_mass(formula, expected):
    assert_allclose(xsw.molar_mass(formula), expected)


def test_formula_scale_density_and_mass_override():
    base = xsw.chi_from_formula('SiO2', 2.2, energy=520)
    assert_allclose(xsw.chi_from_formula('Si2O4', 2.2, energy=520), base)
    assert_allclose(xsw.chi_from_formula('Si0.5O', 2.2, energy=520), base)
    assert_allclose(xsw.chi_from_formula('SiO2', 4.4, energy=520), 2*base)
    assert_allclose(xsw.chi_from_formula('SiO2', 2.2, energy=520, molar_mass=xsw.molar_mass('SiO2')/2), 2*base)
    assert xsw.chi_from_formula('SiO2', 0, energy=520) == 0j


def test_energy_array_wavelength_and_solver_integration():
    energy = np.array([[520., 1000.], [10000., 20000.]])
    chi = xsw.chi_from_formula('SiO2', 2.2, energy=energy)
    assert chi.shape == energy.shape
    assert_allclose(chi, xsw.chi_from_formula('SiO2', 2.2, wavelength=xsw.HC_EV_NM/energy))
    assert_allclose(chi[0, 0], xsw.chi_from_formula('SiO2', 2.2, energy=520))
    substrate = xsw.Material(chi[0,0])
    result = xsw.solve(xsw.Structure(100, substrate=substrate), energy=520, alpha=.1, max_order=0)
    assert 0 <= result.reflectance <= 1


def test_radioactive_element_requires_mass_override():
    with pytest.raises(ValueError, match='molar_mass'):
        xsw.chi_from_formula('Tc', 11, energy=10000)
    assert xsw.chi_from_formula('Tc', 11, energy=10000, molar_mass=98).imag > 0


@pytest.mark.parametrize('energy', [1, 20, 30001])
def test_undefined_or_out_of_range_factors_rejected(energy):
    with pytest.raises(ValueError):
        xsw.chi_from_formula('SiO2', 2.2, energy=energy)


def test_first_valid_f1_point_does_not_bridge_missing_data():
    with files('xsw').joinpath('data', 'sf', 'si.nff').open() as stream:
        data = np.loadtxt(stream, skiprows=1)
    index = np.flatnonzero(data[:,1] != -9999)[0]
    assert_allclose(xsw.scattering_factors('Si', data[index,0]), data[index,1:])
    with pytest.raises(ValueError, match='undefined'):
        xsw.scattering_factors('Si', (data[index,0]+data[index-1,0])/2)


@pytest.mark.parametrize('formula', ['', 'siO2', 'Xx2', 'Si0O2', 'Si-1', 'Si()',
                                      '(SiO2', 'SiO2)', 'SiO2·H2O', 'SiO2+', '2H2O'])
def test_invalid_formula_rejected(formula):
    with pytest.raises(ValueError):
        xsw.chi_from_formula(formula, 2.2, energy=1000)


@pytest.mark.parametrize('kwargs', [
    {'density': -1}, {'density': np.nan}, {'density': [1,2]},
    {'energy': 0}, {'energy': []}, {'energy': np.inf}, {'energy': 100+2j},
    {'energy': None}, {'wavelength': .124}, {'molar_mass': 0}, {'molar_mass': [60,61]},
])
def test_invalid_physical_inputs(kwargs):
    args = dict(density=2.2, energy=1000)
    args.update(kwargs)
    with pytest.raises(ValueError):
        xsw.chi_from_formula('SiO2', **args)
