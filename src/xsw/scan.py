"""Ordered, bounded-memory angular scans with explicit process/thread controls."""
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from numbers import Integral

import numpy as np
from threadpoolctl import threadpool_limits

from .solver import prepare

_WORKER = None


def _initialize(payload):
    global _WORKER
    _WORKER = payload


def _evaluate(payload, angles):
    prepared, settings, material, attenuation, blas_threads = payload
    result = []
    with threadpool_limits(limits=blas_threads):
        for alpha, phi in angles:
            solution = prepared.solve(alpha=alpha, phi=phi, **settings)
            result.append(solution.fluorescence(material, attenuation=attenuation))
    return result


def _worker(angles):
    return _evaluate(_WORKER, angles)


def fluorescence_map(structure, alpha, phi, *, wavelength=None, energy=None,
                     polarization="s", max_order=15, material=None,
                     attenuation=0.0, workers=1, chunk_size=32, blas_threads=1):
    """Broadcast angle arrays and return relative yields with the same shape.

    For a rectangular scan pass alpha[:,None], phi[None,:]. Scalars return a
    zero-dimensional array. Chunks bound task submission and worker memory;
    dense eigensolves remain per illumination, rather than stacked in RAM.
    Guard the calling script's main entry point when workers > 1.
    """
    for name, value in [('workers', workers), ('chunk_size', chunk_size), ('blas_threads', blas_threads)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    alpha, phi = np.broadcast_arrays(np.asarray(alpha, float), np.asarray(phi, float))
    if not np.isfinite(alpha).all() or not np.isfinite(phi).all() or np.any((alpha <= 0) | (alpha > np.pi/2)):
        raise ValueError("angles must be finite; alpha must be in (0,pi/2] radians")
    prepared = prepare(structure, max_order=max_order)
    settings = dict(wavelength=wavelength, energy=energy, polarization=polarization)
    # One payload preserves shared Material identity across the process boundary.
    payload = (prepared, settings, material, attenuation, blas_threads)
    angles = iter(zip(alpha.flat, phi.flat))
    output = np.empty(alpha.size)
    offset = 0
    if workers == 1:
        while batch := list(islice(angles, chunk_size)):
            values = _evaluate(payload, batch)
            output[offset:offset+len(values)] = values
            offset += len(values)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize, initargs=(payload,)) as pool:
            while True:
                batches = [list(islice(angles, chunk_size)) for _ in range(workers)]
                batches = [batch for batch in batches if batch]
                if not batches:
                    break
                for values in pool.map(_worker, batches):
                    output[offset:offset+len(values)] = values
                    offset += len(values)
    return output.reshape(alpha.shape)
