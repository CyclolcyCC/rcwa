"""Reproducible small benchmark; timings are observations, not performance tests."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from threadpoolctl import threadpool_info
import xsw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/latest.json'))
    args=parser.parse_args()
    ru=xsw.Material(-4.4676e-5+2.279e-6j,fluorescent=True)
    si=xsw.Material(-.9763e-5+.149e-6j)
    structure=xsw.Structure(136,substrate=si).add_layer(xsw.Layer(25).add_box(-18.55,18.55,ru))
    alpha=np.deg2rad(np.linspace(.05,.7,10))[:,None]
    phi=np.deg2rad(np.linspace(-.7,.7,10))[None,:]
    config=dict(wavelength=.124,polarization='p',max_order=7,material=ru,chunk_size=16,blas_threads=1)
    xsw.fluorescence_map(structure,alpha[:1],phi,**config)  # warmup
    timings={}
    reference=None
    error=0.
    for workers in [1,2]:
        samples=[]
        for _ in range(3):
            start=time.perf_counter()
            values=xsw.fluorescence_map(structure,alpha,phi,workers=workers,**config)
            samples.append(time.perf_counter()-start)
        if reference is None:
            reference=values
        else:
            error=float(np.max(abs(values-reference))/np.max(abs(reference)))
        timings[str(workers)]=dict(seconds=samples,median_seconds=float(np.median(samples)))
    record=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                platform=platform.platform(),blas=threadpool_info(),period_nm=136,height_nm=25,
                width_nm=37.1,chi_ru=[ru.chi.real,ru.chi.imag],chi_si=[si.chi.real,si.chi.imag],
                wavelength_nm=.124,alpha_degrees=[.05,.7,10],phi_degrees=[-.7,.7,10],
                polarization='p',max_order=7,chunk_size=16,blas_threads=1,
                timings=timings,parallel_relative_error=error,
                memory_note='At most workers*chunk_size points submitted per group; no stacked eigensystems.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(timings=timings,parallel_relative_error=error),indent=2))


if __name__ == '__main__':
    main()
