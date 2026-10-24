"""Ru grating example, following the legacy material/layer/box workflow.

Sharp boxes (sigma=0); the legacy sigma=6 nm Fourier smoothing is not applied.
Run after installing xsw. Output is numerical by default; plotting is optional.
"""
import argparse
from pathlib import Path
import numpy as np
import xsw


def sample():
    si = xsw.Material(-0.9763e-5 + 0.149e-6j, name="Si")
    ru = xsw.Material(-4.4676e-5 + 2.279e-6j, name="Ru", fluorescent=True)
    model = xsw.Structure(period=136, substrate=si)
    model.add_layer(xsw.Layer(25).add_box(-37.1/2, 37.1/2, ru))
    return model, ru


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('examples/output/ru.npz'))
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-order', type=int, default=7)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    model, ru = sample()
    alpha_deg = np.linspace(.02, .7, 21)
    phi_deg = np.linspace(-.7, .7, 25)
    maps = {}
    for pol in ['s', 'p']:
        maps[pol] = xsw.fluorescence_map(
            model, np.deg2rad(alpha_deg)[:,None], np.deg2rad(phi_deg)[None,:],
            wavelength=.124, polarization=pol, max_order=args.max_order,
            material=ru, workers=args.workers)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output, alpha_deg=alpha_deg, phi_deg=phi_deg,
             yield_s=maps['s'], yield_p=maps['p'], wavelength=.124, max_order=args.max_order)
    print(f'Saved {args.output}; shape={maps["s"].shape}, orders={2*args.max_order+1}')
    if args.plot:
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        scale=max(m.max() for m in maps.values())
        for ax,(pol,values) in zip(axes,maps.items()):
            im=ax.imshow(xsw.normalize(values,scale=scale),origin='lower',aspect='auto',
                         extent=[phi_deg[0],phi_deg[-1],alpha_deg[0],alpha_deg[-1]],vmin=0,vmax=1)
            ax.set(xlabel='Azimuth (degrees)',ylabel='Grazing angle (degrees)',title=f'{pol} excitation')
        fig.colorbar(im,ax=axes,label='Relative fluorescence (common scale)')
        fig.savefig(args.output.with_suffix('.png'),dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
