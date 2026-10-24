"""Compare vector electric intensity for s and p excitation of a Ru grating."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import xsw
from a_simple_structure import sample


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('examples/output/fields.png'))
    args=parser.parse_args()
    model,_=sample()
    prepared=xsw.prepare(model,max_order=15)
    x,z=np.linspace(-68,68,301),np.linspace(-10,40,251)
    with threadpool_limits(limits=1):
        values=[prepared.solve(wavelength=.124,alpha=np.deg2rad(.3),phi=np.deg2rad(.2),
                               polarization=pol).intensity(x,z) for pol in ['s','p']]
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,pol,value in zip(axes,['s','p'],values):
        im=ax.imshow(value,extent=[x[0],x[-1],z[-1],z[0]],aspect='auto',
                     cmap='inferno',vmin=0,vmax=max(v.max() for v in values))
        ax.plot([-68,-18.55,-18.55,18.55,18.55,68],[25,25,0,0,25,25],'w-',lw=.8)
        ax.set(xlabel='x (nm)',ylabel='Depth z (nm)',title=f'{pol} excitation')
    fig.colorbar(im,ax=axes,label='Electric intensity / incident intensity')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160)
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
