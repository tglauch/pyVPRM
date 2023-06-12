import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), 'lib'))
from fancy_plot import *
import sys
import os
import xarray as xr
import glob
import argparse

p = argparse.ArgumentParser(
        description = "Commend Line Arguments",
        formatter_class = argparse.RawTextHelpFormatter)
p.add_argument("--day_start", type=int)
p.add_argument("--day_stop", type=int) 
args = p.parse_args()

for d in np.arange(args.day_start, args.day_stop):
    print(d)
    datasets = [xr.open_dataset(i) for i in glob.glob('/work/bd1231/tglauch/gpp_eu_2015/*{:03d}.h5'.format(d)) if 'h20' not in i ]
    if len(datasets) == 0:
        continue
    merged = xr.combine_by_coords(datasets)
    del datasets
    for j in range(24):
        print(j)
        fig, ax = newfig(2.0)
        merged['__xarray_dataarray_variable__'].isel({'time': j}).plot.imshow(x='x', y='y', cmap='Greens',
                                                                              vmin=0, vmax=25, ax=ax)
        fig.savefig('/work/bd1231/tglauch/gpp_eu_2015/plots/{:03d}_{:02d}.png'.format(d, j), bbox_inches='tight', dpi=150)
        plt.close(fig)
    del merged
