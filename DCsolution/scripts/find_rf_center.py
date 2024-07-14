#!/usr/bin/python

import h5py
import os
import os.path
import sys
# import matplotlib.pyplot as plt
import numpy as np

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(root_path, "data")
sys.path.append(root_path)

from trap_dc import solutions, potentials

if len(sys.argv) != 3:
    print(f'''
Calculate RF center location along trap axis.

Usage:

    {sys.argv[0]} trap_name potential_file

Arguments:

    trap_name: one of `hoa`, `phoenix` or `peregrine`.

    potential_file: the voltage solution file provided by Sandia.''')
    exit(1)

trap = sys.argv[1]
pfile = sys.argv[2]

os.makedirs(data_path, exist_ok=True)

potential = potentials.Potential.import_64(pfile, trap=trap)
# RF is electrode 1 (ground is 0)
centers = solutions.find_all_flat_points(potential.data[1, :, :, :])
centers_um = np.empty(centers.shape)

xs_um = [potential.x_index_to_axis(i) * 1000 for i in range(potential.nx)]

for i in range(potential.nx):
    centers_um[0, i] = potential.y_index_to_axis(centers[0, i]) * 1000
    centers_um[1, i] = potential.z_index_to_axis(centers[1, i]) * 1000

with h5py.File(os.path.join(data_path, f"rf_center_{trap}.h5"), 'w') as fh:
    fh.create_dataset('yz_index', data=centers)
    fh.create_dataset('yz_um', data=centers_um)
    fh.create_dataset('xs_um', data=xs_um)

# xs_um = [potential.x_index_to_axis(i) * 1000 for i in range(potential.nx)]

# plt.figure(figsize=[6.4 * 2, 4.8])
# plt.subplot(1, 2, 1)
# plt.plot(xs_um, centers_um[1, :])
# plt.xlabel("X ($\\mu$m)")
# plt.ylabel("Z ($\\mu$m)")
# plt.title("RF null Z position")
# plt.grid()

# plt.subplot(1, 2, 2)
# plt.plot(xs_um, centers_um[0, :])
# plt.xlabel("X ($\\mu$m)")
# plt.ylabel("Y ($\\mu$m)")
# plt.title("RF null Y position")
# plt.grid()

# plt.show()
