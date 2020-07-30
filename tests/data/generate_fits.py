from astropy.io import fits
import numpy as np


def write_data(name, data):
    fits.PrimaryHDU(data).writeto(f'./{name}.fits', overwrite=True)
    if len(data.shape) < 3:
        np.savetxt(f'./{name}.txt', data)


A = np.ones((4, 3, 2), dtype=int)
A[1] *= 2
A[2] *= 3
A[3] *= 4
write_data('sequential_cube_2cols_3rows_4planes', A)

indices_16x4 = np.arange(64).reshape(16, 4)
write_data('indices_16x4', indices_16x4)
write_data('indices_16x4_float', indices_16x4.astype(float))

lapacke_dsyevr_row_ex = np.asarray([
    [0.67, -0.20, 0.19,  -1.06,   0.46],
    [-0.20, 3.82, -0.13,  1.06,  -0.48],
    [0.19, -0.13, 3.27,   0.11,   1.10],
    [-1.06, 1.06, 0.11,   5.86,  -0.98],
    [0.46, -0.48, 1.10,  -0.98,   3.54],
])

write_data('lapacke_dsyevr_row_ex', lapacke_dsyevr_row_ex)

np.random.seed(1)
write_data('noise_20_30', np.random.randn(20, 30))