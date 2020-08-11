import sys
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import doodads as dd

def adjust_evec_signs(evecs_a, evecs_b):
    '''Adjusts the signs of evecs_b to match evecs_a
    '''
    signs = np.zeros(evecs_a.shape[1])
    for col in range(evecs_a.shape[1]):
        signs[col] = 1 if np.allclose(evecs_a[:,col], evecs_b[:,col]) else -1
    return signs * evecs_b

def compare_u(evecs_a, evecs_b, display=False):
    '''Ensure two matrices of eigenvectors agree up to a factor of +/-1
    '''
    evecs_b_mod = adjust_evec_signs(evecs_a, evecs_b)
    if display:
        import matplotlib.pyplot as plt
        vmax = np.percentile(np.abs([evecs_a, evecs_b]), 99)
        fig, (ax_iu, ax_fu, ax_du) = plt.subplots(ncols=3, figsize=(12, 4))
        dd.add_colorbar(ax_iu.imshow(evecs_a, vmin=-vmax, vmax=vmax))
        ax_iu.set_title(r'$\mathbf{U}_\mathrm{first}$')
        dd.add_colorbar(ax_fu.imshow(evecs_b_mod, vmin=-vmax, vmax=vmax))
        ax_fu.set_title(r'(signs) * $\mathbf{U}_\mathrm{second}$')
        dd.add_colorbar(ax_du.imshow(evecs_b_mod - evecs_a, cmap='RdBu_r'))
        ax_du.set_title(r'(signs) * $\mathbf{U}_\mathrm{second}$ - $\mathbf{U}_\mathrm{first}$')
    return np.allclose(evecs_b_mod, evecs_a)

def main():
    input_data_fn, c_cov_fn, c_evecs_fn, c_evals_fn = sys.argv[1:]
    input_data = fits.getdata(input_data_fn)
    if len(input_data.shape) > 2:
        plane, row, col = input_data.shape
        input_data = input_data.reshape(plane, row * col)
    c_cov = fits.getdata(c_cov_fn)
    c_evecs = fits.getdata(c_evecs_fn)
    c_evals = fits.getdata(c_evals_fn)
    np_cov = np.cov(input_data)
    np_evals, np_evecs = np.linalg.eigh(np_cov)
    try:
        assert np.allclose(c_cov, np_cov)
        assert np.allclose(c_evals, np_evals)
        assert compare_u(c_evecs, np_evecs)
    except AssertionError:
        compare_u(c_evecs, np_evecs, display=True)
        plt.savefig('compare_evecs.png')
        plt.clf()
        plt.plot(c_evals, label='C')
        plt.plot(np_evals, label='Python')
        plt.yscale('log')
        plt.legend()
        plt.savefig('compare_evals.png')
        
        import IPython
        IPython.embed()


if __name__ == "__main__":
    main()
    sys.exit(0)