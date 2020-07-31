import sys
import numpy as np
from astropy.io import fits

def compare_u(init_u, final_u, display=False):
    '''Ensure two matrices of eigenvectors agree up to a factor of +/-1
    '''
    signs = np.zeros(init_u.shape[1])
    for col in range(init_u.shape[1]):
        signs[col] = 1 if np.allclose(init_u[:,col], final_u[:,col]) else -1
    vmax = np.max(np.abs([init_u, final_u]))
    final_u_mod = signs * final_u
    if display:
        import matplotlib.pyplot as plt
        fig, (ax_iu, ax_fu, ax_du) = plt.subplots(ncols=3, figsize=(12, 4))
        ax_iu.imshow(init_u, vmin=-vmax, vmax=vmax)
        ax_iu.set_title(r'$\mathbf{U}_\mathrm{first}$')
        ax_fu.imshow(final_u_mod, vmin=-vmax, vmax=vmax)
        ax_fu.set_title(r'(signs) * $\mathbf{U}_\mathrm{second}$')
        plt.colorbar(ax_du.imshow(final_u_mod - init_u, cmap='RdBu_r'))
        ax_du.set_title(r'(signs) * $\mathbf{U}_\mathrm{second}$ - $\mathbf{U}_\mathrm{first}$')
    return np.allclose(final_u_mod, init_u)

def main():
    input_data_fn, c_cov_fn, c_evecs_fn, c_evals_fn = sys.argv[1:]
    input_data = fits.getdata(input_data_fn)
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
        import IPython
        IPython.embed()


if __name__ == "__main__":
    main()
    sys.exit(0)