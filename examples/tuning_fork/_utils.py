
import numpy as np
import time

def test(beam3d, init_design, demo=True, deriv=False):
    if demo:
        # now build 3D beam solver and solve the eigenvalue problem
        freqs = beam3d.get_frequencies(init_design)
        print(f"{freqs=}")
        nmodes = 5
        beam3d.plot_eigenmodes(
            nmodes=nmodes,
            show=False,
            def_scale=0.5
        )

        # see if we can get derivatives of the natural frequencies now
        freq_grads = np.zeros((beam3d.num_dvs, nmodes))
        start_time = time.time()
        print("Getting eigenvalue derivatives:")
        for imode in range(5):
            freq_grads[:,imode] = beam3d.get_frequency_gradient(init_design, imode)
        dt = time.time() - start_time
        print(f"\tcomputed freq grads in {dt:.4f} seconds.") 

    if deriv:
        # FD test on the gradients
        for imode in range(4):
            beam3d.freq_FD_test(init_design, imode, h=1e-3)
            beam3d.dKdx_FD_test(init_design, imode, h=1e-5)
            beam3d.dMdx_FD_test(init_design, imode, h=1e-5)