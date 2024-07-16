import numpy as np
import matplotlib.pyplot as plt
import lightbeam
from lightbeam import screen
import lightbeam.LPmodes as LPmodes
from lightbeam.misc import normalize

# If in main
NScreens = 1
n_input = 2**11
n_focal = 256

if __name__ == "__main__":
    n = n_input

    D = 0.7 # [m]  telescope diameter
    p = D/n # [m/pix] sampling scale
    # set wind parameters
    vy, vx = 0., 10. # [m/s] wind velocity vector
    T = 0.01 # [s]  sampling interval
    # set turbulence parameters
    r0  = 0.15 # [m]
    wl0 = 1.550 #[um]
    wl  = 1 #[um]

    psgen = screen.PhaseScreenGenerator(D, p, vy, vx, T, r0, wl0, wl,alpha_mag=0.9)


    xa = ya = np.linspace(-50/2,50/2,n)
    xg,yg = np.meshgrid(xa,ya)
    rcore = 3
    ncore = 1.4504 + 0.0088
    nclad = 1.4504
    u0 = normalize(LPmodes.lpfield(xg,yg,0,1,rcore,wl,ncore,nclad))

    fig,ax = plt.subplots(1,2)
    ax[0].imshow(np.abs(u0)**2)
    ax[1].imshow(np.angle(u0))
    plt.show()

    for i in range(NScreens):
        P = psgen.generate()
        p = np.exp(1j*P)

        u1 = u0*p

        u1_f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u1)))

        # Stitch Out
        minI,maxI = n_input//2-n_focal//2, n_input//+n_focal//2
        fp = u1_f #[minI:maxI,minI:maxI]
        plt.imshow(np.abs(fp)**2)
        plt.show()
