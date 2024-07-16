import numpy as np
import matplotlib.pyplot as plt
import aotools
from LBClass import Waveguide
from MMF2 import *
from telescope import telescope
from phaseScreens import phaseScreen, finitePhaseScreen
from lightbeam.misc import normalize,overlap_nonu,norm_nonu
from lightbeam import LPmodes
from numbaLib import zoom

#
N = 2**10
obscuration = 0.3
r0 = 0.3
L0,l0 = 10,0.00001
D = 0.7
f = 8.41
wvl = 1550e-9
width = 64

def showComplex(E):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(np.angle(E))
    ax[1].imshow(np.abs(E)**2)
    return fig,ax

if __name__ == "__main__":
    rc700 = telescope(wvl, D, f, pupil_size=N,pixels=N,obs=obscuration)
    waveguide = Waveguide(optic, wvl, n0)

    finitePhaseScreenParms = {'r0': r0, 'delta': D/N, 'L0': L0, 'l0': l0, 'subHarmonics': False}
    phaseScreenGenerator = phaseScreen(N, finitePhaseScreen(), finitePhaseScreenParms)
    P = phaseScreenGenerator.generate()

    u0 = np.ones_like(P)*np.exp(1j*P)
    u1 = rc700.propagate(u0)
    # Add a row of zeros to u0 and a row of zeros to u1
    # This is a hack to get the array sizes to match
    # u1 = np.vstack((u1,np.zeros((1,u1.shape[1]),dtype=np.complex128)))
    # u1 = np.hstack((u1,np.zeros((u1.shape[0],1),dtype=np.complex128)))
    u1 = normalize(u1)
    print(f"u1 shape: {u1.shape}")

    # # Cut out center of u1
    ds = rc700.cameraPixelScale
    print(f"ds: {ds}")
    print(f"Pixels In 225x3 {(225*3)/ds}")
    # xw0 = ds*N
    xw0 = ds*width

    # xw0 = 7.5*rclad
    print(f"xw0: {xw0}")
    # print(f"xw0__: {xw0__}")
    
    # Need to figure out how to match the array sizes
    waveguide.createMesh(xw0,zex,ds,dz)
    waveguide.initPropagator()

    xg = waveguide.mesh.xg[waveguide.num_PML:-waveguide.num_PML,waveguide.num_PML:-waveguide.num_PML]
    yg = waveguide.mesh.yg[waveguide.num_PML:-waveguide.num_PML,waveguide.num_PML:-waveguide.num_PML]
    xa = xg[0,:]
    ya = yg[:,0]
    print(f"xg shape: {xg.shape}")
    print(f"yg shape: {yg.shape}")
    print(f"xa shape: {xa.shape}")
    print(f"ya shape: {ya.shape}")
    print(f"====================")
    # Cutout the centre of u1
    u1Zoom = u1[N//2-len(xa)//2:N//2+len(xa)//2+1,N//2-len(ya)//2:N//2+len(ya)//2+1]
    u1Zoom = normalize(u1Zoom)
    # save as np array
    np.save('A.npy',u1)

    u0Fig     = showComplex(u0)
    u1Fig     = showComplex(u1)
    u1ZFig    = showComplex(u1Zoom)

    from lightbeam import LPmodes
    u1Zoom = normalize(LPmodes.lpfield(xg,yg,1,1,rcore,wvl,ncore,nclad) )
    print(f"xa: {xg.shape}, ya: {yg.shape}")
    print(f"u1Zoom shape: {u1Zoom.shape}")
    u1ZFig, u1ZAx = plt.subplots(1,2)
    u1ZAx[0].imshow(np.real(u1Zoom))
    u1ZAx[1].imshow(np.abs(u1Zoom)**2)
    # u1ZoomFig = showComplex(u1Zoom)


    waveguideFig = waveguide.show('front')
    plt.show()
    # u2 = waveguide.propagate(u1)
    u2 = waveguide.propagate(u1Zoom)

    u2Fig = showComplex(u2)
    plt.show()


