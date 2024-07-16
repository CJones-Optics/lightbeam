import aotools
import numpy as np
import matplotlib.pyplot as plt
from numbaLib import bilinear_interp,zoom

class telescope(object):
    def __init__(self, wvl, d1, f, pupil_size=100, pixels=256,obs=0.3):
        self.wvl = wvl
        self.d1 = d1
        self.f = f
        self.fov = 2*np.arctan(self.d1/(2*self.f))
        self.pupil_size = pupil_size
        self.pixels = pixels
        self.obscuration = obs

        self.fovPerPix = self.fov / self.pupil_size
        self.fovPixNum = int(round(self.d1 * self.fovPerPix / self.wvl))

        self.cameraPixelScale = d1/pixels/self.fovPixNum*1e6

        print(f"zoomed Pixel Scale: {self.cameraPixelScale} um")

        self.method = 'interpolated'
    
    def getMask(self,u0):
        N = u0.shape[0]
        mask = aotools.circle(N/2,N) - aotools.circle(N/2*self.obscuration,N)
        self.mask = mask
        return mask

    def propagate(self,u0):
        u0 = self.getMask(u0)*u0

        if self.method == 'lensAgainst':
            u1 = aotools.opticalpropagation.lensAgainst(u0, self.wvl, self.d1, self.f)
        if self.method == 'fft':
            u1 = np.fft.fftshift( np.fft.fft2(u0) )
        if self.method == 'interpolated':
            P = np.angle(u0)

            # I am not sure which of the following is correct:
            # fovPerPix = self.fov / self.pixels
            # fovPerPix = self.fov / self.pupil_size
            # fovPixNum = int(round(self.d1 * fovPerPix / self.wvl))

            scaledMask = zoom(self.mask, self.fovPixNum)
    
            interp_coord = np.linspace(0, self.pupil_size, self.fovPixNum).astype(np.float32)

            interpArray  = np.zeros((self.fovPixNum,self.fovPixNum),dtype=np.complex64)
            interpPhase  = bilinear_interp(P, interp_coord, interp_coord, interpArray, bounds_check=True)

            E_pupil = np.zeros( (self.pixels,self.pixels), dtype=np.complex64 )
            E_pupil[:self.fovPixNum,:self.fovPixNum] = np.exp(1j*interpPhase)*scaledMask
            u1 = np.fft.fftshift( np.fft.fft2(E_pupil) )
        return u1

if __name__ == "__main__":
    #---------------------------
    wvl = 1550e-9 # m
    N = 2**10 
    D = 0.7 # m
    obscuration = 0.3
    d1 = D/N # m
    f = 8.41
    pixels = 2**8
    #---------------------------
    simPad = 0
    pupil_size = N
    t = telescope(wvl, D, f, pupil_size=pupil_size,pixels=pixels)
    #---------------------------
    r0 = 0.10
    L0,l0 = 10,0.00001
    P = aotools.turbulence.phasescreen.ft_sh_phase_screen(r0, N, d1, L0, l0, FFT=None, seed=None)
    #---------------------------
    mask = aotools.circle(N/2,N) - aotools.circle(N/2*obscuration,N)
    u0 = mask*np.exp(1j*P)
    E_focal = t.propagate(u0)   
    #---------------------------
    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(np.angle(u0))
    ax[0,1].imshow(np.abs(u0)**2)
    ax[1,0].imshow(np.angle(E_focal))
    ax[1,1].imshow(np.abs(E_focal)**2)
    plt.show()
    #---------------------------
    
