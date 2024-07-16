import aotools
import numpy as np
import matplotlib.pyplot as plt
from numbaLib import bilinear_interp,zoom

class telescope(object):
    def __init__(self, wvl, d, fov, N_p,N_f,simPad=1,FFTOversamp=2, obs=0 ):
        from numbaLib import bilinear_interp, zoom

        self.wvl = wvl
        self.d = d
        self.fov = fov
        self.N_p = N_p
        self.N_f = N_f
        self.simPad = simPad
        self.obscuration = obs
        self.FFTOversamp = FFTOversamp

        self.sim_size  = self.N_p + 2*self.simPad
        self.fovPixNum = (np.round(self.d*self.fov/self.wvl)).astype(int)

        self.FFTPadding = N_f * self.FFTOversamp
        if self.FFTPadding < self.fovPixNum:
            print(f"Increasing FFTPadding")
            print(f"Old FFTPadding: {self.FFTPadding}")
            while self.FFTPadding < self.fovPixNum:
                self.FFTOversamp += 1
                self.FFTPadding = self.FFTOversamp*N_f
            print(f"New FFTPadding: {self.FFTPadding}")
        
        self.FFTInput = np.zeros((self.FFTPadding,self.FFTPadding),dtype=np.complex128)
        self.interpCoords = np.linspace(self.simPad,self.simPad+self.N_p,self.fovPixNum)

        self.getMask()
        self.effective_focal_length()
        self.ppScale()
        self.fpScale()

        self.method = 'interpolated'
    
    def getMask(self):
        N = self.N_p
        mask = aotools.circle(N/2,N) 
        if self.obscuration > 0: # Only bother with the central obscuration if it is non-zero
            mask -= aotools.circle(N/2*self.obscuration,N)
        self.mask = mask
        self.scaledMask = zoom(mask,self.fovPixNum) # This is the mask for the FOV
        return mask
    
    def effective_focal_length(self):
        self.eff = self.d/(2*np.tan(self.fov/2) )
        return self.eff
    def ppScale(self):
        self.ppScale = self.d / self.N_p
        return self.ppScale
    def fpScale(self):
        self.fpScale = (self.wvl*self.eff ) / (self.d*self.N_f)
        return self.fpScale
    


    def interp(self,P,coords):
        # Wrapper function to improve the ergonomics of the bilinear_interp function
        interpArray = np.zeros((coords.size,coords.size),dtype=np.complex128)
        bilinear_interp(P,coords,coords,interpArray)
        return interpArray

    def propagate(self,u0):
        u0 = self.mask*u0

        # These arn't being used, leaving in just in case
        # if self.method == 'lensAgainst':
        #     u1 = aotools.opticalpropagation.lensAgainst(u0, self.wvl, self.d1, self.f)
        # if self.method == 'fft':
        #     u1 = np.fft.fftshift( np.fft.fft2(u0) )

        if self.method == 'interpolated':
            # Grab the phase from the input field
            P = np.angle(u0)
            # Extract the FOV region of iterest
            phaseInterp = self.interp(P,self.interpCoords)
            E_fov = np.exp(1j*phaseInterp)*self.scaledMask
            # Reset the FFTInput to zero's just incase it isnt
            self.FFTInput *= 0
            # Set the Region of interest the the FOV, leaving the rest as zero padding
            self.FFTInput[:self.fovPixNum,:self.fovPixNum] = E_fov
            # Use FFT to transform the FOV to the focal plane
            E_focal = np.fft.fftshift(np.fft.fft2(self.FFTInput))
            # Resize to self.N_f
            u1 = np.zeros((self.N_f,self.N_f),dtype=np.complex128)
            u1 = bin_img(E_focal,self.FFTOversamp, u1)
        return u1
    
    def getPupil(self,u0):
        self.pupil = self.mask*u0
        return self.pupil
    def getFocus(self,u0):
        self.focus = self.propagate(u0)
        return self.focus
    
    def showPupilAndFocus(self,u0,show=True):
        def calculate_extent(array, ds):
            ny, nx = array.shape
            extent = [-nx/2 * ds, nx/2 * ds, -ny/2 * ds, ny/2 * ds]
            return extent
        ds = self.fpScale * 1e6 # [um]
        if self.pupil is None:
            self.getPupil(u0)
        if self.focus is None:
            self.getFocus(u0)
        fig, ax = plt.subplots(2,2)
        ax[0,0].imshow(np.angle(self.pupil),  cmap='bwr', extent=calculate_extent(self.pupil, self.ppScale*1e3))
        ax[1,0].imshow(np.abs(self.pupil)**2, cmap='hot', extent=calculate_extent(self.pupil, self.ppScale*1e3))
        ax[0,1].imshow(np.angle(self.focus),  cmap='bwr', extent=calculate_extent(self.focus, self.fpScale*1e6))
        ax[1,1].imshow(np.abs(self.focus)**2, cmap='hot', extent=calculate_extent(self.focus, self.fpScale*1e6))

        ax[0,0].set_title('Pupil')
        ax[0,1].set_title('Focus')

        ax[1,0].set_xlabel('mm')
        ax[1,1].set_xlabel('um')

        if show:
            plt.show()
        return




# Image binning algorithm

import multiprocessing
N_CPU = multiprocessing.cpu_count()
from threading import Thread

# python3 has queue, python2 has Queue
try:
    import queue
except ImportError:
    import Queue as queue

import numpy
import numba

def bin_img(input_img, bin_size, binned_img, threads=None):
    N_CPU = 4
    if threads is None:
        threads = N_CPU

    n_rows = binned_img.shape[0]

    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=bin_img_numba,
                         args=(
                             input_img, bin_size, binned_img,
                             numpy.array([int(t * n_rows / threads), int((t + 1) * n_rows / threads)]),
                         )
                         ))
        Ts[t].start()

    for T in Ts:
        T.join()

    return binned_img

@numba.jit(nopython=True, nogil=True)
def bin_img_numba(imgs, bin_size, new_img, row_indices):
    # loop over each element in new array
    for i in range(row_indices[0], row_indices[1]):
        x1 = i * bin_size

        for j in range(new_img.shape[1]):
            y1 = j * bin_size
            new_img[i, j] = 0

            # loop over the values to sum
            for x in range(bin_size):
                for y in range(bin_size):
                    new_img[i, j] += imgs[x1 + x, y1 + y]





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
    
