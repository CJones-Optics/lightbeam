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
import numpy as np
from scipy.interpolate import RegularGridInterpolator





def bilinear_interp(data, xCoords, yCoords, interpArray, bounds_check=True):
    """
    A function which deals with numba interpolation.

    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation
        bounds_check (bool, optional): Do bounds checkign in algorithm? Faster if False, but dangerous! Default is True
    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    if bounds_check:
        bilinear_interp_numba(data, xCoords, yCoords, interpArray)
    else:
        bilinear_interp_numba_inbounds(data, xCoords, yCoords, interpArray)

    return interpArray

@numba.jit(nopython=True, parallel=True)
def bilinear_interp_numba(data, xCoords, yCoords, interpArray):
    """
    2-D interpolation using purely python - fast if compiled with numba
    This version also accepts a parameter specifying how much of the array
    to operate on. This is useful for multi-threading applications.

    Bounds are checks to ensure no out of bounds access is attempted to avoid
    mysterious seg-faults

    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    jRange = range(yCoords.shape[0])
    for i in numba.prange(xCoords.shape[0]):
        x = xCoords[i]
        if x >= data.shape[0] - 1:
            x = data.shape[0] - 1 - 1e-9
        x1 = numba.int32(x)
        for j in jRange:
            y = yCoords[j]
            if y >= data.shape[1] - 1:
                y = data.shape[1] - 1 - 1e-9
            y1 = numba.int32(y)

            xGrad1 = data[x1 + 1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1 * (x - x1)

            xGrad2 = data[x1 + 1, y1 + 1] - data[x1, y1 + 1]
            a2 = data[x1, y1 + 1] + xGrad2 * (x - x1)

            yGrad = a2 - a1
            interpArray[i, j] = a1 + yGrad * (y - y1)
    return interpArray


def zoom(array,newSize):
    x,y = np.arange(array.shape[0]),np.arange(array.shape[1])
    zx = np.linspace(0,array.shape[0]-1,newSize)
    zy = np.linspace(0,array.shape[1]-1,newSize)
    ZX,ZY = np.meshgrid(zx,zy)
    interp = RegularGridInterpolator((x,y),array)
    return interp((ZX,ZY))


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