
import numpy as np
################################
## free space wavelength (um) ##
################################

# wl0 = 1.55
wl0 = 1.020

########################
## lantern parameters ##
########################

zex = 30000 # length of lantern, in um
scale = 1 # how much smaller the input end is wrt the output end
rcore = 4.5 # how large the lantern cores are, at the input (um)
rclad = 16.5 # how large the lantern cladding is, at the input (um)
# rjack = 500
ncore = 1.451 # lantern core refractive index
nclad = 1.391  # cladding index
njack = 1.4504 - 5.5e-3 # jacket index

###################################
## sampling grid parameters (um) ##
###################################

xw0 = 128 # simulation zone x width (um)
yw0 = xw0 # simulation zone y width (um)
zw = zex
ds = 1 # base grid resolution (um)
dz = 3 # z stepping resolution (um)

#############################
## mesh refinement options ##
#############################

ref_val = 1e-4 # controls adaptive meshing. lower -> more careful
remesh_every = 50 # how many z-steps to go before recomputing the adaptive mesh
max_remesh_iters = 6 # maximum amount of subdivisions when computing the adaptive mesh

xw_func = None # optional functions which allow the simulation zone to "grow" with z, which may save on computation time
yw_func = None

##################
## PML settings ##
##################

num_PML = 12
sig_max = 3. + 0.j


######################
## set launch field ##
######################
import numpy as np
import matplotlib.pyplot as plt
from lightbeam import LPmodes
from lightbeam.misc import normalize

xa = np.linspace(-xw0/2,xw0/2,int(xw0/ds)+1)
ya = np.linspace(-yw0/2,yw0/2,int(yw0/ds)+1)
xg,yg  = np.meshgrid(xa,ya)

u0 = normalize(LPmodes.lpfield(xg,yg,2,1,rclad,wl0,nclad,njack,'cos'))

fplanewidth = 0 # manually reset the width of the input field. set to 0 to match field extent with grid extent.

#####################
## reference index ##
#####################

n0 = 1.4504 
dynamic_n0 = False

###################
## monitor field ##
###################

monitor_func = None

#############################
## write out field dist to ##
#############################

writeto = None

from lightbeam import optics

class MMF(optics.OpticSys):
    def __init__(self,rcore,rclad,ncore,nclad,njack,offset0,z_ex,nb=1):
        core = optics.scaled_cyl([0,0],rcore,z_ex,ncore,nclad,final_scale=1)
        clad = optics.scaled_cyl([0,0],rclad,z_ex,nclad,njack,final_scale=1)
        # jack = optics.scaled_cyl([0,0],rjack,z_ex,njack,nb,final_scale=1)
        elements = [clad,core]
        super().__init__(elements,njack)
        self.core_locs = np.array([[0,0]])

optic = MMF(rcore,rclad,ncore,nclad,njack,0,zex)

class MMFBundleSubFiber():
    def __init__(self,pos,rcore,rclad,ncore,nclad,njack,offset0,z_ex,nb=1):
        core = optics.scaled_cyl(pos,rcore,z_ex,ncore,nclad,final_scale=1)
        clad = optics.scaled_cyl(pos,rclad,z_ex,nclad,njack,final_scale=1)
        elements = [clad,core]
        self.pos = pos
        self.elements = elements

class MMFBundle(optics.OpticSys):
    def __init__(self,rcore,rclad,ncore,nclad,njack,offset0,z_ex,nb=1):
        self.subfibers = []
        self.r = 2*rclad
        for pos in self.get_core_locs():
            self.subfibers.append(MMFBundleSubFiber(pos,rcore,rclad,ncore,nclad,njack,offset0,z_ex,nb=1))
        # System elements
        self.elements = []
        for subfiber in self.subfibers:
            self.elements += subfiber.elements
        super().__init__(self.elements,njack)
        
    
    def get_core_locs(self):
        pos=[(0,0)]
        deltaTheta = 2*np.pi/6
        for i in range(1,7):
            x = self.r*np.cos(i*deltaTheta)
            y = self.r*np.sin(i*deltaTheta)
            pos.append((x,y))
        self.core_locs = np.array(pos)
        return self.core_locs
            
            
        




 
#######################
## initial core locs ##
#######################

xpos_i = optic.core_locs[:,0]
ypos_i = optic.core_locs[:,1]

#####################
## final core locs ##
#####################

xpos = xpos_i / scale
ypos = ypos_i / scale

