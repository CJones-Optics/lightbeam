''' example configuration file for run_bpm_example.py '''
import numpy as np
################################
## free space wavelength (um) ##
################################

wl0 = 1.55

########################
## lantern parameters ##
########################

zex = 20000 # length of lantern, in um
scale = 1 # how much smaller the input end is wrt the output end

# Based on Thorlabs - FT200EMT
rcore = 200 * scale # how large the lantern cores are, at the input (um)
rclad = 225 # how large the lantern cladding is, at the input (um)

# # Based on Thorlabs - FT600EMT
# rcore = 600
# rclad = 630

ncore = 1.4504 + 0.0088 # lantern core refractive index
nclad = 1.4504 # cladding index
njack = 1.4504 - 5.5e-3 # jacket index

###################################
## sampling grid parameters (um) ##
###################################

xw0 = 7*512 # simulation zone x width (um)
yw0 = xw0
# yw0 = 3*512 # simulation zone y width (um)
zw = zex
ds = 64 # base grid resolution (um)
dz = 64 # z stepping resolution (um)

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

# generate optical element
from lightbeam import optics
# optic = optics.lant19(rcore,rclad,ncore,nclad,njack,rclad/3,zex,final_scale=1/scale)

class MMF(optics.OpticSys):
    def __init__(self,rcore,rclad,ncore,nclad,njack,offset0,z_ex,nb=1):
        core = optics.scaled_cyl([0,0],rcore,z_ex,ncore,nclad,final_scale=1)
        clad = optics.scaled_cyl([0,0],rclad,z_ex,nclad,njack,final_scale=1)
        # jack = optics.scaled_cyl([0,0],rjack,z_ex,njack,nb,final_scale=1)
        elements = [clad,core]
        super().__init__(elements,njack)
        self.core_locs = np.array([[0,0]])



class MMFBundleSubFiber():
    def __init__(self,pos,rcore,rclad,ncore,nclad,njack,offset0,z_ex,nb=1):
        core = optics.scaled_cyl(pos,rcore,z_ex,ncore,nclad,final_scale=1)
        clad = optics.scaled_cyl(pos,rclad,z_ex,nclad,njack,final_scale=1)
        elements = [clad,core]
        self.pos = pos
        self.elements = elements

class MMFBundle(optics.OpticSys):
    def __init__(self,rcore,rclad,ncore,nclad,njack,offset0,z_ex,nb=1):
        r = 2*rclad
        t = 2*np.pi/6
        self.z = z_ex
        self.ncore = ncore
        self.nclad = nclad
        self.njack = njack
        pos0 = [0,0]
        core0 = optics.scaled_cyl(pos0,rcore,z_ex,ncore,nclad,final_scale=1)
        clad0 = optics.scaled_cyl(pos0,rclad,z_ex,nclad,njack,final_scale=1)
        pos1 = [r*np.cos(t),r*np.sin(t)]
        core1 = optics.scaled_cyl(pos1,rcore,z_ex,ncore,nclad,final_scale=1)
        clad1 = optics.scaled_cyl(pos1,rclad,z_ex,nclad,njack,final_scale=1)
        pos2 = [r*np.cos(2*t),r*np.sin(2*t)]
        core2 = optics.scaled_cyl(pos2,rcore,z_ex,ncore,nclad,final_scale=1)
        clad2 = optics.scaled_cyl(pos2,rclad,z_ex,nclad,njack,final_scale=1)
        pos3 = [r*np.cos(3*t),r*np.sin(3*t)]
        core3 = optics.scaled_cyl(pos3,rcore,z_ex,ncore,nclad,final_scale=1)
        clad3 = optics.scaled_cyl(pos3,rclad,z_ex,nclad,njack,final_scale=1)
        pos4 = [r*np.cos(4*t),r*np.sin(4*t)]
        core4 = optics.scaled_cyl(pos4,rcore,z_ex,ncore,nclad,final_scale=1)
        clad4 = optics.scaled_cyl(pos4,rclad,z_ex,nclad,njack,final_scale=1)
        pos5 = [r*np.cos(5*t),r*np.sin(5*t)]
        core5 = optics.scaled_cyl(pos5,rcore,z_ex,ncore,nclad,final_scale=1)
        clad5 = optics.scaled_cyl(pos5,rclad,z_ex,nclad,njack,final_scale=1)
        pos6 = [r*np.cos(6*t),r*np.sin(6*t)]
        core6 = optics.scaled_cyl(pos6,rcore,z_ex,ncore,nclad,final_scale=1)
        clad6 = optics.scaled_cyl(pos6,rclad,z_ex,nclad,njack,final_scale=1)
        elements = [clad0,core0,clad1,core1,clad2,core2,clad3,core3,clad4,core4,clad5,core5,clad6,core6]
        # elements = [clad0,core0]
        super().__init__(elements,njack)
        self.core_locs = np.array([pos0,pos1,pos2,pos3,pos4,pos5,pos6])


# optic = MMF(rcore,rclad,ncore,nclad,njack,0,zex)
optic = MMFBundle(rcore,rclad,ncore,nclad,njack,0,zex, njack)


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


