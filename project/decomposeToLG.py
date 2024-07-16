import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from LBClass import Waveguide
from lightbeam.optics import *
from lightbeam.misc import normalize,overlap_nonu,norm_nonu
from lightbeam import LPmodes
from lightbeam.mesh import RectMesh3D
from lightbeam.prop import Prop3D
# from MMF2 import *
import lightbeam.LPmodes as LPmodes
from lightbeam.misc import normalize
import aotools

################################
## free space wavelength (um) ##
################################
wl0 = 1.55
#############################
## mesh refinement options ##
#############################
ref_val = 1e-4       # controls adaptive meshing. lower -> more careful
remesh_every = 50    # how many z-steps to go before recomputing the adaptive mesh
max_remesh_iters = 6 # maximum amount of subdivisions when computing the adaptive mesh
xw_func = None       # optional functions which allow the simulation zone to "grow" with z, which may save on computation time
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

from lightbeam.optics import *
class SMF(OpticSys):
    def __init__(self,pos=[0,0], z_ex=20000):
        self.rcore = 3
        self.rclad = 6

        self.ncore = 1.4504 + 0.0088
        self.nclad = 1.4504
        self.njack = 1.4504 - 5.5e-3

        self.offset0 = 0
        self.z_ex = z_ex
       
        core = scaled_cyl(pos,self.rcore,z_ex,self.ncore,self.nclad)
        clad = scaled_cyl(pos,self.rclad,z_ex,self.nclad,self.njack)

        self.core = core
        self.clad = clad

        elmnts = [clad,core]
        self.elements = elmnts
        super().__init__(elmnts,self.njack)

class fiber(OpticSys):
    def __init__(self,pos=[0,0],rcore=3,ncore = 1.4504 + 0.0088,rclad=6,nclad=1.4504,njack=1.4504 - 5.5e-3,z_ex=20000):
        core = scaled_cyl(pos,rcore,z_ex,ncore,nclad)
        clad = scaled_cyl(pos,rclad,z_ex,nclad,njack)
        elements = [clad,core]
        self.elements = elements
        super().__init__(elements,njack)

class fiberBundle(OpticSys):
    def __init__(self,rcore=3,ncore = 1.4504 + 0.0088,rclad=6,nclad=1.4504,njack=1.4504 - 5.5e-3,z_ex=20000):
        self.elements = []
        self.coreLocs = []
        self.rcore = rcore
        self.rclad = rclad
        
        self.ncore = ncore
        self.nclad = nclad
        self.njack = njack

        self.z     = z_ex

        r = 2*self.rclad
        phase = [i*(np.pi*2)/6 for i in range(6)]
        self.elements += fiber([0,0],rcore,ncore,rclad,nclad,njack,z_ex).elements

        for p in phase:
            x = r*np.cos(p)
            y = r*np.sin(p)
            self.elements += fiber([x,y],rcore,ncore,rclad,nclad,njack,z_ex).elements
            self.coreLocs.append(np.array([x,y]))
        
        super().__init__(self.elements,self.njack)



class SMFBundle(OpticSys):
    def __init__(self,z):
        self.elements = []
        self.rcore = 8
        self.rclad = 125

        self.ncore = 1.4504 + 0.0088
        self.nclad = 1.4504
        self.njack = 1.4504 - 5.5e-3


        r = 2*self.rclad
        phase = [i*(np.pi*2)/6 for i in range(6)]
        self.elements += SMF([0,0],z).elements
        for p in phase:
            x = r*np.cos(p)
            y = r*np.sin(p)
            print(f"x: {x}, y: {y}")
            # self.elements += [core,clad]o
            self.elements += SMF([x,y],z).elements
        super().__init__(self.elements,self.njack)


#### Mesh ####

from lightbeam.mesh import RectMesh3D

xw = 225*3 #um
yw = xw
# yw = 512 #um
zw = 10000 #um
num_PML = 16 #grid units

# ds = 0.25 #um
ds = 4 #um
dz = 4 #um

_mesh = RectMesh3D(xw,yw,zw,ds,dz,num_PML)
xg, yg = _mesh.xg[num_PML:-num_PML,num_PML:-num_PML],_mesh.yg[num_PML:-num_PML,num_PML:-num_PML]

#### Optic (SMF) ####
smf = SMF(z_ex=zw)
# smfBundle = SMFBundle(zw)
smfBundle = fiberBundle(rcore=200/2,
                        rclad=225/2, 
                        z_ex=zw)


smfBundle.set_sampling(_mesh.xy) # OpticSys objects have a function that can send a RectMesh2D object to all contianed primitives,
                            # in order to set the sampling

out = np.zeros(_mesh.xy.shape)
smfBundle.set_IORsq(out,0)
plt.imshow(out,vmin=smfBundle.njack*smfBundle.njack,vmax=smfBundle.ncore*smfBundle.ncore)
plt.show()

#### Launch field ####
u0 = normalize(LPmodes.lpfield(xg,yg,
                               10,7,
                               2*smfBundle.rclad,
                            #    smfBundle.rcore,
                               wl0,
                               smfBundle.ncore,
                               smfBundle.nclad))
# u0 = np.roll(u0, 50, axis=0)

# print()
# u0 = normalize(aotools.circle(smfBundle.rcore/ds, xg.shape[0], [0,50] ))
# print(f"smf Core Locs:{smfBundle.coreLocs}")
# print(f"smf core locs: {np.round(smfBundle.coreLocs[1]+[xw/2,yw/2])}")

print(f"u1.shape: {u0.shape}")
plt.imshow(np.abs(u0)**2)
plt.show()


print(f"u0.shape: {u0.shape}")

#### propagation ####
prop = Prop3D(wl0,
              _mesh,
              smfBundle,
              smf.nclad)

# u, u0 = prop.prop2end(u0)
u = prop.prop2end_uniform(u0)

print("output field: ")
outFig, outAx = plt.subplots(2,2)
outAx[0,0].imshow(np.abs(u)**2)
outAx[0,1].imshow(np.angle(u))
outAx[1,0].imshow(np.abs(u0)**2)
outAx[1,1].imshow(np.angle(u0))

plt.show()