import lightbeam
import numpy as np
import matplotlib.pyplot as plt
from lightbeam import optics


# Create a class for waveguide
class Waveguide(object):
    def __init__(self, waveguide, wl,n0):
        """
        waveguide: Lightbeam.optics.OpticSys
        mesh: Lightbeam.mesh.RectMesh3D
        wl: float, wavelength
        u0: np.ndarray(shape=(mesh.xy.shape), dtype=np.complex128), input E-field
        n0: float, background index
        
        """
        self.waveguide = waveguide
        self.n0 = n0
        self.wl0 = wl
        self.num_PML = 12

        self.monitor_func = None
        self.writeto = None
        self.ref_val = 1e-4
        self.remesh_every = 50
        self.dynamic_n0 = False
        self.fplanewidth = 0
    
    def createMesh(self,xw0, zw,ds,dz):
        yw0 = xw0
        sig_max =  3. + 0.j
        max_remesh_iters = 6
        xw_func = None
        yw_func = None
        mesh = lightbeam.mesh.RectMesh3D(xw0,yw0,zw,ds,dz,self.num_PML,xw_func,yw_func)
        mesh.xy.max_iters = max_remesh_iters
        mesh.sigma_max = sig_max

        # cyl.xymesh = _mesh.xy # pass in the mesh
        self.waveguide.xymesh = mesh.xy
        self.waveguide.set_sampling(mesh.xy)

        print(f"Mesh Shape: {mesh.xy.shape}")
        self.mesh = mesh

    def initPropagator(self):
        from lightbeam.prop import Prop3D
        self.prop = Prop3D(self.wl0, self.mesh, self.waveguide,self.n0)
    def propagate(self,u0):
        print('launch field')
        # if u0.shape != self.mesh.xy.shape:
        #     print(f"u0 shape: {u0.shape}")
        #     # Pad u0
        #     inputField = np.zeros((self.mesh.xy.shape[0]+1,self.mesh.xy.shape[1]+1) , dtype=np.complex128)
        #     print(f"inputField shape: {inputField.shape}")
        #     # It would be better to center it, but who cares
        #     inputField[:u0.shape[0],:u0.shape[1]] = u0
        #     u0 = inputField
        #     print(f"u0 shape: {u0.shape}")

        # try:
        print(f"u0 shape: {u0.shape}")
        print(f"mesh.xy.shape: {self.mesh.xy.shape}")

        # u,u0 = self.prop.prop2end(u0)
        u = self.prop.prop2end_uniform(u0)
        # except Exception as e:
        #     print(e)
        #     print(f"u0.shape: {u0.shape}")
        #     print(f"mesh.xy.shape: {self.mesh.xy.shape}")
        #     return None

        self.u1 = u
        return u

    def show(self,kward):
        out = np.zeros( self.mesh.xy.shape )
        if kward == 'front':
            print(f"Showing the front")
            self.waveguide.set_IORsq(out,10)
        elif kward == 'back':
            print(f"Showing the back")
            self.waveguide.set_IORsq(out,self.waveguide.z)
        else:
            print('Invalid keyword')
            return None

        vmin = self.waveguide.njack*self.waveguide.njack
        vmax = self.waveguide.ncore*self.waveguide.ncore
        fig, ax = plt.subplots()
        ax.imshow(out,vmin=vmin,vmax=vmax)
        # plt.show()           
        return fig,ax




class MMFBundle(optics.OpticSys):
    def __init__(self,rcore,rclad,ncore,nclad,njack,offset0,z_ex,nb=1):
        
        r = 2*rclad
        t = 2*np.pi/6
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

