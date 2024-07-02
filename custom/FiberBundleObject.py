from lightbeam.optics import *

class lant5(OpticSys):
    '''corrigan et al. 2018 style photonic lantern'''
    def __init__(self,rcore,rclad,rjack,ncore,nclad,njack,offset0,z_ex,scale_func=None,final_scale=1,nb=1):
        core0 = scaled_cyl([0,0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        core1 = scaled_cyl([offset0,0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        core2 = scaled_cyl([0,offset0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        core3 = scaled_cyl([-offset0,0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        core4 = scaled_cyl([0,-offset0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        clad = scaled_cyl([0,0],rclad,z_ex,nclad,njack,scale_func=scale_func,final_scale=final_scale)
        jack = scaled_cyl([0,0],rjack,z_ex,njack,nb,scale_func=scale_func,final_scale=final_scale)
        elmnts = [jack,clad,core4,core3,core2,core1,core0]
        
        super().__init__(elmnts,nb)


class translated_cyl(OpticPrim):
    """
    Cylinder whose radius is constant, but the center is translated in the x-y plane.
    """
    def __init__(self,xy0,xy1,r,z_ex,n,nb,z_offset=0):
        super().__init__(n)
        self.p1 = [xy0[0],xy0[1],z_offset]
        self.p2 = [xy1[0],xy1[1],z_ex+z_offset]

        self.r = r
        self.rsq = r*r
        self.nb2 = nb*nb
        self.n2 = n*n
        self.z_ex = z_ex
        self.z_offset = z_offset
        
        self.AA = True


    def _contains(self,x,y,z):
        if not (self.z_offset <= z <= self.z_offset+self.z_ex):
            return False
        xdist = x - self.p1[0]
        ydist = y - self.p1[1]
        return (xdist*xdist + ydist*ydist <= self.rsq)

    def _bbox(self,z):
        xc = self.p1[0]
        yc = self.p1[1]
        xmax = xc+self.r
        xmin = xc-self.r
        ymax = yc+self.r
        ymin = yc-self.r
        return (xmin,xmax,ymin,ymax)

    def set_IORsq(self,out,z,coeff=1):
        if not (self.z_offset <= z <= self.z_offset+self.z_ex):
            return
        bbox,bboxh = self.bbox_idx(z)
        xg = self.xymesh.xg[bbox]
        yg = self.xymesh.yg[bbox]
        mask = self._contains(xg,yg,z)
        out[bbox][mask] = self.n2*coeff


class sevenCoreBundle(OpticSys):
    """
    Representing an optic fiber bundle
    """
    def __init__(self, rCore, rClad, rJack,nCore,nClad,nJack,z_ex,final_scale=1,nb=1):
        elements = []
        core0 = scaled_cyl([0,0],rCore,z_ex,nCore,nClad,final_scale=final_scale)
        clad0 = scaled_cyl([0,0],rClad,z_ex,nClad,nJack,final_scale=final_scale)
        elements += [clad0,core0]

        angle = 2*np.pi/6
        offset = 2.2 * rClad
        for i in range(6):
            x,y = offset*np.cos(i*angle), offset*np.sin(i*angle)
            core = scaled_cyl([x,y],rCore,z_ex,nCore,nClad,final_scale=final_scale)
            clad = scaled_cyl([x,y],rClad,z_ex,nClad,nJack,final_scale=final_scale)
            elements += [clad,core]

        super().__init__(elements,nb)
        
# If main
if __name__ == "__main__":
    import matplotlib.pyplot as plt


########################
## lantern parameters ##
########################

    zex = 30000 # length of lantern, in um
    scale = 1/4 # how much smaller the input end is wrt the output end
    rcore = 4.5 * scale # how large the lantern cores are, at the input (um)
    rclad = 16.5 # how large the lantern cladding is, at the input (um)
    rJack = 16.5*2*3
    ncore = 1.4504 + 0.0088 # lantern core refractive index
    nclad = 1.4504 # cladding index
    njack = 1.4504 - 5.5e-3 # jacket index

    # Test the fiber bundle
    optic = sevenCoreBundle(rcore,rclad,rJack,ncore,nclad,njack,zex,final_scale=0.25 )

    ###################################
    ## sampling grid parameters (um) ##
    ###################################

    xw0 = 128 # simulation zone x width (um)
    yw0 = 128 # simulation zone y width (um)
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

    from lightbeam.mesh import RectMesh3D
    from lightbeam.prop import Prop3D
    from lightbeam.misc import normalize,overlap_nonu,norm_nonu
    from lightbeam import LPmodes

    num_PML = 12
    sig_max = 3. + 0.j
    wl0 = 1.55
    n0 = 1.4504 


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



    # mesh initialization (required)
    mesh = RectMesh3D(xw0,yw0,zw,ds,dz,num_PML,xw_func,yw_func)
    xg,yg = mesh.xy.xg,mesh.xy.yg

    mesh.xy.max_iters = max_remesh_iters
    mesh.sigma_max = sig_max

    # propagator initialization (required)
    prop = Prop3D(wl0,mesh,optic,n0)

    fig, ax = plt.subplots(1,2)
    out = np.zeros(mesh.xy.shape)
    # See input 
    optic.set_IORsq(out,0)
    ax[0].imshow(out)

    out = np.zeros(mesh.xy.shape)
    optic.set_IORsq(out,30000)
    ax[1].imshow(out)


    plt.show()


