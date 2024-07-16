''' example script for running the beamprop code in prop.py'''
import numpy as np
from lightbeam.mesh import RectMesh3D
from lightbeam.prop import Prop3D
from lightbeam.misc import normalize,overlap_nonu,norm_nonu
from lightbeam import LPmodes
import matplotlib.pyplot as plt
# from config_example import *
from MMF2 import *

if __name__ == "__main__":
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

    # u0 = normalize(LPmodes.lpfield(xg,yg,0,0,1.5*rclad,wl0,nclad,njack,'cos'))
    u0 =  normalize(LPmodes.lpfield(xg,yg,2,2,1.5*rclad,wl0,nclad,njack,'cos'))
    u0 +=  0.3*normalize(LPmodes.lpfield(xg,yg,2,1,1.5*rclad,wl0,nclad,njack,'cos'))
    u0 +=  1*normalize(LPmodes.lpfield(xg,yg,1,1,1.5*rclad,wl0,nclad,njack,'cos'))


    fplanewidth = 0 # manually reset the width of the input field. set to 0 to match field extent with grid extent.



    # mesh initialization (required)
    mesh = RectMesh3D(xw0,yw0,zw,ds,dz,num_PML,xw_func,yw_func)
    xg,yg = mesh.xy.xg,mesh.xy.yg

    mesh.xy.max_iters = max_remesh_iters
    mesh.sigma_max = sig_max

    # propagator initialization (required)
    prop = Prop3D(wl0,mesh,optic,n0)

    print('launch field')
    plt.imshow(np.real(u0))
    plt.show()

    out = np.zeros( mesh.xy.shape )
    print(f"Out shape: {out.shape}")  
    print(f"u0 shape: {u0.shape}")
    optic.set_IORsq(out,1)
    plt.imshow(out,vmin=njack*njack,vmax=ncore*ncore)
    plt.show()




    print(f"Out shape: {out.shape}")  
    print(f"u0 shape: {u0.shape}")
    print(f"Mesh shape: {mesh.xy.shape}")
    

    # run the propagator (required)
    u,u0 = prop.prop2end(u0,monitor_func=monitor_func,xyslice=None,zslice=None,writeto=writeto,ref_val=ref_val,remesh_every=remesh_every,dynamic_n0=dynamic_n0,fplanewidth=fplanewidth)

    # compute power in output ports (optional)

    xg,yg = np.meshgrid(mesh.xy.xa,mesh.xy.ya,indexing='ij')

    w = mesh.xy.get_weights()

    xg0,yg0 = np.meshgrid(mesh.xy.xa0,mesh.xy.ya0,indexing='ij')
    w0 = mesh.xy.dx0*mesh.xy.dy0

    modes = []
    for x,y in zip(xpos,ypos):
        mode = norm_nonu(LPmodes.lpfield(xg-x,yg-y,0,1,rcore/scale,wl0,ncore,nclad),w)
        modes.append(mode)
    
    SMFpower=0
    print("final field power decomposition:")
    for i in range(len(modes)):
        _p = np.power(overlap_nonu(u,modes[i],w),2)
        print("mode"+str(i)+": ", _p)
        SMFpower += _p
    
    print("total power in SMFs: ", SMFpower)

    # plotting (optional)
    print("final field dist:")
    plt.imshow(np.abs(u0)[num_PML:-num_PML,num_PML:-num_PML]) 
    plt.show()