import aotools
import numpy as np
import matplotlib.pyplot as plt

#To Do:
# - Enable adding coeffiencent at generation time

class phaseScreen(object):
    # Using Stategy Pattern to generate the phase screens using and 
    # arbitrary generator function. E.g. Zernike, KL, Kolmogorov, etc.
    def __init__(self, N, strategy, generatorParameters):
        self.N = N
        self.generator = strategy
        self.generatorParameters = generatorParameters
        # Placeholder Attributes
        self.scrn = np.zeros((N,N))
    def setParams(self, **kwargs):
        self.generatorParameters = kwargs
    def generate(self, **kwargs):
        self.scrn = self.generator.generate(self.N, **self.generatorParameters)
        return self.scrn
    def decompose(self, type="zernike",nModes=50):
        if type == "zernike":
            mask = aotools.circle(self.N//2,self.N)
            A = mask*self.scrn
            Z = aotools.zernikeArray(nModes, self.N)
            z = np.array([np.trace(Zj.conj().T @ A ) for Zj in Z]) /self.N**2
            return z
        else:
            # Generate Error
            raise ValueError("Decomposition type not recognised")

# Generator Interface
class generatorStrategy(object):
    def generate(self, N, **kwargs):
        pass

class zernike(generatorStrategy):
    def generate(self, N, **kwargs):
        # unpack the parameters
        zernikeModes = kwargs['modes']
        # Create the Zernike phase screen
        screen = np.zeros((N,N))
        for mode, coeff in zernikeModes.items():
            # print(f" mode: {mode} \n coeff: {coeff}")
            screen += aotools.zernike_noll(mode,N)*coeff
        # print(f" screen shape: {screen.shape} \n screen Type: {type(screen)}")
        return screen

class KL(generatorStrategy):
    def generate(self, N, **kwargs):
        # unpack the parameters
        nModes = kwargs['nModes']
        coeff = kwargs['coeff']
        ri = kwargs['ri']
        # Create the KL phase screens
        kl, _, _, _ = aotools.functions.karhunenLoeve.make_kl(nModes, N,ri=ri)
        screen = np.zeros((N,N))
        for mode in range(nModes):
            screen += coeff[mode]*kl[mode]
        return screen

class kolmogorov(generatorStrategy):
    def generate(self, N, **kwargs):
        # unpack the parameters
        r0 = kwargs['r0']
        # Create the Kolmogorov phase screen
        screen = aotools.turbulence.atmosphericScreen(N, r0)
        return screen

class finitePhaseScreen(generatorStrategy):
    def generate(self, N, **kwargs):
        # unpack the parameters
        r0 = kwargs['r0']
        delta = kwargs['delta']
        L0,l0 = kwargs['L0'], kwargs['l0']
        subHarmonics = kwargs['subHarmonics']
        if subHarmonics == True:
            screen = aotools.turbulence.phasescreen.ft_sh_phase_screen(r0, N, delta, L0, l0)
        else:
            screen = aotools.turbulence.phasescreen.ft_phase_screen(r0, N, delta, L0, l0)
        return screen

class PCAModeScreen(generatorStrategy):
    def __init__(self,path):
        
        self.pcScreens = np.zeros((50,128,128))
        for i in range(50):
            self.pcScreens[i] = np.loadtxt(path+f'/pc_{i}.csv', delimiter=',')
        self.spectrum = np.loadtxt(path+'/spectrum.csv', delimiter=',')
    def generate(self, N, **kwargs):
        # check if coeff is in kwargs
        if 'coeff' in kwargs:
            coeff = kwargs['coeff']
        else:
            coeff = np.zeros(50)
            for i,s in enumerate(self.spectrum):
                coeff[i] = np.random.normal(0, np.sqrt(s))
        scrn = np.zeros((N,N))
        for i in range(50):
            scrn+=coeff[i]*self.pcScreens[i]
        return scrn


# Main function
if __name__ == '__main__':
    # Define the parameters
    n = 64
    # zernikeGeneratorParms = {'modes': {2: 1, 3: 1, 4: 1,15:5}}
    # phaseScreenGenerator = phaseScreen(n, zernike(), zernikeGeneratorParms)

    # KLGeneratorParms = {'nModes': 100, 'coeff': np.random.rand(100), 'ri': 0.1}
    # phaseScreenGenerator = phaseScreen(n, KL(), KLGeneratorParms)

    finitePhaseScreenParms = {'r0': 0.1, 'delta': 0.1, 'L0': 100, 'l0': 0.01, 'subHarmonics': True}
    phaseScreenGenerator = phaseScreen(n, finitePhaseScreen(), finitePhaseScreenParms)
    # PCA_Coeff = np.zeros(50)
    # PCA_Coeff[10] = 1
    # PCAGeneratorParms = {'coeff': PCA_Coeff}
    # # PCAGeneratorParms = {}
    # path = './Data/PrincipalComponents'
    
    # phaseScreenGenerator = phaseScreen(n, PCAModeScreen(path),{})
    # phaseScreenGenerator.setParams(**PCAGeneratorParms)
    scrn = phaseScreenGenerator.generate()
    # print(scrn)
    nModes = 128
    z = phaseScreenGenerator.decompose(nModes=nModes)
    print(z)
    # Reconstruction
    
    Z = aotools.zernikeArray(nModes, n)
    Rscrn = np.zeros((n,n))
    for i in range(nModes):
        Rscrn += z[i]*Z[i]
    
    # Print summary stats for both screens
    print(f"Original Screen: \n Mean: {np.mean(scrn)} \n Std: {np.std(scrn)}, \n Max: {np.max(scrn)} \n Min: {np.min(scrn)}")
    print(f"Reconstructed Screen: \n Mean: {np.mean(Rscrn)} \n Std: {np.std(Rscrn)}, \n Max: {np.max(Rscrn)} \n Min: {np.min(Rscrn)}")
    histFigm, histAx = plt.subplots()
    histAx.hist(scrn.flatten(), bins=100, alpha=0.5, label='Original Screen')
    histAx.hist(Rscrn.flatten(), bins=100, alpha=0.5, label='Reconstructed Screen')
    histAx.legend()

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(scrn)
    axs[0].set_title("Original Screen")
    axs[1].imshow(Rscrn)
    axs[1].set_title("Reconstructed Screen") 
    plt.show()


