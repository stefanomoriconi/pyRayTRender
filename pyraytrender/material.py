import numpy as np

class Material():
    # Material defined with the Blinn-Phong Model
    def __init__(self, ambRGB=[0.1, 0.0, 0.0], dffRGB=[0.7, 0.0, 0.0], spcRGB=[1.0, 1.0, 1.0], shnC=16.0, rflC=0.0, alpha=1.0, MaxDepth=1):
        self.ambRGB = ambRGB # Ambient Colour RGB
        self.dffRGB = dffRGB # Diffuse Colour RGB
        self.spcRGB = spcRGB # Specular Colour RGB (of incident Light)
        self.shnC = shnC # Glossiness/Shininess Coefficient [1, 100]
        self.rflC = rflC # Reflection Coefficient (Matte:0.0, Mirror -> 1.0)
        self.MaxDepth = MaxDepth  # Number of light bounces (Matte:[0,3], Mirror:[10,15])
        self.alpha = alpha # Transparency Index [0, 1] with 0: Transparent, 1: Fully Solid Color
        
        self._iniMat()
        # A Mirror Material usually has:
            # Ambient RGB: [51, 51, 51]/255 = [.2, .2, .2]
            # Diffuse RGB: [15, 15, 15]/255 = [.02, .02, .02]
            # Specular RGB = [215, 215, 215]/255 = [.85, .85, .85] 
            # Shine Coef: 0.996
            # Reflect Coef: 1.0
            # MaxDepth: 10
            
    def _iniMat(self):
        self.ambRGB = np.array(self.ambRGB) 
        self.dffRGB = np.array(self.dffRGB) 
        self.spcRGB = np.array(self.spcRGB) 
        self.shnC = np.array(self.shnC)
        self.rflC = np.array(self.rflC) 
        self.MaxDepth = np.array(self.MaxDepth)

    def resetMat(self):
        self.ambRGB = [0.1, 0.0, 0.0]
        self.dffRGB = [0.7, 0.0, 0.0] 
        self.spcRGB = [1.0, 1.0, 1.0] 
        self.shnC = 16.0 
        self.rflC = 0.0 
        self.MaxDepth = 1
        
        self._iniMat()
