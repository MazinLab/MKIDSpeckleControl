import numpy as np
import scipy.linalg as scilin
import matplotlib.pyplot as plt
import pickle as pkl

class GMat(object):
    
    def __init__(self, beta, cij, lOverD, corrWin, center, ctrlRegionStart, ctrlRegionEnd, badPixMask, yFlip=False):
        """
            ctrlRegionStart: r, c control region start boundary wrt to center
            ctrlRegionEnd: r, c control region end boundary wrt to center
        """
        coordImage = np.mgrid[ctrlRegionStart[0]:ctrlRegionEnd[0], ctrlRegionStart[1]:ctrlRegionEnd[1]]
        coordImage = np.transpose(coordImage, axes=(1, 2, 0)) # should be indexed r, c, coordAxis
        coordList = np.reshape(coordImage, (-1, 2))

        modeImage = coordImage*lOverD*2*np.pi
        modeList = np.reshape(modeList, (-1, 2))

        #upper (and lower) initial diagonal for G (n_pix x n_modes)
        matBlock = np.diag(beta*np.ones(coordImage.shape[0]*coordImage.shape[1])) 
        for i in range(corrWin)
            np.fill_diagonal(matBlock[i+1:], beta*cij**(i+1))
            np.fill_diagonal(matBlock[:,i+1:], beta*cij**(i+1))

        badPixMask = badPixMask.astype(bool)
        badPixMaskList = badPixMask.flatten()
        #coordImage[badPixMask, :] = np.array([np.nan, np.nan])
        goodPixMaskList = ~badPixMaskList

        coordList = coordList[goodPixMask]
        matBlock = matBlock[goodPixMask, :]

        self.mat = scilin.block_diag(matBlock, matBlock)
        self.badPixMask = badPixMask
        self.coordImage = coordImage
        self.modeImage = modeImage
        self.coordList = coordList
        self.modeList = modeList




