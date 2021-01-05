import numpy as np
import scipy.linalg as scilin
import matplotlib.pyplot as plt
import pickle as pkl

class GMat(object):
    """
    Container for modal control/calibration matrix

    Attributes:
        mat: 2nPix x nMode control matrix. 
            rows map to real, then imag E-field of flattened pix array (good pix mask applied)
            ditto for columns/modes.
        coordList: List of good pixel coordinates in control region, relative to center
        modeList: List of pixel modes located in control region. not limited to good pixels
            contains kVecs of modes; so 1/2 the size of actual list of modes (no phase value)
    """

    def __init__(self, beta, cij, lOverD, corrWin, center, ctrlRegionStart, ctrlRegionEnd, badPixMask, yFlip=False):
        """
            ctrlRegionStart: r, c control region start boundary wrt to center
            ctrlRegionEnd: r, c control region end boundary wrt to center
        """
        coordImage = np.mgrid[0:(ctrlRegionEnd[0] - ctrlRegionStart[0]), 0:(ctrlRegionEnd[1] -  ctrlRegionStart[1])]
        coordImage = np.transpose(coordImage, axes=(1, 2, 0)) # should be indexed r, c, coordAxis
        coordList = np.reshape(coordImage, (-1, 2))

        modeImage = lOverD*2*np.pi*np.mgrid[ctrlRegionStart[0]:ctrlRegionEnd[0], ctrlRegionStart[1]:ctrlRegionEnd[1]]
        modeImage = np.transpose(modeImage, axes=(1, 2, 0)) # should be indexed r, c, coordAxis
        modeList = np.reshape(modeImage, (-1, 2))

        assert coordImage.shape == modeImage.shape

        if badPixMask.shape != coordImage.shape[:2]:
            badPixMask = badPixMask[(center[0] + ctrlRegionStart[0]):(center[0] + ctrlRegionEnd[0]), 
                    (center[1] + ctrlRegionStart[1]):(center[1] + ctrlRegionEnd[1])]

        #upper (and lower) initial diagonal for G (n_pix x n_modes)
        matBlock = np.diag(beta*np.ones(coordImage.shape[0]*coordImage.shape[1])) 
        for i, coord in enumerate(coordList):
            coordDiff = coordList - coord
            withinRangeMask = (coordDiff[:, 0] <= corrWin) & (coordDiff[:, 1] <= corrWin) #cut box around coords
            matBlock[i, withinRangeMask] = beta*cij**np.sqrt(coordDiff[withinRangeMask, 0]**2 + coordDiff[withinRangeMask, 1]**2)
            #for j in range(corrWin+1):
            #    matBlock[i][withinRangeMask] = beta*cij**j
        #for i in range(corrWin)
        #    np.fill_diagonal(matBlock[i+1:], beta*cij**(i+1))
        #    np.fill_diagonal(matBlock[:,i+1:], beta*cij**(i+1))

        badPixMask = badPixMask.astype(bool)
        badPixMaskList = badPixMask.flatten()
        #coordImage[badPixMask, :] = np.array([np.nan, np.nan])
        goodPixMaskList = ~badPixMaskList

        coordList = coordList[goodPixMaskList] 
        matBlock = matBlock[goodPixMaskList, :]

        self.mat = scilin.block_diag(matBlock, matBlock)
        self.badPixMask = badPixMask
        self.coordImage = coordImage #ctrl region image containing coordinate of each point, starting at 0,0
        self.modeImage = modeImage #ctrl region image containing kvecs for that pixel
        self.coordList = coordList #bad pix masked list of ctrl region coords starting at 0,0
        self.modeCoordList = np.reshape(coordImage, (-1, 2)) #coordList but not bad pix masked (i.e. integer form of kVecs)
        self.modeList = modeList #list of (real) kVec modes. Full mode list is 2x size

    def getDMSpeckles(self, modeVec):
        if len(modeVec) != 2*len(self.modeList):
            raise Exception('mode vector should be {} elements'.format(2*len(self.modeList)))

        modeVec = np.reshape(modeVec, (-1, 2))
        #modeInds = np.where((modeVec[:len(self.modeList)] != 0)|(modeVec[len(self.modeList):] != 0)[0]
        modeInds = np.unique(np.where(modeVec!=0)[0])

        ampList = []
        phaseList = []
        kVecList = []

        for ind in modeInds:
            cxAmp = modeVec[ind]
            ampList.append(np.sqrt(cxAmp[0]**2 + cxAmp[1]**2))
            phaseList.append(np.arctan2(cxAmp[1], cxAmp[0]))
            kVecList.append(self.modeList[ind])

        return np.array(ampList), np.array(phaseList), np.array(kVecList)





