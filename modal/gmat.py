import numpy as np
import scipy.linalg as scilin
import matplotlib.pyplot as plt
import pickle as pkl
import copy

class GMat(object):
    """
    Container for modal control/calibration matrix

    Attributes:
        mat: 2nPix x nMode control matrix. 
            rows map to real, then imag E-field of flattened pix array (good pix mask applied)
            ditto for columns/modes.
        coordList: List of good pixel coordinates in control region, relative to top left (i.e. 
            start at [0, 0])
        modeList: List of pixel modes located in control region. not limited to good pixels
            contains kVecs of modes; so 1/2 the size of actual list of modes (no phase value)
    """

    def __init__(self, beta, cij, lOverD, corrWin, center, ctrlRegionStart, ctrlRegionEnd, badPixMask, yFlip=False):
        """
            ctrlRegionStart: r, c control region start boundary wrt to center
            ctrlRegionEnd: r, c control region end boundary wrt to center
            corrWin: single sided, inclusive window size (e.g. corrWin = 5 will have 11 valid adjacent modes
        """
        coordImage = np.mgrid[0:(ctrlRegionEnd[0] - ctrlRegionStart[0]), 0:(ctrlRegionEnd[1] -  ctrlRegionStart[1])]
        coordImage = np.transpose(coordImage, axes=(1, 2, 0)) # should be indexed r, c, coordAxis
        coordList = np.reshape(coordImage, (-1, 2))

        modeImage = 2*np.pi/lOverD*np.mgrid[ctrlRegionStart[0]:ctrlRegionEnd[0], ctrlRegionStart[1]:ctrlRegionEnd[1]]
        modeImage = np.transpose(modeImage, axes=(1, 2, 0)) # should be indexed r, c, coordAxis
        modeList = np.reshape(modeImage, (-1, 2))
        self.intTime = None

        assert coordImage.shape == modeImage.shape

        if yFlip:
            self.yFlip = -1
        else:
            self.yFlip = 1

        #ensure badPixMask covers only control region
        if badPixMask.shape != coordImage.shape[:2]:
            badPixMask = badPixMask[(center[0] + ctrlRegionStart[0]):(center[0] + ctrlRegionEnd[0]), 
                    (center[1] + ctrlRegionStart[1]):(center[1] + ctrlRegionEnd[1])]

        #upper (and lower) initial diagonal for G (n_pix x n_modes)
        matBlock = np.diag(beta*np.ones(coordImage.shape[0]*coordImage.shape[1])) 
        for i, coord in enumerate(coordList):
            coordDiff = np.abs(coordList - coord)
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
        self.nPix = np.sum(goodPixMaskList)
        self.nHalfModes = len(self.modeList)
        self.pixIndImage = self._getPixIndImage()
        self.center = center
        self.ctrlRegionStart = ctrlRegionStart
        self.ctrlRegionEnd = ctrlRegionEnd
        self.pixIndImage = self._genPixIndImage()
        self.intTime = None
        self.corrWin = corrWin

    def _genPixIndImage(self):
        pixIndImage = np.nan*np.zeros(self.badPixMask.shape, dtype=int)
        for i in range(self.nPix):
            pixIndImage[self.coordList[i, 0], self.coordList[i, 1]] = i
        return pixIndImage
        

    def __getitem__(self, inds):
        gMat = copy.copy(self)
        gMat.badPixMask = None
        gMat.coordImage = None
        gMat.modeImage = np.copy(self.modeImage)
        gMat.modeCoordList = np.copy(self.modeCoordList)
        gMat.nHalfModes = self.nHalfModes
        gMat.intTime = self.intTime

        if isinstance(inds, int):
            gMat.mat = np.copy(self.mat[[inds, inds+self.nPix]])
            gMat.nPix = 1
        elif isinstance(inds, slice):
            gMat.mat = np.append(self.mat[inds], self.mat[(inds.start+self.nPix):(inds.stop+self.nPix)], axis=0)
            gMat.nPix = inds.stop - inds.start
        else:
            raise Exception('index must be int or slice')

        return gMat

    def __setitem__(self, inds, gMat):
        if isinstance(inds, int):
            if gMat.nPix != 1:
                raise Exception('Cannot copy {} pixels to 1 pixel dest slice'.format(gMat.nPix))
            self.mat[[inds, inds+self.nPix]] = gMat.mat
        elif isinstance(inds, slice):
            nPixSlice = inds.stop - inds.start
            if gMat.nPix != nPixSlice:
                raise Exception('Cannot copy {} pixels to {} pixel dest slice'.format(gMat.nPix, nPixSlice))
            self.mat[inds] = np.copy(gMat.mat[:nPixSlice])
            self.mat[inds.start + self.nPix: inds.stop + self.nPix] = np.copy(gMat.mat[nPixSlice:])
        else:
            raise Exception('index must be int or slice')

    def recomputeMat(self, beta, cij, corrWin):
        matBlock = np.diag(beta*np.ones(self.coordImage.shape[0]*self.coordImage.shape[1])) 
        coordList = np.reshape(self.coordImage, (-1, 2))
        for i, coord in enumerate(coordList):
            coordDiff = np.abs(coordList - coord)
            withinRangeMask = (coordDiff[:, 0] <= corrWin) & (coordDiff[:, 1] <= corrWin) #cut box around coords
            matBlock[i, withinRangeMask] = beta*cij**np.sqrt(coordDiff[withinRangeMask, 0]**2 + coordDiff[withinRangeMask, 1]**2)

        badPixMaskList = self.badPixMask.flatten()
        #coordImage[badPixMask, :] = np.array([np.nan, np.nan])
        goodPixMaskList = ~badPixMaskList
        matBlock = matBlock[goodPixMaskList, :]
        self.mat = scilin.block_diag(matBlock, matBlock)
        self.corrWin = corrWin


    def getDMSpeckles(self, modeVec):
        if len(modeVec) != 2*len(self.modeList):
            raise Exception('mode vector should be {} elements'.format(2*len(self.modeList)))

        modeVec = np.reshape(modeVec, (2, -1)).T
        #modeInds = np.where((modeVec[:len(self.modeList)] != 0)|(modeVec[len(self.modeList):] != 0)[0]
        modeInds = np.unique(np.where(modeVec!=0)[0])

        ampList = []
        phaseList = []
        kVecList = []

        for ind in modeInds:
            cxAmp = modeVec[ind]
            ampList.append(np.sqrt(cxAmp[0]**2 + cxAmp[1]**2))
            phaseList.append(np.arctan2(cxAmp[1], cxAmp[0]))
            #switch kVecs from r, c to x, y indexing
            kVecList.append([self.modeList[ind, 1], self.yFlip*self.modeList[ind, 0]])

        return np.array(ampList), np.array(phaseList), np.array(kVecList)

    def getPixMaskFromModeVec(self, modeVec):
        """
        get pixels in region of influence of nonzero elements of modeVec
        modeVec can either be full or half
        """
        if len(modeVec) == self.nHalfModes:
            modeMask = modeVec != 0
        else:
            modeMask = (modeVec[:self.nHalfModes] != 0) | (modeVec[self.nHalfModes:] != 0)

        pixMask = np.matmul(self.mat[:self.nPix, :self.nHalfModes], modeMask.astype(int)).astype(bool)
        #coords = self.coordList[pixMask, :]
        return pixMask

    def checkPixModeVec(self, modeVec, pixInd):
        """
        get pixels in region of influence of nonzero elements of modeVec
        modeVec can either be full or half
        """
        if len(modeVec) == self.nHalfModes:
            modeMask = modeVec != 0
        else:
            modeMask = (modeVec[:self.nHalfModes] != 0) | (modeVec[self.nHalfModes:] != 0)

        pixMask = np.matmul(self.mat[:self.nPix, :self.nHalfModes], modeMask.astype(int)).astype(bool)
        return pixMask[pixInd]








