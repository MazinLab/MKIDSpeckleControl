import numpy as np
import numpy.linalg as nlg
import scipy.linalg as scilin
import matplotlib.pyplot as plt
import os, time
import pickle as pkl
import copy

import imageUtils as imu
import speckpy as sp
import mkidreadout.readout.sharedmem as shm

class Speckle(object):
    
    def __init__(self, coords, initImage, probeImg): 
        """
        initImage - initial intensity image
        coords - coords in control region
        probeImg - ctrlRegion.shape dim image w/ up
        """
        self.coords = coords
        self.i0Img = initImage
        self.upImg = probeImg
        self.zImg = np.zeros(probeImg.shape+(2,))
        self.zCovImg = np.zeros(probeImg.shape+(2,))
        self.nIters = 0
        self.i1Img = None

    def addProbeCycle(self, z, zCov):
        pass

    def getProbeSNR(self):
        pass

class Controller(object):
    
    def __init__(self, shmImName, dmChanName, qModel, probeBeta, probeApRad, wvlRange=None, sim=False, badPixMask=None): #intTime for backwards comp
        """
        probeBeta is beta integrated across an aperture w/ radius probeApRad
        """
        self.shmim = shm.ImageCube(shmImName)
        self.dmChan = sp.SpeckleToDM(dmChanName)

        if wvlRange:
            self.shmim.set_wvlStart(wvlRange[0])
            self.shmim.set_wvlStop(wvlRange[1])
            self.shmim.set_useWvl(True)
        else:
            self.shmim.set_useWvl(False)

        # ctrl region boundaries in real image coordinates
        self.imgStart = np.array(gMat.ctrlRegionStart) + np.array(gMat.center)
        self.imgEnd = np.array(gMat.ctrlRegionEnd) + np.array(gMat.center)
        self.sim = sim
        self.qModel = qModel

        if badPixMask is not None:
            self.badPixMaskCtrl = badPixMask[(qModel.center[0] + gMat.ctrlRegionStart[0]):(gMat.center[0] + gMat.ctrlRegionEnd[0]), 
                    (qModel.center[1] + gMat.ctrlRegionStart[1]):(gMat.center[1] + gMat.ctrlRegionEnd[1])]
        else:
            self.badPixMaskCtrl = np.zeros(qModel.imgShape)

        self.setProbeParams(probeBeta, probeApRad)

    def setProbeParams(self, probeBeta, probeApRad)
        self.probeBeta = probeBeta
        self.probeApRad = probeApRad
        self.probeFiltSize = 2*probeApRad + 1

    def runLoop(self, nIters, intTime, maxSpecks, exclusionZone, maxProbeIters, snrThresh=3, plot=True):
        self.speckles = []
        self.dmChan.clearProbeSpeckles()
        self.dmChan.clearNullingSpeckles()
        self.dmChan.updateDM()
        lc = []
        dt = np.arange(0, nIters*intTime*5, intTime*5)
        for i in range(nIters):
            #detect + re probe + im probe/null
            self.shmim.startIntegration(integrationTime=intTime)
            ctrlRegionImage = self.shmim.receiveImage()[self.imgStart[0]:self.imgEnd[0], 
                    self.imgStart[1]:self.imgEnd[1]]

            ctrlRegionImage[self.badPixMaskCtrl] = 0
            lc.append(np.sum(ctrlRegionImage))
            if plot:
                plt.imshow(ctrlRegionImage)
                plt.show()

            ctrlRegionImage = ctrlRegionImage.astype(float)/intTime
            #for speck in self.speckles:
            #    speck.update(ctrlRegionImage, ctrlRegionImage)
            self._detectSpeckles(ctrlRegionImage, maxSpecks, exclusionZone)

            if i > 0:
                for speck in doneSpecks:
                    speck.i1Img = ctrlRegionImage
                    self.qModel.addIter()
            

            #probing
            probeZ = np.zeros(self.qModel.imgShape + (2,))
            probeVar = np.zeros(self.qModel.imgShape + (2,))
            for j in range(2): #re and im probes
                modeImg = np.zeros(self.qModel.imgShape + (2,))
                for speck in self.speckles:
                    modeImg[:,:,j] += speck.upImg
                probeZ[:, :, j], probeVar[:, :, j] = self._probePair(modeImg, intTime)

            for speck in self.speckles:
                speck.addProbeCycle(probeZ, probeVar)

            #nulling
            specksToDelete = []
            doneSpecks = []
            modeImg = np.zeros(self.qModel.imgShape + (2,))
            for j, speck in enumerate(self.speckles):
                snr = speck.getProbeSNR()
                speck.iter += 1
                if snr >= snrThresh
                    print 'Nulling speckle at: ', speck.coords
                    specksToDelete.append(j)
                    modeImg += self.qModel.getUc(speck.zImg, speck.upImg, speck.i0Img, self.eps)
                    doneSpecks.append(speck)
                elif speck.iter >= maxProbeIters:
                    print 'Deleting speckle at: ', speck.coords
                    specksToDelete.append(j)
            

            for j, ind in enumerate(specksToDelete):
                del self.speckles[ind - j]

        return dt, lc


    def _detectSpeckles(self, ctrlRegionImage, maxSpecks, exclusionZone):
        speckleCoords = imu.identify_bright_points(ctrlRegionImage, exclusionZone)
        #speckleCoords = imu.filterpoints(speckleCoords, exclusionZone, maxSpecks)

        for i, coord in enumerate(speckleCoords):
            addSpeckle = True
            coord = np.array(coord)
            if np.any(coord < self.probeApRad) or (coord[0] >= ctrlRegionImage.shape[0] - self.probeApRad) 
                    or (coord[1] >= ctrlRegionImage.shape[1] - self.probeApRad):
                addSpeckle = False
                continue
            for speck in self.speckles:
                if np.sqrt((speck.centerCoords[0] - coord[0])**2 + (speck.centerCoords[1] - coord[1])**2) < exclusionZone:
                    addSpeckle = False
                    break
            if addSpeckle:
                if len(self.speckles) < maxSpecks:
                    upImg = self._getProbeImg(coords, ctrlRegionImage)
                    speckle = Speckle(coord, ctrlRegionImage, upImg)
                    self.speckles.append(speckle)
                    print 'Detected speckle at', coord
                else:
                    break

    def _probePair(self, modeImg, intTime):
        """
        modeImg: nRow x nCol x 2 img of up
        """
        probeImgs = []
        for i in range(2): #+ then - probe img
            if i == 1:
                modeImg*= -1
            self._applyToDM(modeImg, 'probe')
            self.shmim.startIntegration(integrationTime=intTime)
            print 'recv img'
            img = self.shmim.receiveImage()
            print 'done'
            probeImgs.append(img[self.imgStart[0]:self.imgEnd[0], 
                self.imgStart[1]:self.imgEnd[1]])

        probeZ = (probeImgs[0] - probeImgs[1])/intTime
        probeVar = (probeImgs[0] + probeImgs[1])/intTime**2
        return probeZ, probeVar #order is real, -real, im, -im

    def _applyToDM(self, modeImg, modeType, clearProbes=True, update=True):
        modeType = modeType.lower()
        if modeType != 'probe' and modeType != 'null':
            raise Exception('modeType must be probe or null')

        if clearProbes:
            self.dmChan.clearProbeSpeckles()

        amps, phases, kVecs = self.qModel.getDMSpeckles(modeImg)

        for i in range(len(amps)):
            if modeType == 'probe':
                self.dmChan.addProbeSpeckle(kVecs[i,0], kVecs[i,1], amps[i], phases[i])
            else:
                self.dmChan.addNullingSpeckle(kVecs[i,0], kVecs[i,1], amps[i], phases[i])

        if update:  
            self.dmChan.updateDM()

    def _getProbeImg(self, coords, i0Img):
        """
        scales speckle aperture intensity by self.probeBeta and returns nRows x nCol image w/ up
        """
        filtImg = imu.smartBadPixFilt(i0Img, self.badPixMaskCtrl)
        i0 = np.sum(filtImg[coords[0] - self.probeApRad:coords[0] + self.probeApRad + 1, coords[1] - self.probeApRad:coords[1] + self.probeApRad + 1])
        up = np.sqrt(i0)/self.probeBeta
        probeImg = np.zeros(self.qModel.imgShape)
        probeImg[coords[0], coords[1]] = up
        return probeImg
        
