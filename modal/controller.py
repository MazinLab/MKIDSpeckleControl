import numpy as np
import numpy.linalg as nlg
import scipy.linalg as scilin
import matplotlib.pyplot as plt
import os, time
import pickle as pkl
import copy

from gmat import GMat
import imageUtils as imu
import speckpy as sp
import mkidreadout.readout.sharedmem as shm

class Speckle(object):
    def __init__(self, gMat, coords, radius, initImage, snrThresh, reg): 
        """
        coords are r,c indexed wrt to ctrl region origin (0, 0)
        assume gMat.mat is in cps
        reg: controller regularization parameter
        """

        self.centerCoords = np.array(coords).astype(int)
        self.coordList = np.mgrid[(coords[0] - radius):(coords[0] + radius + 1), 
                (coords[1] - radius):(coords[1] + radius + 1)].astype(int)
        self.coordList = np.reshape(np.transpose(self.coordList, (1, 2, 0)), (-1, 2)) #shape: [i, (x or y)]
        self.pixIndList = gMat.pixIndImage[self.coordList[:,0], self.coordList[:,1]] #check this statement for correctness
        gpm = ~np.isnan(self.pixIndList)
        self.pixIndList = self.pixIndList[gpm].astype(int)
        self.coordList = self.coordList[gpm]
        self.nPix = len(self.pixIndList)

        #slice of gMat corresponding to pixIndList
        self.mat = np.append(gMat.mat[self.pixIndList], gMat.mat[self.pixIndList + gMat.nPix], axis=0)
        self.iter = 0 #number of complete probe/ctrl cycles 
        self.state = 'reprobe' #reprobe, improbe, or null
        self.snrThresh = snrThresh
        self.reg = reg

        self.reProbeZ = np.zeros(len(self.pixIndList))
        self.imProbeZ = np.zeros(len(self.pixIndList))
        self.reProbeZVar = np.zeros(len(self.pixIndList))
        self.imProbeZVar = np.zeros(len(self.pixIndList))

        #TODO: what if coords are on a bad pix?
        self.halfModeProbeVec = np.zeros(gMat.nHalfModes)
        probeModeInd = np.argmax(gMat.mat[gMat.pixIndImage[self.centerCoords[0], self.centerCoords[1]].astype(int)])
        self.halfModeProbeVec[probeModeInd] = np.sqrt(initImage[self.centerCoords[0], self.centerCoords[1]])/gMat.mat[gMat.pixIndImage[self.centerCoords[0], self.centerCoords[1]].astype(int), probeModeInd]

    def update(self, ctrlRegionImage, ctrlRegionImageVar):
        vec = ctrlRegionImage[self.coordList[:, 0], self.coordList[:, 1]].flatten()
        varVec = ctrlRegionImageVar[self.coordList[:, 0], self.coordList[:, 1]].flatten()
        if self.state == 'reprobe':
            self.reProbeZ = (self.iter*self.reProbeZ + vec)/(self.iter + 1)
            self.reProbeZVar = (self.iter*self.reProbeZVar + varVec)/(self.iter + 1)**2
            self.state = 'improbe'
        elif self.state == 'improbe':
            self.imProbeZ = (self.iter*self.imProbeZ + vec)/(self.iter + 1)
            self.imProbeZVar = (self.iter*self.imProbeZVar + varVec)/(self.iter + 1)**2
            self.state = 'null'
        elif self.state == 'null':
            self.state = 'reprobe'
            #maybe recentroid/recompute probe here?




    def getNextModeVec(self):
        """
        return next pairwise probe modeVec OR nulling modeVec
        returns a modeVec followed by 'probe' or 'null'
        """
        if self.state == 'reprobe':
            return np.append(self.halfModeProbeVec, np.zeros(len(self.halfModeProbeVec)))
        elif self.state == 'improbe':
            return np.append(np.zeros(len(self.halfModeProbeVec)), self.halfModeProbeVec)
        elif self.state == 'null':
            self.iter += 1
            return self._computeControl()

    def _computeControl(self):
        Hre = 4*np.dot(self.mat[:self.nPix, :len(self.halfModeProbeVec)], self.halfModeProbeVec).T
        Him = 4*np.dot(self.mat[self.nPix:, len(self.halfModeProbeVec):], self.halfModeProbeVec).T

        xre = 1/Hre*self.reProbeZ
        xim = 1/Him*self.imProbeZ
        Xrecov = np.dot(np.diag(1/Hre), np.dot(np.diag(self.reProbeZVar), np.diag(1/Hre)))
        Ximcov = np.dot(np.diag(1/Him), np.dot(np.diag(self.imProbeZVar), np.diag(1/Him)))
        
        x = np.append(xre, xim)
        xCov = scilin.block_diag(Xrecov, Ximcov)
        ctrlMat = -np.dot(nlg.inv(np.dot(self.mat.T, self.mat) + self.reg*np.diag(np.ones(2*len(self.halfModeProbeVec)))), self.mat.T)

        ctrlModeVec = np.dot(ctrlMat, x)
        ctrlModeCov = np.dot(ctrlMat, np.dot(xCov, ctrlMat.T))

        ctrlModeVec[np.isnan(ctrlModeVec)] = 0
        ctrlModeCov[np.isnan(ctrlModeCov)] = 10000

        snrVec = ctrlModeVec/np.sqrt(np.diag(ctrlModeCov)) #SNR of each element of ctrlModeVec
        meanSNR = np.average(snrVec, weights=ctrlModeVec)

        print 'xre', xre
        print 'xim', xim
        print 'ctrlModeVec', ctrlModeVec

        if meanSNR >= self.snrThresh:
            return ctrlModeVec
        else:
            return np.zeros(len(ctrlModeVec))

class Controller(object):
    def __init__(self, shmImName, dmChanName, gMat, wvlRange=None, sim=False): #intTime for backwards comp
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
        
        self.gMat = copy.deepcopy(gMat)
        self.gMat.changeIntTime(1)
        self.imgStart = np.array(gMat.ctrlRegionStart) + np.array(gMat.center)
        self.imgEnd = np.array(gMat.ctrlRegionEnd) + np.array(gMat.center)
        self.badPixMaskCtrl = gMat.badPixMask

    def runLoop(self, nIters, intTime, maxSpecks, exclusionZone, maxProbeIters, speckleRad=4, reg=0.1, snrThresh=3):
        self.speckles = []
        self.dmChan.clearProbeSpeckles()
        self.dmChan.clearNullingSpeckles()
        self.dmChan.updateDM()
        for i in range(nIters):
            #detect + re probe + im probe/null
            self.shmim.startIntegration(integrationTime=intTime)
            ctrlRegionImage = self.shmim.receiveImage()[self.imgStart[0]:self.imgEnd[0], 
                    self.imgStart[1]:self.imgEnd[1]]
            ctrlRegionImage[self.badPixMaskCtrl] = 0
            ctrlRegionImage = ctrlRegionImage.astype(float)/intTime
            for speck in self.speckles:
                speck.update(ctrlRegionImage, ctrlRegionImage)
            self._detectSpeckles(ctrlRegionImage, maxSpecks, exclusionZone, speckleRad, reg, snrThresh)

            for j in range(2): #re and im probes
                modeVec = np.zeros(2*self.gMat.nHalfModes)
                for speck in self.speckles:
                    modeVec += speck.getNextModeVec()
                    print speck.state
                probeZ, probeVar = self._probePair(modeVec, intTime)
                for speck in self.speckles:
                    speck.update(probeZ, probeVar)

            specksToDelete = []
            modeVec = np.zeros(2*self.gMat.nHalfModes)
            for j, speck in enumerate(self.speckles):
                smv = speck.getNextModeVec()
                print speck.state
                modeVec += smv
                if np.any(smv > 0):
                    print 'Nulling speckle at: ', speck.centerCoords
                    specksToDelete.append(j)
                elif speck.iter >= maxProbeIters:
                    print 'Deleting speckle at: ', speck.centerCoords
                    specksToDelete.append(j)
            
            self._applyToDM(modeVec, 'null')

            for j, ind in enumerate(specksToDelete):
                del self.speckles[ind - j]


    def _detectSpeckles(self, ctrlRegionImage, maxSpecks, exclusionZone, speckleRad, reg, snrThresh):
        speckleCoords = imu.identify_bright_points(ctrlRegionImage, exclusionZone)
        speckleCoords = imu.filterpoints(speckleCoords, exclusionZone, maxSpecks)

        for i, coord in enumerate(speckleCoords):
            addSpeckle = True
            if np.any(coord < speckleRad) or (coord[0] >= ctrlRegionImage.shape[0] - speckleRad) or (coord[1] >= ctrlRegionImage.shape[1] - speckleRad):
                addSpeckle = False
                continue
            for speck in self.speckles:
                if np.sqrt((speck.centerCoords[0] - coord[0])**2 + (speck.centerCoords[1] - coord[1])**2) < exclusionZone:
                    addSpeckle = False
                    break
            if addSpeckle:
                speckle = Speckle(self.gMat, coord, speckleRad, ctrlRegionImage, snrThresh, reg)
                if len(self.speckles) < maxSpecks:
                    self.speckles.append(speckle)
                    print 'Detected speckle at', coord
                else:
                    break

    def _probePair(self, modeVec, intTime):
        probeImgs = []
        for i in range(2): #2x real probes followed by 2x imag
            if i == 1:
                modeVec *= -1
            self._applyToDM(modeVec, 'probe')
            self.shmim.startIntegration(integrationTime=intTime)
            print 'recv img'
            img = self.shmim.receiveImage()
            print 'done'
            probeImgs.append(img[self.imgStart[0]:self.imgEnd[0], 

                self.imgStart[1]:self.imgEnd[1]])

        probeZ = (probeImgs[0] - probeImgs[1])/intTime
        probeVar = (probeImgs[0] + probeImgs[1])/intTime**2
        return probeZ, probeVar #order is real, -real, im, -im

    def _applyToDM(self, modeVec, modeType, clearProbes=True, update=True):
        modeType = modeType.lower()
        if modeType != 'probe' and modeType != 'null':
            raise Exception('modeType must be probe or null')

        if clearProbes:
            self.dmChan.clearProbeSpeckles()

        amps, phases, kVecs = self.gMat.getDMSpeckles(modeVec)

        for i in range(len(amps)):
            if modeType == 'probe':
                self.dmChan.addProbeSpeckle(kVecs[i,0], kVecs[i,1], amps[i], phases[i])
            else:
                self.dmChan.addNullingSpeckle(kVecs[i,0], kVecs[i,1], amps[i], phases[i])

        if update:  
            self.dmChan.updateDM()
