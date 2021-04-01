import numpy as np
import numpy.linalg as nlg
import scipy.linalg as scilin
import matplotlib.pyplot as plt
import os, time
import pickle as pkl

from gmat import GMat
import imageUtils as imu
import speckpy as sp
import mkidreadout.readout.sharedmem as shm

class Speckle(object):
    def __init__(self, gMat, coords, radius, initImage, snrThresh, reg): 
        """
        coords are r,c indexed wrt to ctrl region origin (0, 0)
        assume gMat.mat is in cps
        """

        self.centerCoords = coords
        self.coordList = np.mgrid[(coords[0] - radius):(coords[0] + radius + 1), 
                (coords[1] - radius):(coords[1] + radius + 1)]
        self.coordList = np.reshape(np.transpose(self.coordList, (1, 2, 0)), (-1, 2)) #shape: [i, (x or y)]
        self.pixIndList = gMat.pixIndImage[self.coordList] #check this statement for correctness
        gpm = ~np.isnan(self.pixIndList)
        self.pixIndList = self.pixIndList[gpm]
        self.coordList = self.coordList[gpm]
        self.nPix = len(self.pixIndList)

        #slice of gMat corresponding to pixIndList
        self.mat = np.append(gMat.mat[self.pixIndList], gMat.mat[self.pixIndList + gMat.nPix])
        self.iter = 0 #number of complete probe/ctrl cycles 
        self.state = 'reprobe' #reprobe, improbe, or null
        self.snrThresh = snrThresh
        self.reg = reg

        self.reProbeZ = np.zeros(len(pixIndList))
        self.imProbeZ = np.zeros(len(pixIndList))
        self.reProbeZVar = np.zeros(len(pixIndList))
        self.imProbeZVar = np.zeros(len(pixIndList))

        #TODO: what if coords are on a bad pix?
        self.halfModeProbeVec = np.zeros(gMat.nHalfModes)
        probeModeInd = np.argmax(gMat.mat[gMat.pixIndImage[coords]])
        self.halfModeProbeVec[probeModeInd] = np.sqrt(initImage[coords])/gMat.mat[gMat.pixIndImage[coords], probeModeInd]

    def update(self, ctrlRegionImage, ctrlRegionImageVar):
        vec = ctrlRegionImage[self.coordList].flatten()
        varVec = ctrlRegionImageVar[self.coordList].flatten()
        if self.state == 'reprobe':
            self.reProbeZ = (self.iter*self.reProbeZ + vec)/(self.iter + 1)
            self.reProbeZVar = (self.iter*self.reProbeZ + vec)/(self.iter + 1)**2
            self.state == 'improbe'
        elif self.state == 'improbe':
            self.imProbeZ = (self.iter*self.imProbeZ + vec)/(self.iter + 1)
            self.imProbeZVar = (self.iter*self.imProbeZ + vec)/(self.iter + 1)**2
            self.state == 'null'
        elif self.state == 'null':
            self.state = 'reprobe'
            self.iter += 1
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
            return self._computeControl()

    def _computeControl(self):
        Hre = 4*np.dot(self.mat[:self.nPix, :len(self.halfModeProbeVec)], self.halfModeProbeVec).T
        Him = 4*np.dot(self.mat[self.nPix:, len(self.halfModeProbeVec):], self.halfModeProbeVec).T

        xre = np.dot(nlg.inv(Hre), self.reProbeZ)
        xim = np.dot(nlg.inv(Him), self.imProbeZ)
        Xrecov = np.dot(nlg.inv(Hre), np.dot(np.diag(self.reProbeZVar), nlg.inv(Hre).T))
        Ximcov = np.dot(nlg.inv(Him), np.dot(np.diag(self.imProbeZVar), nlg.inv(Him).T))
        
        x = np.append(xre, xim)
        xCov = scillin.block_diag(Xrecov, Ximcov)
        ctrlMat = -np.dot(nlg.inv(np.dot(self.mat.T, self.mat) + self.reg*np.diag(np.ones(2*len(self.halfModeProbeVec)))), self.mat.T)
        ctrlModeVec = np.dot(ctrlMat, x)
        ctrlModeCov = np.dot(ctrlMat, np.dot(xCov, ctrlMat.T))
        snrVec = ctrlModeVec/np.sqrt(np.diag(ctrlModeCov)) #SNR of each element of ctrlModeVec
        meanSNR = np.average(snrVec, weights=ctrlModeVec)

        if meanSNR >= self.snrThresh:
            return ctrlModeVec
        else:
            return np.zeros(ctrlModeVec)

class Controller(object):
    def __init__(self, shmImName, dmChanName, gMat, wvlRange=None): #intTime for backwards comp
        self.shmim = shm.ImageCube(shmImName)
        self.dmChan = sp.SpeckleToDM(dmChanName)

        if wvlRange:
            self.shmim.set_wvlStart(wvlRange[0])
            self.shmim.set_wvlStop(wvlRange[1])
            self.shmim.set_useWvl(True)
        else:
            self.shmim.set_useWvl(False)

        # ctrl region boundaries in real image coordinates
        self.imgStart = np.array(ctrlRegionStart) + np.array(center)
        self.imgEnd = np.array(ctrlRegionEnd) + np.array(center)
        self.sim = sim
        
        self.gMat = gMat
        self.gMat.mat /= np.sqrt(self.gMat.intTime) #convert gMat to cps
        self.imgStart = np.array(gmat.ctrlRegionStart) + np.array(gmat.center)
        self.imgEnd = np.array(gmat.ctrlRegionEnd) + np.array(gmat.center)
        self.badPixMaskCtrl = gMat.badPixMask

    def runLoop(self, nIters, intTime, maxSpecks, exclusionsZone):
        self.speckles = []
        for i in range(nIters):
            for j in range(3): #detect + re probe + im probe
                self.shmim.startIntegration(integrationTime=intTime)
                ctrlRegionImage = self.shmim.receiveImage()[self.imgStart[0]:self.imgEnd[0], 
                        self.imgStart[1]:self.imgEnd[1])
                ctrlRegionImage[self.badPixMaskCtrl] = 0

    def _detectSpeckles(self, ctrlRegionImage, maxSpecks, exclusionZone):
        speckleCoords = imu.identify_bright_points(ctrlRegionImage)
        speckleCoords = imu.filterpoints(speckleCoords, exclusionZone, maxSpecks)

        for i, coord in enumerate(speckleCoords):
            addSpeckle = True
            for speck in self.speckles:
                if np.sqrt((speck.centerCoords[0] - coord[0])**2 + (speck.centerCoords[1] - coord[1])**2) < exclusionZone:
                    addSpeckle = False
                    break
            if addSpeckle:
                speckle = Speckle(self.gMat, coords)
                if len(self.specklesList) < self.paramDict['maxSpeckles']:
                    self.speckles.append(speckle)
                    print 'Detected speckle at', coord
                else:
                    break

    def _probeCycle(self, halfModeVec, intTime):
        probeImgs = []
        for i in range(4): #2x real probes followed by 2x imag
            if i%2 == 1:
                modeVec = -halfModeVec
            else:
                modeVec = halfModeVec
            if i < 2:
                modeVec = np.append(modeVec, np.zeros(len(modeVec)))
            else:
                modeVec = np.append(np.zeros(len(modeVec)), modeVec)

            self._applyToDM(modeVec, 'probe')
            self.shmim.startIntegration(integrationTime=intTime)
            print 'recv img'
            img = self.shmim.receiveImage()
            print 'done'
            probeImgs.append(img[self.imgStart[0]:self.imgEnd[0], 

                self.imgStart[1]:self.imgEnd[1]])

        return probeImgs #order is real, -real, im, -im

    def _addCtrlModes(self, halfModeVec, halfModePhaseVec):
        modeVec = halfModeVec*np.cos(halfModePhaseVec)
        modeVec = np.append(modeVec, halfModeVec*np.sin(halfModePhaseVec))
        self._applyToDM(modeVec, 'null', update=(not self.sim))

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
