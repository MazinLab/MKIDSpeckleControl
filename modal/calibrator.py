import numpy as np
import scipy.linalg as scilin
import matplotlib.pyplot as plt
import pickle as pkl

from gmat import GMat
import speckpy as sp
import mkidreadout.readout.sharedmem as shm

class Calibrator(object):

    def __init__(self, shmImName, dmChanName,  beta, cij, lOverD, corrWin, center, 
            ctrlRegionStart, ctrlRegionEnd, badPixMask, wvlRange=None, yFlip=False):
        """
        Takes calibration data set
        """
        self.gMat = GMat(beta, cij, lOverD, corrWin, center, 
                ctrlRegionStart, ctrlRegionEnd, badPixMask, yFlip=False)
        self.reProbeImgs = [] #pairwise probe images
        self.imProbeImgs = []
        self.ctrlVecs = [] #vector of control offsets from probes (referenced to GMat.modelist)
        self.reProbeVecs = []
        self.imProbeVecs = []

        self.shmim = shm.ImageCube(shmImName)
        self.dmChan = sp.SpeckleToDM(dmChanName)

        if wvlRange:
            self.shmim.set_wvlStart(wvlRange[0])
            self.shmim.set_wvlStop(wvlRange[1])
            self.shmim.set_useWvl(True)
        else:
            self.shmim.set_useWvl(False)

        # ctrl region boundaries in real image coordinates
        self.imgStart = ctrlRegionStart + center
        self.imgEnd = ctrlRegionEnd + center


    def run(self, nIters, nInitProbeIters, nProbesPerCtrl, maxProbes, maxCtrl, dmAmpRange, exclusionZone, intTime):
        for i in range(nInitProbeIters):
            modeInds = self._pickRandomModes(maxProbes, exclusionZone)
            halfModeVec = np.zeros(len(self.gmat.modeList))
            halfModeVec[modeInds] = dmAmpRange*np.random.random(len(modeInds))
            self._probeCycle(halfModeVec, intTime)
            self.ctrlVecs.append(np.zeros(2*len(halfModeVec)))

        for i in range(nIters):
            #control then probe as per KF equations
            modeInds = self._pickRandomModes(maxProbes, exclusionZone)
            halfModeVec = np.zeros(len(self.gmat.modeList))
            halfModeVec[modeInds] = dmAmpRange*np.random.random(len(modeInds))
            halfModePhaseVec = np.zeros(len(self.gmat.modeList))
            halfModePhaseVec[modeInds] = 2*np.pi*np.random.random(len(modeInds))
            self._addCtrlModes(halfModeVec, halfModePhaseVec)

            modeInds = self._pickRandomModes(maxProbes, exclusionZone)
            halfModeVec = np.zeros(len(self.gmat.modeList))
            halfModeVec[modeInds] = dmAmpRange*np.random.random(len(modeInds))
            self._probeCycle(halfModeVec, intTime)

        self._save()

    def runProbeCtrlProbe(self, nIters, maxModes, dmAmpRange, exclusionZone, intTime):
        for i in range(nIters):
            modeInds = self._pickRandomModes(maxProbes, exclusionZone)
            halfModeVec = np.zeros(len(self.gmat.modeList))
            halfModeVec[modeInds] = dmAmpRange*np.random.random(len(modeInds))
            self._probeCycle(halfModeVec, intTime)

            halfModePhaseVec = np.zeros(len(self.gmat.modeList))
            halfModeVec[modeInds] = dmAmpRange*np.random.random(len(modeInds))
            halfModePhaseVec[modeInds] = 2*np.pi*np.random.random(len(modeInds))
            self._addCtrlModes(halfModeVec, halfModePhaseVec)

            halfModeVec[modeInds] = dmAmpRange*np.random.random(len(modeInds))
            self._probeCycle(halfModeVec, intTime)


    def _save():
        pass

    def _probeCycle(halfModeVec, intTime):
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
            self.shmim.startIntegration(intTime=intTime)
            probeImgs.append(self.shmim.receiveImage()[imgStart[0]:imgEnd[0], imgStart[1]:imgEnd[1]])

        self.reProbeVecs.append(np.append(halfModeVec, np.zeros(len(halfModeVec))))
        self.imProbeVecs.append(np.append(np.zeros(len(halfModeVec)), halfModeVec))
        self.reProbeImgs.append(probeImgs[0] - probeImgs[1])
        self.imProbeImgs.append(probeImgs[2] - probeImgs[3])


    def _addCtrlModes(self, halfModeVec, halfModePhaseVec):
        modeVec = halfModeVec*np.cos(halfModePhaseVec)
        modeVec = np.append(halfModeVec*np.sin(halfModePhaseVec))
        self._applyToDM(modeVec, 'null')

    def _pickRandomModes(self, nModes, exclusionZone):
        validModeMask = np.ones(len(self.gMat.modeList))
        modeInds = []
        for i in range(nModes):
            modeInd = np.random.choice(np.where(validModeMask)[0])
            modeCoord = self.gMat.modeCoordList[modeInd]
            coordDists = np.sqrt((self.gMat.modeCoordList[:,0] - modeCoord[0])**2 + (self.gMat.modeCoordList[:,1] - modeCoord[1])**2)
            validModeMask[coordDists <= exclusionZone] = 0
            modeInds.append(modeInd)

        return np.array(modeInds)

    def _applyToDM(self, modeVec, modeType, clearProbes=True):
        modeType = modeType.lower()
        if modeType != 'probe' and modeType != 'null':
            raise Exception('modeType must be probe or null')

        if clearProbes:
            self.dmChan.clearProbeSpeckles()

        amps, phases, kVecs = self.gmat.getDMSpeckles(modeVec)

        for i in range(len(amps)):
            if modeType == 'probe':
                self.dmChan.addProbeSpeckle(kVecs[i,0], kVecs[i,1], amps[i], phases[i])
            else:
                self.dmChan.addNullingSpeckle(kVecs[i,0], kVecs[i,1], amps[i], phases[i])

        self.dmChan.updateDM()



