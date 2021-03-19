import numpy as np
import matplotlib.pyplot as plt
import os, time
import pickle as pkl

from gmat import GMat
import speckpy as sp
import mkidreadout.readout.sharedmem as shm

class Speckle(object):
    def __init__(self, gMat, coords, radius, initImage, intTime, snrThresh): 
        """
        coords are r,c indexed wrt to ctrl region origin (0, 0)
        """
        self.centerCoords = coords
        self.coordList = np.mgrid[(coords[0] - radius):(coords[0] + radius + 1), 
                (coords[1] - radius):(coords[1] + radius + 1)]
        self.coordList = np.reshape(np.transpose(self.coordList, (1, 2, 0)), (-1, 2)) #shape: [i, (x or y)]
        self.pixIndList = gMat.pixIndImage[self.coordList] #check this statement for correctness
        gpm = ~np.isnan(self.pixIndList)
        self.pixIndList = self.pixIndList[gpm]
        self.coordList = self.coordList[gpm]

        #slice of gMat corresponding to pixIndList
        self.mat = np.append(gMat.mat[self.pixIndList], gMat.mat[self.pixIndList + gMat.nPix])

    def update(self, image):
        pass

    def getNextSpeckle(self):
        pass

    def _computeControl(self):
        pass

class Controller(object):
    def __init__(self, shmImName, dmChanName, gMat, wvlRange=None, intTime=None): #intTime for backwards comp
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

    def _getSpeckleProbes(self, ctrlRegionImage, maxSpecks, exclusionZone):
        pass

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
