import numpy as np
import scipy.linalg as scilin
import matplotlib.pyplot as plt
import pickle as pkl

from GMat import GMat
import speckpy as sp
import mkidreadout.readout.sharedmem as shm

class Calibrator(object):

    def __init__(self, shmImName, dmChanName,  beta, cij, lOverD, corrWin, center, 
            ctrlRegionStart, ctrlRegionEnd, badPixMask, yFlip=False):
        """
        Takes calibration data set
        """
        self.gMat = GMat(beta, cij, lOverD, corrWin, center, 
                ctrlRegionStart, ctrlRegionEnd, badPixMask, yFlip=False)
        self.reProbeImgs = [] #pairwise probe images
        self.imProbeImgs = []
        self.ctrlVecs = [] #vector of control offsets from probes (referenced to GMat.modelist)

        self.shmim = shm.ImageCube(shmImName)
        self.dmChan = sp.SpeckleToDM(dmChanName)


    def run(nIters, nInitIters, nProbesPerCtrl, maxProbes, maxCtrl, dmAmpRange, exclusionZone):
        pass

    def _save():
        pass

    def _probeCycle(halfModeVec, intTime):
        pass

    def _addCtrlModes():
        pass

    def _pickModes():
        pass
