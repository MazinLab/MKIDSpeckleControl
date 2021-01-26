from gmat import GMat
from calibrator import Calibrator

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

class OfflineEM(object):
    def __init__(self, tsString, path='.'):
        with open(os.path.join(path, 'gmat_{}.p'.format(tsString))) as f:
            self.gMat = pkl.load(f)

        data = np.load(os.path.join(path, 'cal_data_{}.npz'.format(tsString)))

        self.reZs = [np.array([]) for i in range(self.gMat.nPix)]
        self.imZs = [np.array([]) for i in range(self.gMat.nPix)]
        self.reUps = [np.array([]) for i in range(self.gMat.nPix)]
        self.imUps = [np.array([]) for i in range(self.gMat.nPix)]
        self.uCs = [np.array([]) for i in range(self.gMat.nPix)] 


        reProbeZ = data['reProbeImgs']
        imProbeZ = data['imProbeImgs']
        badPixMaskList = self.gmat.badPixMask.flatten()
        reProbeZ = np.reshape(reProbeZ, (-1, len(badPixMaskList)))
        imProbeZ = np.imshape(imProbeZ, (-1, len(badPixMaskList)))
        reProbeZ = reProbeZ[:, badPixMaskList]
        imProbeZ = imProbeZ[:, badPixMaskList]


        for pixInd in range(nPix):
            curState = None
            for i in range(len(data['ctrlVecs'])):
                if checkPixModeVec(data['ctrlVecs'][i], pixInd):
                    if curState is None: #or curState == 'probe':
                        curState = 'ctrl'
                        self.uCs[pixInd] = data['ctrlVecs'][i]
                    elif curState == 'ctrl':
                        self.uCs[pixInd] += data['ctrlVecs'][i]
                    else:
                        curState = 'ctrl'
                        self.uCs[pixInd] = np.vstack((self.uCs[pixInd], data['ctrlVecs'][i]))
                if checkPixModeVec(data['reProbeVecs'][i], pixInd):
                    if curState is None:
                        curState = 'probe'
                        self.reUps[pixInd] = data['reProbeVecs'][i]
                        self.imUps[pixInd] = data['imProbeVecs'][i]
                        self.reZs[pixInd] = np.append(self.reZs[pixInd], reProbeZ[i, pixInd])
                        self.imZs[pixInd] = np.append(self.imZs[pixInd], imProbeZ[i, pixInd])
                    elif curState == 'probe':
                        self.uCs[pixInd] = np.vstack((self.uCs[pixInd], np.zeros((2*self.gMat.nHalfModes, axis=1))))
                        self.reUps[pixInd] = np.vstack(self.reUps[pixInd], data['reProbeVecs'][i])
                        self.imUps[pixInd] = np.vstack(self.imUps[pixInd], data['imProbeVecs'][i])
                        self.reZs[pixInd] = np.append(self.reZs[pixInd], reProbeZ[i, pixInd])
                        self.imZs[pixInd] = np.append(self.imZs[pixInd], imProbeZ[i, pixInd])
                    else:
                        self.reUps[pixInd] = np.vstack(self.reUps[pixInd], data['reProbeVecs'][i])
                        self.imUps[pixInd] = np.vstack(self.imUps[pixInd], data['imProbeVecs'][i])
                        self.reZs[pixInd] = np.append(self.reZs[pixInd], reProbeZ[i, pixInd])
                        self.imZs[pixInd] = np.append(self.imZs[pixInd], imProbeZ[i, pixInd])
                        curState = 'probe'
        
        self.r = np.sqrt(np.average(reProbeZ)) #set measurement noise to avg poisson noise
        self.q = 1 #just hardcode this for now

        self.x = [np.array([]) for i in range(self.gMat.nPix)] 
        self.P = [np.array([]) for i in range(self.gMat.nPix)] 
        self.R = [np.array([]) for i in range(self.gMat.nPix)] 
        self.Q = [np.array([]) for i in range(self.gMat.nPix)] 

        for pixInd in range(self.gMat.nPix):
            assert self.reUps[pixInd].shape[0] == self.imUps[pixInd].shape[0] == self.uCs[pixInd].shape[0] 
                    == self.reZs[pixInd].shape[0] == self.imZs[pixInd].shape[0] 
            self.x = np.zeros((len(self.reZs[pixInd]), 2))
            self.P = np.zeros((len(self.reZs[pixInd]), 2, 2))
            self.R = np.zeros((len(self.reZs[pixInd]), 2, 2))
            self.Q = np.zeros((len(self.reZs[pixInd]), 2, 2))

        def applyKalman(self):
            for pixInd in range(self.gMat.nPix):
                #self.x[0] = np.array([0, 0])
                #self.P[0] = 1000*np.ones((2,2))
                #self.R[0] = self.r*np.ones((2,2))
                gSlice = self.gMat[[pixInd, pixInd+nPix], :]
                for i in range(len(self.reZs[pixInd])):
                    self.Q[pixInd, i] = self.q*np.ones((2,2)) #just hardcode all of the Qs for now
                    if i == 0:
                        xpr = np.dot(gSlice, self.uCs[pixInd, 0])
                        Ppr = 1000*np.ones((2,2))
                    else:
                        xpr = self.x[pixInd, i-1] + np.dot(gSlice, self.uCs[pixInd, 0])
                        Ppr = self.P[pixInd, i-1] + self.Q[pixInd, i]
                    self.R[pixInd, i] = self.r*np.diag(np.ones(2))
                    self.Q[pixInd, i] = self.q*np.diag(np.ones(2))
                    














        

