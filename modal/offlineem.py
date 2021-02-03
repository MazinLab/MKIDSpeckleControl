from gmat import GMat
from calibrator import Calibrator

import numpy as np
import os
import numpy.linalg as nlg
import matplotlib.pyplot as plt
import pickle as pkl

class OfflineEM(object):
    def __init__(self, tsString, path='.'):
        with open(os.path.join(path, 'gmat_{}.p'.format(tsString))) as f:
            self.gMat = pkl.load(f)

        data = np.load(os.path.join(path, 'cal_data_{}.npz'.format(tsString)))
        nIters = len(data['ctrlVecs'])

        self.reZs = [np.zeros(nIters) for i in range(self.gMat.nPix)]
        self.imZs = [np.zeros(nIters) for i in range(self.gMat.nPix)]
        self.reUps = [np.zeros((nIters, 2*self.gMat.nHalfModes)) for i in range(self.gMat.nPix)]
        self.imUps = [np.zeros((nIters, 2*self.gMat.nHalfModes)) for i in range(self.gMat.nPix)]
        self.uCs = [np.zeros((nIters, 2*self.gMat.nHalfModes)) for i in range(self.gMat.nPix)] 


        reProbeZ = data['reProbeImgs']
        imProbeZ = data['imProbeImgs']
        badPixMaskList = self.gMat.badPixMask.flatten()
        reProbeZ = np.reshape(reProbeZ, (-1, len(badPixMaskList)))
        imProbeZ = np.reshape(imProbeZ, (-1, len(badPixMaskList)))
        reProbeZ = reProbeZ[:, badPixMaskList]
        imProbeZ = imProbeZ[:, badPixMaskList]


        for pixInd in range(self.gMat.nPix):
            curState = None
            curIter = 0
            for i in range(nIters):
                if self.gMat.checkPixModeVec(data['ctrlVecs'][i], pixInd):
                    curState = 'ctrl'
                    self.uCs[pixInd][curIter] += data['ctrlVecs'][i]

                if self.gMat.checkPixModeVec(data['reProbeVecs'][i], pixInd):
                    if curState is None or curState == 'probe':
                        self.uCs[pixInd][curIter] = np.zeros(2*self.gMat.nHalfModes)
                    self.reUps[pixInd][curIter] = data['reProbeVecs'][i]
                    self.imUps[pixInd][curIter] = data['imProbeVecs'][i]
                    self.reZs[pixInd][curIter] = reProbeZ[i, pixInd]
                    self.imZs[pixInd][curIter] = imProbeZ[i, pixInd]
                    curState = 'probe'
                    curIter += 1

            self.reUps[pixInd] = np.delete(self.reUps[pixInd], range(curIter, nIters), axis=0)
            self.imUps[pixInd] = np.delete(self.imUps[pixInd], range(curIter, nIters), axis=0)
            self.reZs[pixInd] = np.delete(self.reZs[pixInd], range(curIter, nIters), axis=0)
            self.imZs[pixInd] = np.delete(self.imZs[pixInd], range(curIter, nIters), axis=0)


        
        self.r = np.sqrt(np.average(reProbeZ)) #set measurement noise to avg poisson noise
        self.q = 1 #just hardcode this for now

        self.x = [np.array([]) for i in range(self.gMat.nPix)] 
        self.P = [np.array([]) for i in range(self.gMat.nPix)] 
        self.R = [np.array([]) for i in range(self.gMat.nPix)] 
        self.Q = [np.array([]) for i in range(self.gMat.nPix)] 

        for pixInd in range(self.gMat.nPix):
            assert self.reUps[pixInd].shape[0] == self.imUps[pixInd].shape[0] == self.uCs[pixInd].shape[0] == self.reZs[pixInd].shape[0] == self.imZs[pixInd].shape[0] 
            self.x = np.zeros((len(self.reZs[pixInd]), 2))
            self.P = np.zeros((len(self.reZs[pixInd]), 2, 2))
            self.R = np.zeros((len(self.reZs[pixInd]), 2, 2))
            self.Q = np.zeros((len(self.reZs[pixInd]), 2, 2))

    def applyKalman(self):
        for pixInd in range(self.gMat.nPix):
            #self.x[0] = np.array([0, 0])
            #self.P[0] = 1000*np.ones((2,2))
            #self.R[0] = self.r*np.ones((2,2))
            gSlice = self.gMat.mat[[pixInd, pixInd+self.gMat.nPix], :]
            for i in range(len(self.reZs[pixInd])):
                if i == 0:
                    xpr = np.dot(gSlice, self.uCs[pixInd, 0])
                    Ppr = 1000*np.ones((2,2))
                else:
                    xpr = self.x[pixInd, i-1] + np.dot(gSlice, self.uCs[pixInd, 0])
                    Ppr = self.P[pixInd, i-1] + self.Q[pixInd, i]
                self.R[pixInd, i] = self.r*np.diag(np.ones(2))
                self.Q[pixInd, i] = self.q*np.diag(np.ones(2))

                Hre = 4*np.dot(gSlice, self.reUps[pixInd, i]).T
                Kre = np.dot(Ppr, np.dot(Hre.T, nlg.inv(np.dot(Hre, np.dot(Ppr, Hre.T)) + self.R[pixInd, i])))
                self.x[pixInd, i] = xpr + Kre*(self.reZs[pixInd, i] - np.dot(Hre, xpr))
                self.P[pixInd, i] = np.dot(np.diag(np.ones(2)) - np.dot(Kre, Hre), Ppr)
                
                Him = 4*np.dot(gSlice, self.imUps[pixInd, i]).T
                Kim = np.dot(Ppr, np.dot(Him.T, nlg.inv(np.dot(Him, np.dot(Ppr, Him.T)) + self.R[pixInd, i])))
                self.x[pixInd, i] += Kim*(self.imZs[pixInd, i] - np.dot(Him, xpr))
                self.P[pixInd, i] = np.dot(np.diag(np.ones(2)) - np.dot(Kim, Him), self.P[pixInd, i])

    def applyMStep(batchSize=50, learningRate=0.01):
        for i in range(batchSize):
            pixInd = np.random.choice(self.gMat.nPix)
            iterInd = np.random.choice(range(1, len(self.reZs[pixInd])))
            prInd = np.random.choice(2)

            x = np.zeros(2*self.gMat.nPix)
            xprev = np.zeros(2*self.gMat.nPix)
            x[pixInd] = self.x[pixInd, iterInd, 0]
            x[pixInd + self.gMat.nPix] = self.x[pixInd, iterInd, 1]
            xprev[pixInd] = self.x[pixInd, iterInd-1, 0]
            xprev[pixInd + self.gMat.nPix] = self.x[pixInd, iterInd-1, 1]

            if prInd==0:
                up = self.reUps[pixInd, iterInd]
                z = self.reZs[pixInd, iterInd]
            else:
                up = self.imUps[pixInd, iterInd]
                z = self.imZs[pixInd, iterInd]
            
            q = self.Q[pixInd, iterInd, prInd, prInd]
            r = self.R[pixInd, iterInd, prInd, prInd]
            uc = self.uCs[pixInd, iterInd]

            self.gMat.mat += learningRate/q*(np.dot(x - xprev, uc.T) - 2*np.dot(self.gMat.mat, np.dot(uc, uc.T)))
            self.gMat.mat += learningRate/r*(4*z*np.dot(x, up.T) - 8*np.dot(np.dot(x, x.T), np.dot(self.gMat.mat, np.dot(up, up.T))))




                    














        

