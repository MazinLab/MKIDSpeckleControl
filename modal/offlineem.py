from gmat import GMat

import numpy as np
import os
import time
import numpy.linalg as nlg
import matplotlib.pyplot as plt
import pickle as pkl
import ipdb
import copy
import multiprocessing
import traceback
from functools import partial

class OfflineEM(object):
    def __init__(self, tsString, path='.'):
        with open(os.path.join(path, 'gmat_{}.p'.format(tsString))) as f:
            self.gMat = pkl.load(f)

        data = np.load(os.path.join(path, 'cal_data_{}.npz'.format(tsString)))
        nIters = len(data['ctrlVecs'])

        self.reZs = [np.zeros(nIters) for i in range(self.gMat.nPix)]
        self.imZs = [np.zeros(nIters) for i in range(self.gMat.nPix)]
        self.reUpInds = [np.zeros(nIters, dtype=int) for i in range(self.gMat.nPix)]
        self.imUpInds = [np.zeros(nIters, dtype=int) for i in range(self.gMat.nPix)]
        self.uCInds = [[[] for j in range(nIters)] for i in range(self.gMat.nPix)] 

        self.reUps = data['reProbeVecs']
        self.imUps = data['imProbeVecs']
        self.uCs = data['ctrlVecs']


        reProbeZ = data['reProbeImgs']
        imProbeZ = data['imProbeImgs']
        badPixMaskList = self.gMat.badPixMask.flatten()
        reProbeZ = np.reshape(reProbeZ, (-1, len(badPixMaskList)))
        imProbeZ = np.reshape(imProbeZ, (-1, len(badPixMaskList)))
        reProbeZ = reProbeZ[:, ~badPixMaskList]
        imProbeZ = imProbeZ[:, ~badPixMaskList]

        probePixMasks = np.zeros((self.gMat.nPix, nIters))
        ctrlPixMasks = np.zeros((self.gMat.nPix, nIters))

        for i in range(nIters):
            probePixMasks[:, i] = self.gMat.getPixMaskFromModeVec(data['reProbeVecs'][i])
            ctrlPixMasks[:, i] = self.gMat.getPixMaskFromModeVec(data['ctrlVecs'][i])


        for pixInd in range(self.gMat.nPix):
            curState = None
            curIter = 0
            print 'pixInd: {}/{}'.format(pixInd, self.gMat.nPix)
            for i in range(nIters):
                if ctrlPixMasks[pixInd, i]:
                    curState = 'ctrl'
                    self.uCInds[pixInd][curIter].append(i) 
                    #print 'iter', i, 'ctrl'

                if probePixMasks[pixInd, i]:
                    if curState is None or curState == 'probe':
                        self.uCInds[pixInd][curIter].append(-1)
                    self.reUpInds[pixInd][curIter] = i 
                    self.imUpInds[pixInd][curIter] = i 
                    self.reZs[pixInd][curIter] = reProbeZ[i, pixInd]
                    self.imZs[pixInd][curIter] = imProbeZ[i, pixInd]
                    #print 'iter', i , 'probe'
                    curState = 'probe'
                    curIter += 1

            self.reUpInds[pixInd] = np.delete(self.reUpInds[pixInd], range(curIter, nIters), axis=0)
            self.imUpInds[pixInd] = np.delete(self.imUpInds[pixInd], range(curIter, nIters), axis=0)
            self.reZs[pixInd] = np.delete(self.reZs[pixInd], range(curIter, nIters), axis=0)
            self.imZs[pixInd] = np.delete(self.imZs[pixInd], range(curIter, nIters), axis=0)
            del(self.uCInds[pixInd][curIter:])


        
        self.r = 200 #np.sqrt(np.average(reProbeZ)) #set measurement noise to avg poisson noise
        self.q = 0.2 #just hardcode this for now

        self.x = [np.array([]) for i in range(self.gMat.nPix)] 
        self.P = [np.array([]) for i in range(self.gMat.nPix)] 
        self.R = [np.array([]) for i in range(self.gMat.nPix)] 
        self.Q = [np.array([]) for i in range(self.gMat.nPix)] 
        self.reExpZ = [np.array([]) for i in range(self.gMat.nPix)]
        self.imExpZ = [np.array([]) for i in range(self.gMat.nPix)]
        self.reExpZCtrl = [np.array([]) for i in range(self.gMat.nPix)]
        self.imExpZCtrl = [np.array([]) for i in range(self.gMat.nPix)]
        self.expXCtrl = [np.array([]) for i in range(self.gMat.nPix)]
        self.reZResid = [np.array([]) for i in range(self.gMat.nPix)] 
        self.imZResid = [np.array([]) for i in range(self.gMat.nPix)] 

        for pixInd in range(self.gMat.nPix):
            assert self.reUpInds[pixInd].shape[0] == self.imUpInds[pixInd].shape[0] == len(self.uCInds[pixInd]) == self.reZs[pixInd].shape[0] == self.imZs[pixInd].shape[0] 
            self.x[pixInd] = np.zeros((len(self.reZs[pixInd]), 2))
            self.P[pixInd] = np.zeros((len(self.reZs[pixInd]), 2, 2))
            self.R[pixInd] = np.zeros(len(self.reZs[pixInd]))
            self.Q[pixInd] = np.zeros((len(self.reZs[pixInd]), 2, 2))
            self.reExpZ[pixInd] = np.zeros(len(self.reZs[pixInd]))
            self.imExpZ[pixInd] = np.zeros(len(self.reZs[pixInd]))
            self.reExpZCtrl[pixInd] = np.zeros(len(self.reZs[pixInd]))
            self.imExpZCtrl[pixInd] = np.zeros(len(self.reZs[pixInd]))
            self.expXCtrl[pixInd] = np.zeros((len(self.reZs[pixInd]), 2))

    def __getitem__(self, inds):
        emOpt = copy.copy(self)
        emOpt.x = copy.deepcopy(self.x[inds])
        emOpt.P = copy.deepcopy(self.P[inds])
        emOpt.R = copy.deepcopy(self.R[inds])
        emOpt.Q = copy.deepcopy(self.Q[inds])
        emOpt.reExpZ = copy.deepcopy(self.reExpZ[inds])
        emOpt.imExpZ = copy.deepcopy(self.imExpZ[inds])
        emOpt.imExpZCtrl = copy.deepcopy(self.imExpZCtrl[inds])
        emOpt.reExpZCtrl = copy.deepcopy(self.reExpZCtrl[inds])
        emOpt.expXCtrl = copy.deepcopy(self.expXCtrl[inds])
        emOpt.reZResid = copy.deepcopy(self.reZResid[inds])
        emOpt.imZResid = copy.deepcopy(self.imZResid[inds])

        emOpt.reZs = copy.deepcopy(self.reZs[inds])
        emOpt.imZs = copy.deepcopy(self.imZs[inds])
        emOpt.reUpInds = copy.deepcopy(self.reUpInds[inds])
        emOpt.imUpInds = copy.deepcopy(self.imUpInds[inds])
        emOpt.uCInds = copy.deepcopy(self.uCInds[inds])

        emOpt.reUps = copy.deepcopy(self.reUps)
        emOpt.imUps = copy.deepcopy(self.imUps)
        emOpt.uCs = copy.deepcopy(self.uCs)

        emOpt.gMat = self.gMat[inds]

        return emOpt

    def __setitem__(self, inds, emOpt):
        if isinstance(inds, int):
            if emOpt.gMat.nPix != 1:
                raise Exception('Cannot copy {} pixels to 1 pixel dest slice'.format(emOpt.gMat.nPix))
        elif isinstance(inds, slice):
            nPixSlice = inds.stop - inds.start
            if emOpt.gMat.nPix != nPixSlice:
                raise Exception('Cannot copy {} pixels to {} pixel dest slice'.format(emOpt.gMat.nPix, nPixSlice))
        self.x[inds] = copy.deepcopy(emOpt.x)
        self.P[inds] = copy.deepcopy(emOpt.P)
        self.R[inds] = copy.deepcopy(emOpt.R)
        self.Q[inds] = copy.deepcopy(emOpt.Q)
        self.reExpZ[inds] = copy.deepcopy(emOpt.reExpZ)
        self.imExpZ[inds] = copy.deepcopy(emOpt.imExpZ)
        self.imExpZCtrl[inds] = copy.deepcopy(emOpt.imExpZCtrl)
        self.reExpZCtrl[inds] = copy.deepcopy(emOpt.reExpZCtrl)
        self.expXCtrl[inds] = copy.deepcopy(emOpt.expXCtrl)
        self.reZResid[inds] = copy.deepcopy(emOpt.reZResid)
        self.imZResid[inds] = copy.deepcopy(emOpt.imZResid)

        self.reZs[inds] = copy.deepcopy(emOpt.reZs)
        self.imZs[inds] = copy.deepcopy(emOpt.imZs)
        self.reUpInds[inds] = copy.deepcopy(emOpt.reUpInds)
        self.imUpInds[inds] = copy.deepcopy(emOpt.imUpInds)
        self.uCInds[inds] = copy.deepcopy(emOpt.uCInds)

        self.gMat[inds] = emOpt.gMat

    def applyKalman(self, initPixInd=None):
        if initPixInd is None:
            pixRange = range(self.gMat.nPix)
        else:
            pixRange = [initPixInd]

        for pixInd in pixRange:
            #self.x[0] = np.array([0, 0])
            #self.P[0] = 1000*np.ones((2,2))
            #self.R[0] = self.r*np.ones((2,2))
            gSlice = self.gMat.mat[[pixInd, pixInd+self.gMat.nPix], :]
            for i in range(len(self.reZs[pixInd])):
                uc = self._getUc(self.uCInds[pixInd][i])
                if i == 0:
                    xpr = np.dot(gSlice, uc)
                    Ppr = 100000*np.diag(np.ones(2))
                    self.expXCtrl[pixInd][i] = np.dot(gSlice, uc)
                else:
                    xpr = self.x[pixInd][i-1] + np.dot(gSlice, uc)
                    Ppr = self.P[pixInd][i-1] + self.Q[pixInd][i]
                    self.expXCtrl[pixInd][i] = self.expXCtrl[pixInd][i-1] + np.dot(gSlice, uc)
                self.R[pixInd][i] = self.r
                self.Q[pixInd][i] = self.q*np.diag(np.ones(2))


                Hre = np.expand_dims(4*np.dot(gSlice, self.reUps[self.reUpInds[pixInd][i]]), axis=1).T
                Kre = np.dot(Ppr, np.dot(Hre.T, nlg.inv(np.dot(Hre, np.dot(Ppr, Hre.T)) + self.R[pixInd][i])))
                self.x[pixInd][i] = xpr + np.squeeze(Kre*(self.reZs[pixInd][i] - np.dot(Hre, xpr)))
                self.P[pixInd][i] = np.dot(np.diag(np.ones(2)) - np.dot(Kre, Hre), Ppr)
                self.reExpZ[pixInd][i] = np.dot(Hre, xpr)
                self.reExpZCtrl[pixInd][i] = np.dot(Hre, self.expXCtrl[pixInd][i])
                
                Him = np.expand_dims(4*np.dot(gSlice, self.imUps[self.imUpInds[pixInd][i]]), axis=1).T
                Kim = np.dot(Ppr, np.dot(Him.T, nlg.inv(np.dot(Him, np.dot(Ppr, Him.T)) + self.R[pixInd][i])))
                self.x[pixInd][i] += np.squeeze(Kim*(self.imZs[pixInd][i] - np.dot(Him, xpr)))
                self.P[pixInd][i] = np.dot(np.diag(np.ones(2)) - np.dot(Kim, Him), self.P[pixInd][i])
                self.imExpZ[pixInd][i] = np.dot(Him, xpr)
                self.imExpZCtrl[pixInd][i] = np.dot(Him, self.expXCtrl[pixInd][i])

    def applyMStep(self, batchSize=50, learningRate=1.e-3, initPixInd=None, reg=1.e-1):
        for i in range(batchSize):
            if initPixInd is None:
                pixInd = np.random.choice(self.gMat.nPix)
            else:
                pixInd = initPixInd
            iterInd = np.random.choice(range(1, len(self.reZs[pixInd])))
            prInd = np.random.choice(2)

            #x = np.zeros(2*self.gMat.nPix)
            #xprev = np.zeros(2*self.gMat.nPix)
            #x[pixInd] = self.x[pixInd][iterInd, 0]
            #x[pixInd + self.gMat.nPix] = self.x[pixInd][iterInd, 1]
            #xprev[pixInd] = self.x[pixInd][iterInd-1, 0]
            #xprev[pixInd + self.gMat.nPix] = self.x[pixInd][iterInd-1, 1]

            x = np.expand_dims(self.x[pixInd][iterInd], axis=1)
            xprev = np.expand_dims(self.x[pixInd][iterInd - 1], axis=1)

            if prInd==0:
                up = np.expand_dims(self.reUps[self.reUpInds[pixInd][iterInd]], axis=1)
                z = self.reZs[pixInd][iterInd]
            else:
                up = np.expand_dims(self.imUps[self.imUpInds[pixInd][iterInd]], axis=1)
                z = self.imZs[pixInd][iterInd]
            
            q = self.Q[pixInd][iterInd, prInd, prInd]
            r = self.R[pixInd][iterInd]
            uc = np.expand_dims(self._getUc(self.uCInds[pixInd][iterInd]), axis=1)

            gSlice = self.gMat.mat[[pixInd, pixInd + self.gMat.nPix]] #add restrict nonzero
            gsMask = gSlice != 0

            self.gMat.mat[[pixInd, pixInd + self.gMat.nPix]] += learningRate/q*(np.dot(x - xprev, uc.T) 
                    - np.dot(np.dot(gSlice, uc), uc.T))*gsMask 
            self.gMat.mat[[pixInd, pixInd + self.gMat.nPix]] += learningRate/r*(4*z*np.dot(x, up.T) 
                    - 16*np.dot(x, np.dot(x.T, np.dot(np.dot(gSlice, up), up.T))))*gsMask - learningRate*reg*gSlice

            #if np.any(np.abs(self.gMat.mat[[pixInd, pixInd + self.gMat.nPix]]) > 30): #todo: change this hardcode and make gSlice more accessible
            #    ipdb.set_trace()

    
    def runEM(self, learningRate=5.e-4, batchSize=50, nIters=1000, lrDecay=10, initPixInd=None, queue=None):
        if initPixInd is None:
            pixRange = range(self.gMat.nPix)
        else:
            pixRange = [initPixInd]

        for pixInd in pixRange:
            self.reZResid[pixInd] = np.zeros(nIters)
            self.imZResid[pixInd] = np.zeros(nIters)
            for i in range(nIters):
                self.applyKalman(pixInd)
                r = lrDecay**(i/float(nIters))
                self.applyMStep(batchSize, learningRate/r, pixInd)
                self.reZResid[pixInd][i] = np.mean((self.reZs[pixInd] - self.reExpZ[pixInd])**2)
                self.imZResid[pixInd][i] = np.mean((self.imZs[pixInd] - self.imExpZ[pixInd])**2)
                #print 'pixInd: {}; iter: {}'.format(pixInd, i)
                #print '    reResid:', self.reZResid[pixInd][i]
                #print '    imResid:', self.imZResid[pixInd][i]

            if queue is not None:
                queue.put(1)
            else:
                print 'done pix {}/{}'.format(pixInd, self.gMat.nPix-1)



    def _getUc(self, inds):
        if inds[0] == -1:
            return np.zeros(2*self.gMat.nHalfModes)
        return np.sum(self.uCs[inds], axis=0)

def _runEM(emOpt, learningRate=5.e-4, batchSize=50, nIters=1000, lrDecay=10, queue=None):
    #wrapper function for multiprocessing
    try:
        emOpt.runEM(learningRate, batchSize, nIters, lrDecay, queue=queue)
    except Exception as e:
        traceback.print_exc()
        raise e
    return emOpt

def runEMMultProc(emOpt, ncpu, learningRate=5.e-4, batchSize=50, nIters=1000, lrDecay=10):
    cpuFactor = 5
    chunkSize = int(np.ceil(emOpt.gMat.nPix/float(cpuFactor*ncpu)))
    nChunks = int(cpuFactor*ncpu)
    emOptList = []
    procList = []

    for i in range(nChunks):
        endInd = min(emOpt.gMat.nPix, (i+1)*chunkSize)
        emOptList.append(emOpt[i*chunkSize:endInd])

    pool = multiprocessing.Pool(processes=ncpu)
    man = multiprocessing.Manager()
    q = man.Queue()

    runEMChunk = partial(_runEM, learningRate=learningRate, batchSize=batchSize, 
            nIters=nIters, lrDecay=lrDecay, queue=q)
    asyncRes = pool.map_async(runEMChunk, emOptList, chunksize=1)

    nPixDone = 0
    while not asyncRes.ready():
        if not q.empty():
            nPixDone += q.get()
            print 'done {}/{} pixels'.format(nPixDone, emOpt.gMat.nPix)
        time.sleep(0.1)

    print 'done!'

    asyncRes.wait()
    emOptListDone = asyncRes.get()


    for i in range(nChunks):
        endInd = min(emOpt.gMat.nPix, (i+1)*chunkSize)
        emOpt[i*chunkSize:endInd] = emOptListDone[i]

    return emOpt


