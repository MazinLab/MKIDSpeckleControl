import offlineem as em
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import time

emOpt = em.OfflineEM('20210130-032032', '/home/neelay/data/20210129/') 
#start = time.time()
learningRate = 5.e-4
batchSize = 200
nIters = 5000
lrDecay = 3
stopNIters = 50
stopFactor = 0.01
corrWin = 5
r = 200
q = 0.1
beta = .2
emOpt.gMat.recomputeMat(beta, 0.5, corrWin)
em.runEMMultProc(emOpt, 20, learningRate=learningRate, batchSize=batchSize, nIters=nIters, lrDecay=lrDecay, stopNIters=stopNIters, stopFactor=stopFactor, measVar=r, procVar=q)
reDelta = []
imDelta = []
for i in range(emOpt.gMat.nPix):
    reDelta.append(emOpt.reZResid[i][-1]/emOpt.reZResid[i][0])
    imDelta.append(emOpt.imZResid[i][-1]/emOpt.imZResid[i][0])

plt.plot(reDelta)
plt.plot(imDelta)
plt.ylim(0,2)
plt.savefig('emOpt_20210130-032032_lr{}en5_b{}_n{}_lrd{}_r0_es0p0{}n{}_cw{}_beta{}_rs0p5_r{}_q{}.png'.format(int(learningRate*1.e5), batchSize, nIters, lrDecay, int(100*stopFactor), stopNIters, corrWin, beta, r, q))
plt.show()

pkl.dump(emOpt, open('emOpt_20210130-032032_lr{}en5_b{}_n{}_lrd{}_r0_es0p0{}n{}_cw{}_beta{}_rs0p5_r{}_q{}.p'.format(int(learningRate*1.e5), batchSize, nIters, lrDecay, int(100*stopFactor), stopNIters, corrWin, beta, r, q), 'w'))

#pixInd = 185
#emOpt.runEM(learningRate=learningRate, batchSize=batchSize, nIters=nIters, lrDecay=lrDecay, initPixInd=pixInd, stopNIters=stopNIters, stopFactor=stopFactor, measVar=r, procVar=q)
#plt.plot(emOpt.reZResid[pixInd])
#plt.plot(emOpt.imZResid[pixInd])
#plt.show()
#plt.plot(emOpt.gMat.mat[pixInd])
#plt.plot(emOpt.gMat.mat[pixInd+emOpt.gMat.nPix])
#plt.show()
