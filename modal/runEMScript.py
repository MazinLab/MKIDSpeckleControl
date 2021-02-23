import offlineem as em
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import time

emOpt = em.OfflineEM('20210130-032032', '/home/neelay/data/20210129/') 
emOpt.gMat.recomputeMat(1, 0.9, 7)
#start = time.time()
learningRate = 1.e-4
batchSize = 200
nIters = 2000
lrDecay = 1
em.runEMMultProc(emOpt, 20, learningRate=learningRate, batchSize=batchSize, nIters=nIters, lrDecay=lrDecay)
reDelta = []
imDelta = []
for i in range(emOpt.gMat.nPix):
    reDelta.append(emOpt.reZResid[i][-1]/emOpt.reZResid[i][0])
    imDelta.append(emOpt.imZResid[i][-1]/emOpt.imZResid[i][0])

plt.plot(reDelta)
plt.plot(imDelta)
plt.ylim(0,2)
plt.savefig('emOpt_20210130-032032_lr{}en5_b{}_n{}_lrd{}_r0p5_es0p02n50_2.png'.format(int(learningRate*1.e5), batchSize, nIters, lrDecay))
plt.show()

pkl.dump(emOpt, open('emOpt_20210130-032032_lr{}en5_b{}_n{}_lrd{}_r0p5_es0p02n50.p'.format(int(learningRate*1.e5), batchSize, nIters, lrDecay), 'w'))

#pixInd = 81
#emOpt.runEM(learningRate=learningRate, batchSize=batchSize, nIters=nIters, lrDecay=lrDecay, initPixInd=81)
#plt.plot(emOpt.reZResid[pixInd])
#plt.plot(emOpt.imZResid[pixInd])
#plt.show()

