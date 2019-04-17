import speckletodm
import propertytree
import mkidreadout.readout.sharedmem as shm
from mkidcore.corelog import getLogger, create_log

import numpy as np
import matplotlib.pyplot as plt
import os, sys

class Calibrator(object):

    def __init__(self, dmChanName, sharedImageName, useWvl=False, wvlStart=700, wvlStop=1400):
        self.dm = speckletodm.SpeckleToDM(dmChanName)
        self.shmImage = shm.ImageCube(sharedImageName)
        self.shmImage.useWvl = useWvl
        if useWvl:
            self.shmImage.wvlStart = wvlStart
            self.shmImage.wvlStop = wvlStop

    def run(self, start, end, amplitude, nPoints=10, integrationTime=5):
        self.kvecs = np.array([np.linspace(start, end, nPoints), np.linspace(start, end, nPoints)]).T
        self.speckLocs = np.zeros((nPoints*2, 2, 2))

        for i in range(nPoints):
            self.dm.addProbeSpeckle(self.kvecs[i,0], self.kvecs[i,1], amplitude, 0)
            self.dm.addProbeSpeckle(self.kvecs[i,0], -self.kvecs[i,1], amplitude, 0)
            self.dm.updateDM()
            self.shmImage.startIntegration(0, integrationTime)
            image = self.shmImage.receiveImage()

            while(True):
                calgui = CalspotGUI(image)
                try:
                    self.speckLocs[i*2:i*2+2, :, :] = calgui.speckLocs
                except RuntimeError:
                    continue

                break

            self.dm.clearProbeSpeckles()

    def calculateCenter(self):
        self.center = np.nanmean(self.speckLocs, axis=(0,1))

    def calculateLOverD(self):
        pairDiffs = np.diff(self.speckLocs, axis=1)
        pairDiffs = np.squeeze(pairDiffs, axis=1)
        goodPairMask = ~np.isnan(pairDiffs)
        pairDiffs = np.reshape(pairDiffs[goodPairMask], (-1, 2))
        
        kvecs = np.zeros((self.kvecs.shape[0]*2, 2))
        kvecs[::2,:] = self.kvecs
        kvecs[1::2,:] = self.kvecs #duplicate kvecs b/c we have 2 pairs per kvec
        kvecs = np.reshape(kvecs[goodPairMask], (-1, 2))

        self.nLDPerPix = np.pi*np.mean(np.sqrt(pairDiffs[:,0]**2 + pairDiffs[:,1]**2)/np.sqrt(kvecs[:,0]**2 + kvecs[:,1]**2))
        

class CalspotGUI(object):
    
    def __init__(self, image):
        self._speckLocs = np.zeros((2, 2, 2)) # nPairs x 2 specks/pair x 2 coords/speck
        self._speckLocs.fill(np.nan)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(image)

        print 'Click on the first speckle pair. (speckles in pair are across eachother from psf)'
        print 'Press r at any time to clear speckle locs and start over'
        self.speckNum = 0
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('key_press_event', self.restart)
        #self.fig.canvas.mpl_connect('close_event', self.onClose)
        plt.show()

    def onClick(self, event):
        self._speckLocs[self.speckNum/2, self.speckNum%2, :] = np.array([event.xdata, event.ydata])
        print 'Speckle pair: ', self.speckNum/2
        print 'Speckle pair coords: ', self._speckLocs[self.speckNum/2]
        self.speckNum += 1
        if self.speckNum == 2:
            print 'Click on the next speckle pair.'
        elif self.speckNum == 5:
            print 'Done. If you want to start over, click on the first speckle pair, else close this gui'

    def restart(self, event):
        if event.key=='r':
            print 'Restarting. Click on first speckle pair'
            self.speckNum = 0
            self._speckLocs.fill(np.nan)

    @property
    def speckLocs(self):
        for i in range(2):
            if not(np.all(np.isnan(self._speckLocs[i])) or np.all(~np.isnan(self._speckLocs[i]))):
                raise RuntimeError('Must click a conjugate pair for each identified speckle! Restarting...')
        return self._speckLocs

if __name__=='__main__':
    cal = Calibrator('dm04disp00', 'DMCalTest0')
    #create_log(__name__)
    create_log('mkidreadout')
    cal.run(20, 60, 1, 5, 30)
    cal.calculateCenter()
    cal.calculateLOverD()
    print 'center:', cal.center
    print 'l/D', cal.nLDPerPix
