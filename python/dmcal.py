import speckletodm
import propertytree
import mkidreadout.readout.sharedmem as shm

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
        kvecs = np.array([np.linspace(start, end, nPoints), np.linspace(start, end, nPoints)]).T
        self.speckLocs = np.zeros((nPoints*2, 2, 2))

        for i in range(nPoints):
            self.dm.addProbeSpeckle(kvecs[i,0], kvecs[i,1], amplitude, 0)
            self.dm.addProbeSpeckle(kvecs[i,0], -kvecs[i,1], amplitude, 0)
            self.dm.updateDM()
            self.shmImage.startIntegration(0, integrationTime)
            image = self.shmImage.receiveImage()
            calgui = CalspotGUI(image)
            self.speckLocs[i*2:i*2+2, :, :] = calgui.speckLocs
            self.dm.clearProbeSpeckles()
        

class CalspotGUI(object):
    
    def __init__(self, image):
        self.speckLocs = np.zeros((2, 2, 2)) # nPairs x 2 specks/pair x 2 coords/speck
        self.speckLocs.fill(np.nan)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(image)

        print 'Click on the first speckle pair. (speckle pairs are across eachother from psf)'
        print 'Press r at any time to clear speckle locs and start over'
        self.speckNum = 0
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('key_press_event', self.restart)
        plt.show()

    def onClick(self, event):
        self.speckLocs[self.speckNum/2, self.speckNum%2, :] = np.array([event.xdata, event.ydata])
        print 'Speckle pair: ', self.speckNum/2
        print 'Speckle pair coords: ', self.speckLocs[self.speckNum/2]
        self.speckNum += 1
        if self.speckNum == 2:
            print 'Click on the next speckle pair.'
        elif self.speckNum == 5:
            print 'Done. If you want to start over, click on the first speckle pair, else close this gui'

    def restart(self, event):
        if event.key=='r':
            print 'Restarting. Click on first speckle pair'
            self.speckNum = 0
            self.speckLocs.fill(np.nan)

if __name__=='__main__':
    cal = Calibrator('dm04disp00', 'DMCalTest0')
    cal.run(10, 30, 1, 3)
