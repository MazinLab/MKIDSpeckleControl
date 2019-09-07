import speckpy
import mkidreadout.readout.sharedmem as shm
from mkidcore.corelog import getLogger, create_log
from mkidcore.objects import Beammap

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import scipy.ndimage as sciim

class Calibrator(object):

    def __init__(self, dmChanName, sharedImageName, useWvl=False, wvlStart=700, wvlStop=1400, beammap=None):
        self.dm = speckpy.SpeckleToDM(dmChanName)
        self.shmImage = shm.ImageCube(sharedImageName)
        self.shmImage.useWvl = useWvl
        
        if self.shmImage.useEdgeBins or self.shmImage.nWvlBins > 1:
            raise Exception('Multiple wavelengths/edge bins not implemented')

        if useWvl:
            self.shmImage.wvlStart = wvlStart
            self.shmImage.wvlStop = wvlStop

        if beammap:
            self.goodPixMask = ~beammap.failmask
        else:
            self.goodPixMask = np.ones(self.shmImage.shape)

    def run(self, start, end, amplitude, nPoints=10, integrationTime=5, lOverDEst=3, speckWin=None, angle=0):
        self.rotmat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.kvecs = np.array([np.linspace(start, end, nPoints), np.linspace(start, end, nPoints)]).T
        self.speckLocs = np.zeros((nPoints*2, 2, 2)) #NSpecklepairs x 2 speckles x [x, y]
        self.speckIntensities = np.zeros((nPoints*2, 2)) #NSpecklePairs x 2 speckles
        self.speckIntensities.fill(np.nan)
        self.amplitude = amplitude

        if speckWin is None:
            speckWin = int(np.ceil(lOverDEst))

        intensityCorrectionImage = sciim.gaussian_filter(self.goodPixMask.astype(np.float), lOverDEst*0.42)

        self.shmImage.startIntegration(0, integrationTime)
        refimage = self.shmImage.receiveImage()

        for i in range(nPoints):
            k0 = np.matmul(self.rotmat, np.array([self.kvecs[i,0], self.kvecs[i,1]]))
            k1 = np.matmul(self.rotmat, np.array([self.kvecs[i,0], -self.kvecs[i,1]]))
            self.dm.addProbeSpeckle(k0[0], k0[1], amplitude, 0)
            self.dm.addProbeSpeckle(k1[0], k1[1], amplitude, 0)
            #self.dm.addProbeSpeckle(self.kvecs[i,0], self.kvecs[i,1], amplitude, 0)
            #self.dm.addProbeSpeckle(self.kvecs[i,0], -self.kvecs[i,1], amplitude, 0)
            self.dm.updateDM()
            self.shmImage.startIntegration(0, integrationTime)
            image = self.shmImage.receiveImage() - refimage

            while(True):
                calgui = CalspotGUI(image)
                try:
                    self.speckLocs[i*2:i*2+2, :, :] = calgui.speckLocs
                except RuntimeError as err:
                    print err
                    continue

                break

            self.dm.clearProbeSpeckles()

            #image = image/intensityCorrectionImage
            for j in range(4):
                y = self.speckLocs[i*2 + j/2, j%2, 0]
                x = self.speckLocs[i*2 + j/2, j%2, 1]
                if ~np.isnan(x) and ~np.isnan(y):
                    self.speckIntensities[i*2 + j/2, j%2] = np.sum(image[int(y - np.floor(speckWin/2.)) : int(y + np.ceil(speckWin/2.)), 
                                             int(x - np.floor(speckWin/2.)) : int(x + np.ceil(speckWin/2.))])/(intensityCorrectionImage[int(y), int(x)]*integrationTime)

        self.dm.updateDM()



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

        self.nPixPerLD = np.pi*np.mean(np.sqrt(pairDiffs[:,0]**2 + pairDiffs[:,1]**2)/np.sqrt(kvecs[:,0]**2 + kvecs[:,1]**2))

    def calibrateIntensity(self):
        kMags = np.zeros((self.kvecs.shape[0]*2))
        kMags[::2] = np.sqrt(self.kvecs[:, 0]**2 + self.kvecs[:, 1]**2)
        kMags[1::2] = kMags[::2] #duplicate kvecs b/c we have 2 pairs per kvec
        intensities = np.mean(self.speckIntensities, axis=1)
        goodMask = ~np.isnan(intensities)
        a, b, c = np.polyfit(kMags[goodMask], self.amplitude**2/intensities[goodMask], 2)

        plt.plot(kMags[goodMask], intensities[goodMask], '.')
        x = np.linspace(kMags[0], kMags[-1])
        plt.plot(x, self.amplitude**2/(a*x**2 + b*x + c))
        plt.show()

        self.intensityCal = np.array([a, b, c])

    def writeToConfig(self, cfgFn):
        params = speckpy.PropertyTree()
        params.read_info(cfgFn)
        params.put('ImgParams.xCenter', self.center[1])
        params.put('ImgParams.yCenter', self.center[0])
        params.put('ImgParams.lambdaOverD', self.nPixPerLD)
        params.put('DMParams.a', self.intensityCal[0])
        params.put('DMParams.b', self.intensityCal[1])
        params.put('DMParams.c', self.intensityCal[2])
        params.write(cfgFn)
        
        
        

class CalspotGUI(object):
    
    def __init__(self, image):
        self._speckLocs = np.zeros((2, 2, 2)) # nPairs x 2 specks/pair x 2 coords/speck
        self._speckLocs.fill(np.nan)
        self.fig = plt.figure(figsize=(10,10))
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
        self._speckLocs[self.speckNum/2, self.speckNum%2, :] = np.array([event.ydata, event.xdata])
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
    beammap = Beammap(file='/home/scexao/mkids/20190905/finalMap_20181218.bmap', xydim=(140, 146))
    cal = Calibrator('dm00disp06', 'mkidshm1', beammap=beammap)
    #create_log(__name__)
    create_log('mkidreadout')
    cal.run(40, 50, 4, 5, 5, speckWin=5, angle=np.pi/4)
    cal.calculateCenter()
    cal.calculateLOverD()
    cal.calibrateIntensity()
    cal.writeToConfig('/home/scexao/mkids/20190906/speckNullConfig0_1.info')
    print 'center:', cal.center
    print 'l/D:', cal.nPixPerLD
    print 'calCoeffs'
    print '    a:', cal.intensityCal[0]
    print '    b:', cal.intensityCal[1]
    print '    c:', cal.intensityCal[2]
