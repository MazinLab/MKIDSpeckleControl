"""
Cal script for MKIDS/SCEXAO interface. Uses python interfaces to MKID shared memory stream and 
CACAO DM channels. Calibrates PSF center, lambda/D, and DM amplitude -> intensity on detector.
Angle is assumed to be 0. Configuration parameters are specified in file (see dmcal_example.cfg). 
Modes:
    full - uses waffle to calibrate center, intensity, and l/D. Must click on speckles in
        pairs that are across each other.
    center - uses a single speckle pair for calibration. Useful for center cal when PSF is
        off or one one side of the array. Use with caution, code is complicated and might have bugs.
    intensity - uses waffle for intensity cal only. Speckle pair restriction from "full" mode
        not enforced.

"""


import speckpy
from mkidcore.readdict import ReadDict
import mkidreadout.readout.sharedmem as shm
from mkidcore.corelog import getLogger, create_log
from mkidcore.objects import Beammap

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import os, sys
import scipy.ndimage as sciim
import argparse

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

    def run(self, start, end, amplitude, nPoints=10, integrationTime=5, lOverDEst=None, speckWin=None, angle=0, calType='full'):
        calType = calType.lower()
        assert calType=='full' or calType=='intensity' or calType=='center'
        self.calType = calType

        if lOverDEst is None:
            if calType=='center':
                raise Exception('Must provide l/D estimate for center calibration. If unknown run "full" cal')
            lOverDEst = 3

        if calType == 'center':
            self.nPixPerLD = lOverDEst
            useWaffle = False #only use single pair of speckles for center cal. maybe make this a parameter
            nPairs = 1
            enforcePairs = False

        elif calType == 'intensity':
            useWaffle = True
            nPairs = 2
            enforcePairs = False

        else:
            useWaffle = True
            nPairs = 2
            enforcePairs = True

        self.angle = angle

        rotmat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        if useWaffle:
            self.kvecs = np.array([np.linspace(start, end, nPoints), np.zeros(nPoints)]).T
            self.kvecs = np.stack((self.kvecs, np.array([np.zeros(nPoints), np.linspace(start, end, nPoints)]).T), axis=1)
            self.kvecs = np.matmul(rotmat, self.kvecs).transpose((0, 2, 1)) #dims are (nPoints, 2 kvecs per point, 2 coords per kvec)
            self.speckLocs = np.zeros((nPoints, 2, 2, 2)) #nPoints x 2 pairs x 2 speckles x [x, y]
            self.speckIntensities = np.zeros((nPoints, 2, 2)) #nPoints x 2 pairs x 2 speckles
            self.speckIntensities.fill(np.nan)
        else:
            self.kvecs = np.array([np.linspace(start, end, nPoints), np.zeros(nPoints)])
            self.kvecs = np.matmul(rotmat, self.kvecs).T #dims are (nPoints, 2 coords per kvec)
            self.kvecs = np.expand_dims(self.kvecs, 1) #dims are now (nPoints, 1 pair, 2 coords per kvec)
            self.speckLocs = np.zeros((nPoints, 1, 2, 2)) #nPoints x 1 pair x 2 speckles 2 [x, y]
            self.speckIntensities = np.zeros((nPoints, 1, 2))
            self.speckIntensities.fill(np.nan)

        self.amplitude = amplitude

        if speckWin is None:
            speckWin = int(np.ceil(lOverDEst))

        intensityCorrectionImage = sciim.gaussian_filter(self.goodPixMask.astype(np.float), lOverDEst*0.42)

        self.shmImage.startIntegration(0, integrationTime)
        refimage = self.shmImage.receiveImage()

        for i in range(nPoints):
            if useWaffle:
                k0 = self.kvecs[i, 0]
                k1 = self.kvecs[i, 1]
                self.dm.addProbeSpeckle(k0[0], k0[1], amplitude, 0)
                self.dm.addProbeSpeckle(k1[0], k1[1], amplitude, 0)
            else:
                self.dm.addProbeSpeckle(self.kvecs[i,0,0], self.kvecs[i,0,1], amplitude, 0)

            self.dm.updateDM()
            self.shmImage.startIntegration(0, integrationTime)
            image = self.shmImage.receiveImage() - refimage

            while(True):
                calgui = CalspotGUI(image, enforcePairs=enforcePairs, nPairs=1+useWaffle)
                try:
                    self.speckLocs[i] = calgui.speckLocs
                except RuntimeError as err:
                    print err
                    continue

                break

            self.dm.clearProbeSpeckles()

            #image = image/intensityCorrectionImage
            for j in range(nPairs):
                for k in range(2):
                    y = self.speckLocs[i, j, k, 0]
                    x = self.speckLocs[i, j, k, 1]
                    if ~np.isnan(x) and ~np.isnan(y):
                        self.speckIntensities[i, j, k] = np.sum(image[int(y - np.floor(speckWin/2.)) : int(y + np.ceil(speckWin/2.)), 
                                                 int(x - np.floor(speckWin/2.)) : int(x + np.ceil(speckWin/2.))])/(intensityCorrectionImage[int(y), int(x)]*integrationTime)

        self.dm.updateDM()
        if self.calType == 'full':
            self.calculateLOverD()
            self.calculateCenter()
            self.calibrateIntensity()
        elif self.calType == 'center':
            self.calculateCenter()
            self.calibrateIntensity()
        else:
            self.calibrateIntensity()




    def calculateCenter(self):
        if self.calType == 'full':
            self.center = np.nanmean(self.speckLocs, axis=(0,1,2))
        elif self.calType == 'center':
            kvecs = np.squeeze(self.kvecs, axis=1) #there is only one pair so dispense with the BS
            speckLocs = np.squeeze(self.speckLocs, axis=1) # nPoints x 2 speckles x [x, y]
            kMags = np.sqrt(self.kvecs[:, 0, 0]**2 + self.kvecs[:, 0, 1]**2)
            goodLocMask = ~np.isnan(speckLocs)[:,:,0] #n k-points, 2 specks each, [x, y]
            goodKMask = np.sum(goodLocMask, axis=1) #at least one good speckle

            goodKInds = np.where(goodKMask)[0]
            goodLocInds = np.where(goodLocMask)
            pairLocMask = np.nan*np.ones((speckLocs.shape[:2])) #which speckle in pair is it
            firstSpeck = speckLocs[goodLocInds[0][0], goodLocInds[1][0]]
            pairLocMask[goodLocInds[0][0], goodLocInds[1][0]] = 1
            
            for i, ind in enumerate(goodKInds):
                if np.sum(goodLocMask[ind]) == 2: #two good speckles at this k
                    closestSpeck = np.argmin([npl.norm(speckLocs[ind, 0] - firstSpeck), npl.norm(speckLocs[ind, 1] - firstSpeck)])
                    pairLocMask[ind, closestSpeck] = 1
                    pairLocMask[ind, closestSpeck-1] = -1
                elif np.sum(goodLocMask[ind]) == 1:
                    expectedDiff = np.abs(self.nPixPerLD*(kMags[goodKInds[0]] - kMags[ind]))/(2*np.pi) #expected pixel delta 
                    diff = npl.norm(speckLocs[ind][goodLocMask[ind]] - firstSpeck)
                    if diff > 2*expectedDiff:
                        pairLocMask[ind, np.where(goodLocMask[ind])[0]] = -1
                    else:
                        pairLocMask[ind, np.where(goodLocMask[ind])[0]] = 1
                else:
                    raise Exception

            nPairs1 = max(0, np.sum(pairLocMask==1) - 1)
            if np.sum(pairLocMask==1) >= 2:
                speckLocs1 = speckLocs[pairLocMask==1]
                kvecs1 = kvecs[np.where(pairLocMask==1)[0]]
                kMags1 = kMags[np.where(pairLocMask==1)[0]]
                speckDiffs1 = np.diff(speckLocs1, axis=0)
                kDiffs1 = np.diff(kMags1, axis=0)
                nPixPerKVect1 = (speckDiffs1.T/kDiffs1).T
            else:
                nPixPerKVect1 = np.array([0, 0])
                kMags1 = np.array([0])
                speckLocs1 = np.array([0, 0])

            nPairs2 = max(0, np.sum(pairLocMask==-1) - 1)
            if np.sum(pairLocMask==-1) >= 2:
                speckLocs2 = speckLocs[pairLocMask==-1]
                kvecs2 = kvecs[np.where(pairLocMask==-1)[0]]
                kMags2 = kMags[np.where(pairLocMask==-1)[0]]
                speckDiffs2 = np.diff(speckLocs2, axis=0)
                kDiffs2 = np.diff(kMags2, axis=0)
                nPixPerKVect2 = (speckDiffs2.T/kDiffs2).T
            else:
                nPixPerKVect2 = np.array([0, 0])
                kMags2 = np.array([0])
                speckLocs2 = np.array([0, 0])

            nPixPerKVect = (np.mean(nPixPerKVect1, axis=0)*nPairs1 - np.mean(nPixPerKVect2, axis=0)*nPairs2)/(nPairs1 + nPairs2)
            centerLocs1 = speckLocs1 - (np.tile(nPixPerKVect, (len(kMags1),1)).T*kMags1).T
            centerLocs2 = speckLocs2 + (np.tile(nPixPerKVect, (len(kMags2),1)).T*kMags2).T
            print centerLocs1
            print centerLocs2
            self.center = (np.mean(centerLocs1, axis=0)*nPairs1 + np.mean(centerLocs2, axis=0)*nPairs2)/(nPairs1 + nPairs2)
            self.nPixPerLD = npl.norm(nPixPerKVect)*(2*np.pi)
                
            #pixLocs = self.speckLocs/self.nPixPerLD
        else:
            raise Exception('NO! :/')

    def calculateLOverD(self):
        pairDiffs = np.diff(self.speckLocs, axis=2)
        pairDiffs = np.squeeze(pairDiffs, axis=2) #shape: (NKMags, m pairs, 2 coord diffs)
        goodPairMask = ~np.isnan(pairDiffs)
        pairDiffs = np.reshape(pairDiffs[goodPairMask], (-1, 2))
        
        #kvecs = np.zeros((self.kvecs.shape[0]*2, 2))
        #kvecs[::2,:] = self.kvecs
        #kvecs[1::2,:] = self.kvecs #duplicate kvecs b/c we have 2 pairs per kvec
        #kvecs = np.reshape(kvecs[goodPairMask], (-1, 2))

        kvecs = np.reshape(self.kvecs[goodPairMask], (-1, 2)) # (NKMags, m kvecs, 2 coords)

        self.nPixPerLD = np.pi*np.mean(np.sqrt(pairDiffs[:,0]**2 + pairDiffs[:,1]**2)/np.sqrt(kvecs[:,0]**2 + kvecs[:,1]**2))

    def calibrateIntensity(self):
        kMags = np.zeros(self.kvecs.shape[:2])
        kMags = np.sqrt(self.kvecs[:, :, 0]**2 + self.kvecs[:, :, 1]**2)
        intensities = np.nanmean(self.speckIntensities, axis=2)
        goodMask = ~np.isnan(intensities)
        a, b, c = np.polyfit(kMags[goodMask], self.amplitude**2/intensities[goodMask], 2)

        c_only = np.mean(self.amplitude**2/intensities[goodMask])

        plt.plot(kMags[goodMask], intensities[goodMask], '.')
        x = np.linspace(kMags[0], kMags[-1])
        plt.plot(x, self.amplitude**2/(a*x**2 + b*x + c))
        plt.show()

        self.intensityCal = np.array([a, b, c])
        self.intensityCalCOnly = c_only

    def writeToConfig(self, cfgFn):
        params = speckpy.PropertyTree()
        params.read_info(cfgFn)
        if not self.calType=='intensity':
            params.put('ImgParams.xCenter', self.center[1])
            params.put('ImgParams.yCenter', self.center[0])
            params.put('ImgParams.lambdaOverD', self.nPixPerLD)
        params.put('DMParams.a', self.intensityCal[0])
        params.put('DMParams.b', self.intensityCal[1])
        params.put('DMParams.c', self.intensityCal[2])
        params.write(cfgFn)
        
        
        

class CalspotGUI(object):
    
    def __init__(self, image, enforcePairs=True, nPairs=2):
        self._speckLocs = np.zeros((nPairs, 2, 2)) # nPairs x 2 specks/pair x 2 coords/speck
        self._speckLocs.fill(np.nan)
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(image)
        self.enforcePairs = enforcePairs
        self.nPairs = nPairs

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
        for i in range(self.nPairs):
            if not(np.all(np.isnan(self._speckLocs[i])) or np.all(~np.isnan(self._speckLocs[i]))) and self.enforcePairs:
                raise RuntimeError('Must click a conjugate pair for each identified speckle! Restarting...')
        return self._speckLocs

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None)
    args = parser.parse_args()

    config = ReadDict(file=args.config)
    if config['beammap']:
        beammap = Beammap(file=config['beammap'])
    else:
        beammap = None

    cal = Calibrator(config['dmChannel'], config['imageName'], beammap=beammap)
    create_log('mkidreadout')
    cal.run(config['startK'], config['stopK'], config['dmAmp'], config['nPoints'], config['integrationTime'], config['lOverDEst'],
            speckWin=5, angle=config['angle'], calType=config['type'])
    if not config['type']=='intensity':
        print 'center:', cal.center
        print 'l/D:', cal.nPixPerLD
        if config['outputConfig']:
            cal.writeToConfig(config['outputConfig'])
    print 'calCoeffs'
    print '    a:', cal.intensityCal[0]
    print '    b:', cal.intensityCal[1]
    print '    c:', cal.intensityCal[2]
    print 'c_only:', cal.intensityCalCOnly
