import tensorflow as tf
import numpy as np
import scipy.linalg as scilin
import scipy.ndimage as sciim
import matplotlib.pyplot as plt
import pickle as pkl
import copy

class QModel(object):

    def __init__(self, beta, cij, lOverD, corrWin, center, ctrlRegionStart, ctrlRegionEnd, yFlip=False):
        """
        beta is in CPS, unlike gMat
        ctrlRegionStart: r, c control region start boundary wrt to center
        ctrlRegionEnd: r, c control region end boundary wrt to center
        corrWin: single sided, inclusive window size (e.g. corrWin = 5 will have 11 valid adjacent modes
        """

        coordImage = np.mgrid[0:(ctrlRegionEnd[0] - ctrlRegionStart[0]), 0:(ctrlRegionEnd[1] -  ctrlRegionStart[1])]
        coordImage = np.transpose(coordImage, axes=(1, 2, 0)) # should be indexed r, c, coordAxis
        coordList = np.reshape(coordImage, (-1, 2))

        modeImage = 2*np.pi/lOverD*np.mgrid[ctrlRegionStart[0]:ctrlRegionEnd[0], ctrlRegionStart[1]:ctrlRegionEnd[1]]
        modeImage = np.transpose(modeImage, axes=(1, 2, 0)) # should be indexed r, c, coordAxis
        modeList = np.reshape(modeImage, (-1, 2))

        self.coordImage = coordImage #ctrl region image containing coordinate of each point, starting at 0,0
        self.modeImage = modeImage #ctrl region image containing kvecs for that pixel
        self.coordList = coordList #bad pix masked list of ctrl region coords starting at 0,0
        self.modeCoordList = np.reshape(coordImage, (-1, 2)) #coordList but not bad pix masked (i.e. integer form of kVecs)
        self.modeList = modeList #list of (real) kVec modes. Full mode list is 2x size
        self.nPix = len(coordList)
        self.nRows = coordImage.shape[0]
        self.nCols = coordImage.shape[1]
        self.nHalfModes = len(self.modeList)
        self.pixIndImage = self._genPixIndImage()
        self.center = center
        self.ctrlRegionStart = ctrlRegionStart
        self.ctrlRegionEnd = ctrlRegionEnd
        self.pixIndImage = self._genPixIndImage()
        self.intTime = None
        self.corrWin = corrWin
        self.filtSize = 2*corrWin+1
        self.imgShape = tuple(ctrlRegionEnd - ctrlRegionStart)


        self.zData = []
        self.upInvData = []
        self.iData = []
        self.rData = [] 
        self.ucData = []

        if yFlip:
            self.yFlip = -1
        else:
            self.yFlip = 1

    def _genPixIndImage(self):
        pixIndImage = np.zeros(self.coordImage.shape, dtype=int)
        for i in range(self.nPix):
            pixIndImage[self.coordList[i, 0], self.coordList[i, 1]] = i
        return pixIndImage
    
    def _initNetworks(self):
        self.zIn = tf.keras.Input(shape=(self.nRows, self.nCols, 2), name='zIn')
        self.upInvIn = tf.keras.Input(shape=(self.nRows, self.nCols, 1), name='upInvIn') #1/up
        self.iIn = tf.keras.Input(shape=(self.nRows, self.nCols), name='intensity') #initial intensity image
        self.rIn = tf.keras.Input(shape=(1,), name='reward') #diff between initial and final speck intensities
        self.ucIn = tf.keras.Input(shape=(self.nRows, self.nCols, 2), name='ucIn')

        self.zFilt = tf.keras.layers.LocallyConnected2D(2, (self.filtSize, self.filtSize), 
                implementation=2, padding='same')(self.zIn)
        self.upInvFilt = tf.keras.layers.LocallyConnected2D(2, (self.filtSize, self.filtSize), 
                implementation=2, padding='same')(self.upInvIn)
        self.policyOut = tf.keras.layers.Multiply(name='policyOut')([self.zFilt, self.upInvFilt]) #output is uc

        self.ucPolDiff = tf.keras.layers.add([self.policyOut, -1*self.ucIn])

        #square PolDiff and multiply w/ locally connected
        self.ucPDSqSc = tf.keras.layers.LocallyConnected2D(1, (self.filtSize, self.filtSize),
                implementation=2, padding='same')(tf.math.square(self.ucPolDiff)) 
        self.qInt = tf.keras.layers.add([self.iIn, -1*tf.squeeze(self.ucPDSqSc, axis=3)])
        self.qOut = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x), name='qOut')(self.qInt)

        #self.policyModel = tf.keras.Model(inputs=[self.zIn, self.upInvIn], outputs=self.policyOut)
        self.qModel = tf.keras.Model(inputs=[self.zIn, self.upInvIn, self.iIn, self.rIn, self.ucIn], 
                outputs=[self.policyOut, self.qOut])

        #self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        #self.lossModel = tf.keras.losses.Huber()
        self.qModel.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss={'qOut':tf.keras.losses.Huber()})

    def addIter(self, zImg, upImg, ucImg, i0Img, i1Img):
        """
        zImg - 2 color image of probe measurements
        upImg - 1 color image of probes (re and im are assumed to be the same)
        ucImg - 2 color image of control
        i0Img - speckle det image
        i1Img - speckle det image after uc applied
        """
        probeCoords = np.asarray(np.where(upImg))
        for coord in probeCoords:
            probeMask = np.zeros(upImg.shape)
            probeMask[coord] = 1
            probeMask = sciim.filters.uniform_filter(probeMask, size=self.filtSize)
            probeMask[probeMask > 0] = 1 #ones in region surrounding probe

            up = upImg[coord]
            #upInvIn = 1/up*probeMask
            upInvIn = np.zeros(upImg.shape)
            upInvIn[coord] = 1/up #keep up at its original coords
            zIn = zImg*probeMask
            ucIn = ucImg*probeMask
            i0In = i0Img*probeMask
            i1In = i1Img*probeMask
            rIn = np.sum(i0In - i1In)

            self.zData.append(zIn)
            self.upInvData.append(upInvIn)
            self.iData.append(i0In)
            self.rData.append(rIn)
            self.ucData.append(ucIn)

    def getUc(self, z, up, i0, eps=0):
        """
        explore w/ probability epsilon
        """
        pass

    def runTrainStep(self, nEpochs, learningRate=0.001):
        pass

    def getDMSpeckles(self, modeImage):
        """
        modeImage - row x col x 2 image
            first color channel is re probes, second is im probes
        self.modeImage - row x col x coord_axis (0 is row coord, 1 is col coord)
        """
        if modeImage.shape != self.modeImage.shape 
            raise Exception('mode image should be {} image'.format(self.modeImage.shape + (2,)))

        #modeVec = np.reshape(modeVec, (2, -1)).T
        #modeInds = np.where((modeVec[:len(self.modeList)] != 0)|(modeVec[len(self.modeList):] != 0)[0]
        modeCoords = np.asarray(np.where(np.sum(np.abs(modeImage), axis=2) != 0))

        ampList = []
        phaseList = []
        kVecList = []

        for coord in modeInds:
            cxAmp = modeImage[coord[0], coord[1]]
            ampList.append(np.sqrt(cxAmp[0]**2 + cxAmp[1]**2))
            phaseList.append(np.arctan2(cxAmp[1], cxAmp[0]))
            #switch kVecs from r, c to x, y indexing
            kVecList.append([self.modeImage[coords[0], coords[1], 1], self.yFlip*self.modeList[coords[1], coords[0], 0]])

        return np.array(ampList), np.array(phaseList), np.array(kVecList)



        



