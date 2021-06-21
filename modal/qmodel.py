import tensorflow as tf
import numpy as np
import scipy.linalg as scilin
import matplotlib.pyplot as plt
import pickle as pkl
import copy

class QModel(object):

    def __init__(self, beta, cij, lOverD, corrWin, center, ctrlRegionStart, ctrlRegionEnd, yFlip=False):
        """
        beta is in CPS, unlike gMat
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

    def _genPixIndImage(self):
        pixIndImage = np.zeros(self.coordImage.shape, dtype=int)
        for i in range(self.nPix):
            pixIndImage[self.coordList[i, 0], self.coordList[i, 1]] = i
        return pixIndImage
    
    def _initNetworks(self):
        self.zIn = tf.keras.Input(shape=(self.nRows, self.nCols, 2), name='zIn')
        self.upInvIn = tf.keras.Input(shape=(self.nRows, self.nCols, 2), name='upInvIn') #1/up
        self.iIn = tf.keras.Input(shape=(self.nRows, self.nCols), name='intensity') #initial intensity image
        self.rIn = tf.keras.Input(shape=(1,), name='reward') #diff between initial and final speck intensities
        self.ucIn = tf.keras.Input(shape=(self.nRows, self.nCols, 2), name='ucIn')

        self.zFilt = tf.keras.layers.LocallyConnected2D(2, (self.filtSize, self.filtSize), 
                implementation=2, padding='same')(zIn)
        self.pInvFilt = tf.keras.layers.LocallyConnected2D(2, (self.filtSize, self.filtSize), 
                implementation=2, padding='same')(upFilt)
        self.policyOut = tf.keras.Layers.Multiply()(zFilt, upInvFilt) #output is uc

        self.ucPolDiff = tf.keras.layers.add([self.policyOut, -1*self.ucIn])

        #square PolDiff and multiply w/ locally connected
        self.ucPDSqSc = tf.keras.layers.LocallyConnected2D(1, (self.filtSize, self.filtSize),
                implementation=2, padding='same')(tf.math.square(self.ucPolicyDiff)) 
        self.qOut = tf.math.reduce_sum(tf.keras.layers.add([self.iIn, -1*ucPDSqSc]))

        self.policyModel = tf.keras.Model(inputs=[self.zIn, self.upInvIn], outputs=self.policyOut)
        self.qModel = tf.keras.Model(inputs=[self.zIn, self.upInvIn, self.iIn, self.rIn, self.ucIn], 
                outputs=self.qOut)



