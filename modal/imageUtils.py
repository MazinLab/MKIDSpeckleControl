import os, sys, struct, time
import numpy as np
import scipy.ndimage as sciim
import matplotlib.pyplot as plt

def identify_bright_points(image, size = None):
    """WARNING: indexes, NOT coordinates"""
    max_filt = sciim.filters.maximum_filter(image, size= size)
    
    pts_of_interest = (max_filt == image)
    pts_of_interest_in_region = pts_of_interest
    iindex, jindex = np.nonzero((max_filt == image))
    intensities = np.array([])
    iindex_nz = np.array([])
    jindex_nz = np.array([])
    for i in range(len(iindex)):
        if image[iindex[i], jindex[i]]>0:
            intensities = np.append(intensities, image[iindex[i], jindex[i]])
            iindex_nz = np.append(iindex_nz, iindex[i])
            jindex_nz = np.append(jindex_nz, jindex[i])
    order = np.argsort(intensities)[::-1]
    sorted_i = iindex_nz[order]
    sorted_j = jindex_nz[order]
    #print 'sortedj len', len(sorted_j)
    return zip(sorted_i, sorted_j)
    
def filterpoints(pointslist, rad=6.0, max=5):
    plist = np.asarray(pointslist[:])
    finalList = []
    while(True):
        point = plist[0]
        finalList.append(tuple(point))
        diffs = point - plist
        dists = np.sqrt(diffs[:,0]**2 + diffs[:,1]**2)
        deleteInds = np.where(dists<=rad)
        plist = np.delete(plist, deleteInds, 0)
        if len(plist)==0:
            break;

    if len(finalList)>max:
        finalList = finalList[0:max]
    return finalList

def smartBadPixFilt(img, badPixMask, winSize=3):
    image = np.copy(img)
    image[np.where(badPixMask>0)] = 0
    filtImage = sciim.filters.uniform_filter(image, winSize)
    goodPixMask = np.array(np.logical_not(badPixMask), dtype=np.float)
    coeffVals = sciim.filters.uniform_filter(goodPixMask, winSize)
    coeffVals[coeffVals<1./winSize**2] = np.inf
    filtImage = filtImage/coeffVals
    image[np.where(badPixMask>0)] = filtImage[np.where(badPixMask>0)]
    return image

def gaussianFilt(image, badPixMask=None, sigma=1):
    filtImage = sciim.filters.gaussian_filter(image, sigma)
    if badPixMask is None:
        return filtImage
    goodPixMask = np.array(np.logical_not(badPixMask), dtype=np.float)
    coeffVals = sciim.filters.gaussian_filter(goodPixMask, sigma)
    coeffVals[coeffVals<1./(3*sigma)**2] = np.inf
    filtImage = filtImage/coeffVals
    return filtImage
