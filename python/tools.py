import numpy as np
import os, struct
from mkidcore.objects import Beammap
from mkidpipeline.calibration.wavecal import Solution

def saveBadPixBin(beammap, wvlSol=None, hotPixMask=None):
    if hotPixMask is not None:
        raise NotImplementedError('Hot pix masking not yet immplemented')
    resIDMap = beammap.residmap
    badPixMask = beammap.failmask

    if wvlSol:
        for r in resIDMap.shape[0]:
            for c in resIDMap.shape[1]:
                resID = resIDMap[r,c]
                badPixMask[r, c] |= ~wvlSol.has_good_calibration_solution(resid=resID)

    dirname = os.path.dirname(beammap.file)
    fn = os.path.basename(beammap.file).split('.')[0]
    if wvlSol:
        fn += '_' + os.path.basename(str(wvlSol)).split('.')[0]
    fn += '_badPixMask.bin'

    fn = os.path.join(dirname, fn)
    saveBinImg(badPixMask, fn)

    return fn


def saveBinImg(image, imageFn):
    imageArr = image.flatten()
    imageStr = struct.pack('{}{}'.format(len(imageArr), 'H'), *imageArr)
    imgFile = open(imageFn, 'wb')
    imgFile.write(imageStr)
    imgFile.close()
