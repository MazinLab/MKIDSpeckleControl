import numpy as np
import os, struct
from mkidcore.objects import Beammap

def saveBadPixBin(beammap, wvlCoeffs=None, hotPixMask=None):
    resIDMap = beammap.residmap.T
    badPixMask = beammap.failmask
    assert resIDMap.shape == badPixMask.shape
    if hotPixMask is not None:
        badPixMask |= hotPixMask.astype(bool)

    if wvlCoeffs:
        for r in range(resIDMap.shape[0]):
            for c in range(resIDMap.shape[1]):
                resID = resIDMap[r,c]
                if ~np.any(resID == wvlCoeffs['res_ids']):
                    badPixMask[r, c] = True

    dirname = os.path.dirname(beammap.file)
    fn = os.path.basename(beammap.file).split('.')[0]
    if wvlCoeffs:
        fn += '_' + os.path.splitext(os.path.basename(str(wvlCoeffs['solution_file_path'])))[0]
    fn += '_badPixMask.bin'

    fn = os.path.join(dirname, fn)
    saveBinImg(badPixMask, fn)

    return fn, badPixMask


def saveBinImg(image, imageFn):
    imageArr = image.flatten()
    imageStr = struct.pack('{}{}'.format(len(imageArr), 'H'), *imageArr)
    imgFile = open(imageFn, 'wb')
    imgFile.write(imageStr)
    imgFile.close()
