#!/home/scexao/anaconda3/envs/mkids/bin/python

import argparse
import signal
import numpy as np

import mkidreadout.readout.packetmaster as pm
import mkidcore.objects as mko


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-roaches', default=14)
    parser.add_argument('--port', default=50000)
    parser.add_argument('--max-priority', action='store_true', help='runs as realtime process')
    parser.add_argument('--beammap', default=None)
    parser.add_argument('--wvl-coeffs', default=None)
    parser.add_argument('--shm-images', nargs= '+', default=None)
    args = parser.parse_args()
    nRows = 146
    nCols = 140

    if args.shm_images:
        shmImageCfg = {}
        for imgName in args.shm_images:
            shmImageCfg[imgName] = {}
    else:
        shmImageCfg = None
    
    if args.beammap:
        bm = mko.Beammap(args.beammap, (nCols, nRows))
    else:
        bm = None

    if args.wvl_coeffs:
        wvl = np.load(args.wvl_coeffs)
    else:
        wvl = None


    pkm = pm.Packetmaster(int(args.num_roaches), args.port, nRows, nCols, useWriter=False, 
            wvlCoeffs=wvl, beammap=bm, sharedImageCfg=shmImageCfg,
            maximizePriority=args.max_priority, recreate_images=True)


    def quitHandler(signum, frame):
        print 'exiting'
        pkm.quit()

    signal.signal(signal.SIGINT, quitHandler)

    signal.pause()


