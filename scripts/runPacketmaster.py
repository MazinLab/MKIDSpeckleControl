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
    

    pkm = pm.Packetmaster(args.num_roaches, args.port, nRows, nCols, useWriter=False, 
            wvlCoeffs=args.wvl_coeffs, beammap=args.beammap, sharedImageCfg=shmImageCfg,
            maximizePriority=args.max_priority, recreate_images=True)


    def quitHandler(signum, frame):
        print 'exiting'
        pkm.quit()

    signal.signal(signal.SIGINT, quitHandler)

    signal.pause()


