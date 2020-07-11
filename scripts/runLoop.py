#!/home/scexao/anaconda3/envs/mkids/bin/python

import argparse
import speckpy
import matplotlib.pyplot as plt

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfgfile', help='speckle nulling config file (.info)')
    parser.add_argument('-l', '--log', default='info', help='logging level (debug, info, trace)')
    parser.add_argument('-n', '--n-iters', default=500, help='Number of loop iterations (total exposures)')
    parser.add_argument('-p', '--plot', action='store_true')
    args = parser.parse_args()

    pt = speckpy.PropertyTree()
    pt.read_info(args.cfgfile)
    dmNull = speckpy.SpeckleToDM(pt.get('DMParams.channel'))
    dmNull.updateDM()
    if int(args.n_iters) > 0:
        t, lc = speckpy.runLoop(pt, int(args.n_iters), args.log)
    if args.plot:
        plt.plot(t, lc)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Control Region Intensity (counts)')
        plt.show()
