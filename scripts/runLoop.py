#!/usr/bin/env python

import argparse
import speckpy

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfgfile', help='speckle nulling config file (.info)')
    parser.add_argument('-l', '--log', default='info', help='logging level (debug, info, trace)')
    parser.add_argument('-n', '--n-iters', default=500, help='Number of loop iterations (total exposures)')
    args = parser.parse_args()

    pt = speckpy.PropertyTree()
    pt.read_info(args.cfgfile)
    dmNull = speckpy.SpeckleToDM(pt.get('DMParams.channel'))
    dmNull.updateDM()
    speckpy.runLoop(pt, args.n_iters, args.log)
