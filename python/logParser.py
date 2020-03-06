import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#START_STATEMENT = 'updating...'
#END_STATEMENT = 'Clearing probe speckles'

START_STATEMENT = 'Clearing probe speckles'
END_STATEMENT = 'Updating DM with new'

#START_STATEMENT = 'waiting...'
#END_STATEMENT = 'grabbing ctrl region...'

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()

    tList = []

    with open(args.file, 'r') as f:
        line = f.readline()
        ts = line.split(' ')[1][:-1]
        tstart = datetime.strptime(ts, '%H:%M:%S.%f')
        while line:
            if START_STATEMENT in line:
                ts = line.split(' ')[1][:-1]
                t0 = datetime.strptime(ts, '%H:%M:%S.%f')
            elif END_STATEMENT in line:
                ts = line.split(' ')[1][:-1]
                t1 = datetime.strptime(ts, '%H:%M:%S.%f')
                tList.append((t1 - t0).total_seconds())
            line = f.readline()
            if not line:
                tend = t1

    print 'total time: {}/{}'.format(np.sum(tList), (tend - tstart).total_seconds())
    plt.plot(tList)
    plt.show()


