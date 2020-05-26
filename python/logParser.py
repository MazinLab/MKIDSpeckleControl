import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#START_STATEMENT = 'updating...'
#END_STATEMENT = 'Clearing probe speckles'

#START_STATEMENT = 'Clearing probe speckles'
#END_STATEMENT = 'Updating DM with new'

#START_STATEMENT = 'waiting...'
#END_STATEMENT = 'grabbing ctrl region...'

#END_STATEMENT = 'waiting...'
#START_STATEMENT = 'grabbing ctrl region...'

START_STATEMENT = 'ImageGrabber: starting integration'

class IterStep(object):
    def __init__(self, name, start, end):
        self.name = name
        self.start = start
        self.end = end
        self.tList = []
        self.startTime = None
        self.endTime = None

    def setStartTime(self, time):
        if self.startTime is not None:
            raise Exception('start time already set!')
        self.startTime = time

    def setEndTime(self, time):
        if self.endTime is not None:
            raise Exception('end time already set!')
        self.endTime = time

    def consolidateIter(self):
        if self.startTime is None:
            self.tList.append(0)
        else:
            self.tList.append((self.endTime - self.startTime).total_seconds())
        self.startTime = None
        self.endTime = None

def getTimestamp(line):
    ts = line.split(' ')[1][:-1]
    t0 = datetime.strptime(ts, '%H:%M:%S.%f')
    return t0


def parseIter(lines, iterSteps):
    for line in lines:
        for iterStep in iterSteps:
            if iterStep.start in line:
                iterStep.setStartTime(getTimestamp(line))
            elif iterStep.end in line:
                iterStep.setEndTime(getTimestamp(line))

    for iterStep in iterSteps:
        iterStep.consolidateIter()




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()

    tList = []

    with open(args.file, 'r') as f:
        line = f.readline()
        ts = line.split(' ')[1][:-1]
        tstart = datetime.strptime(ts, '%H:%M:%S.%f')
        iterLines = []
        iterSteps = []
        iterSteps.append(IterStep('full', 'starting integration', 'Updating DM'))
        iterSteps.append(IterStep('get image', 'waiting...', 'grabbing ctrl region...'))
        iterSteps.append(IterStep('us filtering', 'gaussian filtering...', 'done filtering'))
        iterSteps.append(IterStep('us filtering: remove bad pix', 'bpfilt: remove bad pix', 'bpfilt: done remove bad pix'))
        iterSteps.append(IterStep('us filtering: upsample', 'bpfilt: upsample', 'bpfilt: done upsample'))
        iterSteps.append(IterStep('us filtering: initial gaussian blur', 'bpfilt: gaussian blur', 'bpfilt: done initial gaussian blur'))
        iterSteps.append(IterStep('us filtering: post gaussian blur', 'bpfilt: done initial gaussian blur', 'bpfilt: done gaussian blur'))
        iterSteps.append(IterStep('loc max', 'finding local maxima...', 'SpeckleNuller: found'))
        iterSteps.append(IterStep('struct dumping', 'SpeckleNuller: found', 'bright spots'))
        iterSteps.append(IterStep('full detection', 'SpeckleNuller: detecting new speckles', 'bright spots'))
        #iterSteps.append(IterStep('img to 64f', 'SpeckleNuller: convert image to float', 'done convert image to float'))
        iterSteps.append(IterStep('dm map', 'Clearing probe speckles', 'Updating DM with new'))
        iterSteps.append(IterStep('update speckles', 'SpeckleNuller: updating...', 'Clearing probe speckles'))
        iterSteps.append(IterStep('update speckle objects', 'SpeckleNuller: updating speckle objects', 'done updating speckle objects'))
        #iterSteps.append(IterStep('exclusion zone cut', 'start exclusionZoneCut', 'done exclusionZoneCut'))
        #iterSteps.append(IterStep('create speckle objects', 'creating speckle objects...', 'done creating speckle objects'))
        while line:
            if START_STATEMENT in line:
                parseIter(iterLines, iterSteps)
                iterLines = []
            iterLines.append(line)
            prevLine = line
            line = f.readline()
            if not line:
                tend = getTimestamp(prevLine)

    print 'total time: {}'.format((tend - tstart).total_seconds())
    for step in iterSteps:
        print '{}: {} per iter'.format(step.name, np.mean(step.tList))
        plt.plot(step.tList, label=step.name)
    #plt.plot(tList)
    plt.plot(np.asarray(iterSteps[0].tList) - np.asarray(iterSteps[1].tList), label='full - get image')
    plt.legend()
    plt.show()


