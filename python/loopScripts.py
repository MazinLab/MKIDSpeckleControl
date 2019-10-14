import numpy as np
import time
import matplotlib.pyplot as plt
import speckpy

def runLoop(cfgFile, nIters, logLevel='info'):
    pt = speckpy.PropertyTree()
    pt.read_info(cfgFile)
    startTs = int(time.time())
    speckpy.addLogfile(str(startTs) + '.log')
    dmNulling = speckpy.SpeckleToDM(pt.get("DMParams.channel"))
    dmNulling.updateDM()
    if logLevel=='info':
        speckpy.setInfoLog()
    elif logLevel=='debug':
        speckpy.setDebugLog()
    elif logLevel=='trace':
        speckpy.setTraceLog()
    else:
        raise Exception('log level not found')
    lc = speckpy.runLoop(nIters, pt, True)
    dt = float(pt.get("NullingParams.integrationTime"))/1000
    t = np.linspace(0, dt*nIters, nIters)
    plt.plot(t, lc)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Control Region Intensity (counts)')
    plt.savefig(str(startTs)+'.png')
    np.savez(str(startTs)+'.npz', lc=lc, t=t)
    pt.write(str(startTs)+'.info')
