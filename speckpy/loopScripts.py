import numpy as np
import time
import matplotlib.pyplot as plt
import _speckpy as sp

def runLoop(cfg, nIters, logLevel='info'):
    if isinstance(cfg, sp.PropertyTree):
        pt = cfg
    elif isinstance(cfg, str):
        pt = sp.PropertyTree()
        pt.read_info(cfg)
    else:
        raise Exception('Unknown cfg type')
    startTs = int(time.time())
    sp.addLogfile(str(startTs) + '.log', False)
    dmNulling = sp.SpeckleToDM(pt.get("DMParams.channel"))
    dmNulling.updateDM()
    if logLevel=='info':
        sp.setInfoLog()
    elif logLevel=='debug':
        sp.setDebugLog()
    elif logLevel=='trace':
        sp.setTraceLog()
    else:
        raise Exception('log level not found')
    lc = sp.runLoop(nIters, pt, True)
    dt = float(pt.get("NullingParams.integrationTime"))/1000
    t = np.linspace(0, dt*nIters, nIters)
    plt.plot(t, lc)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Control Region Intensity (counts)')
    plt.savefig(str(startTs)+'.png')
    plt.close()
    np.savez(str(startTs)+'.npz', lc=lc, t=t)
    pt.write(str(startTs)+'.info')
    return t, lc
