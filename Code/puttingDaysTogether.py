#Since computeUnsmoothedScores computes over multiple days, this script puts
#the days together and smooths over adjacent time periods.
import numpy as np
from datetime import datetime
import pickle
import os
import multiprocessing
import ctypes

#####
# Parameters

#threadSize: CPUs available to use
#degRange: number of degrees computed
#sRange: number of sensors
#gradeRange: number of grades computed

threadSize = 156 
degRange = 3 
gradeRange = 5
outputDirectory = "Output/"

sRange = 11
##############################################33
# Common Set-up for functions

dt = str(datetime.now().strftime("%Y-%m-%d-%H-%M"))


if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)

neededIndices = []
startingIndices = []
for s in range(sRange):
    for d in range(7):
        for t in range(288):  # 288
            neededIndices.append((s, d, t))



medDistPercs = np.full((52, gradeRange, len(neededIndices), degRange), np.nan)

percsTogether_shared = multiprocessing.Array(
    ctypes.c_double, 52 * gradeRange * len(neededIndices) * degRange
)
percsTogether_lump = np.frombuffer(percsTogether_shared.get_obj())
percsTogether = percsTogether_lump.reshape((degRange, len(neededIndices), 52, gradeRange))
percsTogether.fill(np.nan)
################
################
# Functions
def dtGetter(d, t):
    if t < 0:
        return (d - 1) % 7, t % 288
    elif t >= 288:
        return (d + 1) % 7, t % 288
    else:
        return d, t


def indicesToCareAboutGetter(sdtTriplet):
    s, d, t = sdtTriplet
    indicesToCareAbout = []
    for i in range(0, 13, 1):
        di, ti = dtGetter(d, t - (i + 1))
        indicesToCareAbout.append(neededIndices.index((s, di, ti)))
    for i in range(20):
        di, ti = dtGetter(d, t + i)
        indicesToCareAbout.append(neededIndices.index((s, di, ti)))
    return indicesToCareAbout


def computeSmoothedScore(weekI, k, deg, sIdx):
    indicesToCareAbout = indicesToCareAboutGetter(neededIndices[sIdx])
    percy = [medDistPercs[weekI, k, i, deg] for i in indicesToCareAbout]
    thisPerc = np.nanpercentile(percy, 90)
    with percsTogether_shared.get_lock():
        percsTogether[deg, sIdx, weekI, k] = thisPerc
    return

########################
# Run Script
if __name__ == "__main__":
    if not os.path.exists(f"{outputDirectory}/{dt}"):
        os.makedirs(f"{outputDirectory}/{dt}")

    mdPs = []
    with open(
        f"{outputDirectory}/medDistPercsDay{0}.npy", "rb"
    ) as outp:
        md = np.load(outp)
        mdPs.append(md)
    with open(
        f"{outputDirectory}/medDistPercsDay{1}.npy", "rb"
    ) as outp:
        md = np.load(outp)
        mdPs.append(md)
    with open(
        f"{outputDirectory}/medDistPercsDay{2}.npy", "rb"
    ) as outp:
        md = np.load(outp)
        mdPs.append(md)
    with open(
        f"{outputDirectory}/medDistPercsDay{3}.npy", "rb"
    ) as outp:
        md = np.load(outp)
        mdPs.append(md)
    with open(
        f"{outputDirectory}/medDistPercsDay{4}.npy", "rb"
    ) as outp:
        md = np.load(outp)
        mdPs.append(md)
    with open(
        f"{outputDirectory}/medDistPercsDay{5}.npy", "rb"
    ) as outp:
        md = np.load(outp)
        mdPs.append(md)
    with open(
        f"{outputDirectory}/medDistPercsDay{6}.npy", "rb"
    ) as outp:
        md = np.load(outp)
        mdPs.append(md)
    # We must re-construct medDistPercs
    dayCounters = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(neededIndices)):
        s, d, t = neededIndices[i]
        medDistPercs[:, :, i, :] = mdPs[d][:, :, dayCounters[d], :]
        dayCounters[d] = dayCounters[d] + 1
    with multiprocessing.Pool(threadSize) as q:
        q.starmap(
            computeSmoothedScore,
            [
                (w, k, deg, sIdx)
                for w in range(52)
                for k in range(gradeRange)
                for sIdx in range(len(neededIndices))
                for deg in range(degRange)
            ],
        )
    with open(
        f"{outputDirectory}/{dt}/percentiles.npy"
        "wb",
    ) as outp:
        np.save(outp,percsTogether)
