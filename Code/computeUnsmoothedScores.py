#Using previously computed persistence landscapes, this
#program calculates the anomaly score - before smoothing
#It accepts as a command line argument a number for the day
#so that is only computes one day at a time.
import numpy as np
from datetime import datetime
import pickle
import os
import multiprocessing
import ctypes
from scipy import stats
import gc
import sys
import re as rg

#########
#Parameters

#bagSize: number of weeks to include in a bag
#iRange: amount of time-shifting
#degCompute: number of degrees computed in computeLandscapes
#gradeCompute: number of grades computed in computeLandscapes
#threadSize: CPUs available to use
#resolutionCompute: resolution for landscape used in computeLandscapes
#interpResolution: common resolution used for interpolating landscapes used here
#sRange: number of sensors
#degRange: number of degrees to compute
#gradeRange: number of grades to compute

numberOfBags = 15
degCompute = 5
gradeCompute = 5
bagSize = 5
iRange = 12
threadSize = 45
resolutionCompute = 2000
interpResolution = 1700 
degRange = 3 
outputDirectory = "Output/"
sRange = 11
gradeRange = 5


################################################
# Common Set-up for functions

dt = str(datetime.now().strftime("%Y-%m-%d-%H-%M"))
day = int(sys.argv[1])

if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)

neededIndices = []
for s in range(sRange):
    for d in [day]:
        for t in range(288):
            neededIndices.append((s, d, t))

#medDist stores the results after taking the 80th percentile among bags            
medDist_shared = multiprocessing.Array(
    ctypes.c_double, 52 * gradeRange * len(neededIndices) * interpResolution * degRange
)
medDist_lump = np.frombuffer(medDist_shared.get_obj())
medDist = medDist_lump.reshape((52, gradeRange, len(neededIndices), interpResolution, degRange))


#stores how many weeks have data, so we can go from rank -> percentile
scalingWeeks_shared = multiprocessing.Array(
    ctypes.c_double, 52 *gradeRange*len(neededIndices)*numberOfBags*degRange
)
scalingWeeks_lump = np.frombuffer(scalingWeeks_shared.get_obj())
scalingWeeks = scalingWeeks_lump.reshape((52,gradeRange, len(neededIndices),numberOfBags,degRange))

#placeholders for later
#medDistPrePercs is medDist converted to percentiles
#medDistPercs is after taking the 90th percentile among the domain, and the key output of this script
medDistPrePercs = np.empty((1, 1, 1, 1, 1))
medDistPercs = np.empty((1, 1, 1, 1))


testingPoints = [r * 19000 / interpResolution for r in range(interpResolution)]

##############################################3
# Functions 

def dtGetter(d, t):
    if t < 0:
        return (d - 1) % 7, t % 288
    elif t >= 288:
        return (d + 1) % 7, t % 288
    else:
        return d, t


def computeHeightsOnACommonDomain(s, d, t, w):
    heightsW = np.full((gradeRange, numberOfBags, degRange, interpResolution),np.nan)
    try:
        sRW = np.load(f"{outputDirectory}/{s}-{d}-{t}/sample{s}-{d}-{t}-{w}.npy")
        distW = np.load(f"{outputDirectory}/{s}-{d}-{t}/dist{s}-{d}-{t}-{w}.npy")
    except:
        print(f"Failed to load {s}-{d}-{t}-{w}")
        return heightsW
    distRs = distW.reshape((numberOfBags, degCompute, gradeCompute, resolutionCompute))
    lVals = np.fromfunction(lambda kl, il, jl: kl, (resolutionCompute, numberOfBags, degCompute))
    # sRW[:,:,0] is size numberOfBags,degCompute. We can broadcast to earlier dimns of resolution, copying it over
    AllTheXs = np.broadcast_to(
        sRW[:, :, 0], (resolutionCompute, numberOfBags, degCompute)
    ) + lVals * np.broadcast_to(
        (sRW[:, :, 1] - sRW[:, :, 0]) / resolutionCompute, (resolutionCompute, numberOfBags, degCompute)
    )
    for k in range(gradeRange):
        for j in range(numberOfBags):
            for deg in range(degRange):
                heightsW[k, j, deg] = np.interp(
                    testingPoints, AllTheXs[:, j, deg], distRs[j, deg, k]
                )
    del sRW
    del distW
    gc.collect()
    return heightsW


def rankHeightsAndThenBags(i, heights):
    # first take ranks across the weeks axis
    # heights[w,p,j,k,r]
    percs = stats.rankdata(heights,axis=0,nan_policy="omit")/scalingWeeks[:,:,i,:,:,np.newaxis]
    # take the 80th percentile among the bags axis
    # perScore should be (52,gradeRange,interpResolution,degRange)
    perScores = np.nanpercentile(percs, 80, axis=2)
    perScores = np.swapaxes(perScores,2,3)
    with medDist_shared.get_lock():
        medDist[:, :, i, :, :] = perScores


def computeHeightsAndBagPercentile(i):
    s, d, t = neededIndices[i]
    heights = np.array([computeHeightsOnACommonDomain(s, d, t, w) for w in range(52)])
    #heights: (52, gradeRange, numberOfBags, degRange, interpResolution)
    
    #scalings is used to go from rank -> percentile by counting how many weeks have data
    scalings = np.count_nonzero(~np.isnan(heights[:,:,:,:,0]),axis=0)
    with scalingWeeks_shared.get_lock():
        #scalingWeeks:(52,gradeRange,len(neededIndices),numberofBags,degRange)
        scalingWeeks[:, :, i, :, :] = scalings
    try:
        rankHeightsAndThenBags(i, heights)
    except Exception as e:
        print(f"Error:{e}")
    del heights
    gc.collect()
    return

def highScoreAmongstDomain(i):
    return np.nanpercentile(medDistPrePercs[:, :, i, :, :], 90, axis=2)

########
#Run the script

if __name__ == "__main__":
    if not os.path.exists(outputDirectory + "/" + dt):
        os.makedirs(outputDirectory + "/" + dt)

    with multiprocessing.Pool(threadSize) as q:
        q.map(
            computeHeightsAndBagPercentile,
            [i for i in range(len(neededIndices))],
        )
        #the results are stored in medDist
    with open(
        f"{outputDirectory}/{dt}/medDistDay{day}.npy","wb"
    ) as outp:
        np.save(outp,medDist)
    #medDist (52, gradeRange, len(neededIndices), interpResolution, degRange)
    #scalingWeeks:(52,gradeRange,len(neededIndices),numberofBags,degRange)

    #To get a percentile we must know how many weeks were used (didn't have missing data)
    #This is the purpose of scales
    scales = np.nanmax(scalingWeeks,axis=3)
    medDistPrePercs = stats.rankdata(medDist, axis=0, nan_policy="omit") / scales[:,:,:,np.newaxis,:]

    with multiprocessing.Pool(threadSize) as q:
        medDistPercs = np.array(
            q.map(
                highScoreAmongstDomain,
                [i for i in range(len(neededIndices))],
            )
        )
    # neededIndices is now the 0th axis, but we want it to be 2nd axis
    medDistPercs = np.moveaxis(medDistPercs, 0, 2)
    with open(f"{outputDirectory}/{dt}/medDistPercsDay{day}.npy","wb") as outp:
        np.save(outp,medDistPercs)