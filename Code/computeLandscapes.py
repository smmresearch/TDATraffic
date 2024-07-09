#This file computes the persistence landscapes
#It accepts a day number as a command line argument
#so that it only computes one day at a time.
from gudhi import AlphaComplex, SimplexTree
import gudhi as gd
import gudhi.representations
import numpy as np
from datetime import datetime
import pickle
import copy
import os
import multiprocessing
import sys

####
#Parameters 

#numberOfBags
#bagSize: number of vectors in a bag
#iRange: amount of time shifting
#threadSize: number of CPUs available
#resolution: resolution to compute the landscapes at
#sRange: number of sensors
#degRange: number of degrees to compute
#gradeRange: number of grades to compute
numberOfBags = 15
bagSize = 5
iRange = 12
threadSize = 170
resolution = 2000
sRange = 11
degRange = 5
gradeRange = 5
outputDirectory = f"Output/AIOtest"
directoryOfReferenceFiles = "Reference"



####
# Common set-up between functions

with open(f"{directoryOfReferenceFiles}/parsedData5test.pkl", "rb") as outp:
    parsedData = pickle.load(outp)
with open(f"{directoryOfReferenceFiles}/weekToIndex5test.pkl", "rb") as outp:
    weekToIndex = pickle.load(outp)
with open(f"{directoryOfReferenceFiles}/weeksToPick5test-i15.pkl", "rb") as outp:
    weeksToPickSet = pickle.load(outp)

rng = np.random.default_rng()

day = int(sys.argv[1])

if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)

neededIndices = []
for s in range(sRange):
    for d in [day]:
        for t in range(288):
            neededIndices.append((s, d, t))
#################
# Functions

def dtGetter(d, t):
    if t < 0:
        return (d - 1) % 7, t % 288
    elif t >= 288:
        return (d + 1) % 7, t % 288
    else:
        return d, t


def computeLandscape(s, d, t):
    bags = []
    bagToWeeks = []
    weeksToPick = weeksToPickSet[s][d][t]

    #first create base bags
    for j in range(numberOfBags):
        currentBag = []
        weeks = rng.choice(list(weeksToPick), size=bagSize, replace=False).tolist()
        bagToWeeks.append(weeks)
        for i in range(iRange * 2 + 1):
            di, ti = dtGetter(d, t - iRange + i)
            indices = []
            for wi in weeks:
                indices.append(weekToIndex[s][di][ti][wi])
            Dsdt = np.array(parsedData[s][di][ti])
            currentBag += Dsdt[indices, :].tolist()
        bags.append(currentBag)

    trunc = gd.representations.DiagramSelector(use=True, point_type="finite")
    Landscape = gd.representations.Landscape(resolution=resolution)

    if not os.path.exists(f"{outputDirectory}/{s}-{d}-{t}"):
        os.makedirs(f"{outputDirectory}/{s}-{d}-{t}")

    for w in range(52):
        if w not in weeksToPick:
            #missing data, can't compute landscape
            continue
        testedDistances = np.empty((numberOfBags, degRange, resolution * gradeRange))
        testedSampleRanges = np.empty((numberOfBags, degRange, 2))
        #sample ranges are (b,d), hence the 2
        for j in range(numberOfBags):
            #first we take a bag and take a week at random and replace it with w
            #If w is already in the bag, we do not do anything
            modBag = copy.deepcopy(bags[j])
            weeks = bagToWeeks[j]
            weeksNpArr = np.array(weeks)
            if w in weeks:
                repl = w
                replIdx = np.nonzero(weeksNpArr == w)[0][0]
            else:
                repl = w
                replIdx = rng.choice(bagSize, size=1)[0]
            for i in range(iRange * 2 + 1):
                di, ti = dtGetter(d, t - iRange + i)
                Dsdti = np.array(parsedData[s][di][ti])
                modBag[i * bagSize + replIdx] = Dsdti[
                    weekToIndex[s][di][ti][repl]
                ].tolist()
            simplex = AlphaComplex(points=modBag)
            simplexTree = simplex.create_simplex_tree()
            simplexTree.compute_persistence(min_persistence=1)
            for deg in range(degRange):
                truncPersist = np.array(
                    [trunc(simplexTree.persistence_intervals_in_dimension(deg))]
                )
                testedDistances[j, deg] = Landscape.fit_transform(truncPersist)
                testedSampleRanges[j, deg] = Landscape.sample_range_fixed_
                #we need the sample ranges to put the different landscapes on 
                #a common domain in AAIO
        with open(
            f"{outputDirectory}/{s}-{d}-{t}/dist{s}-{d}-{t}-{w}.npy",
            "wb",
        ) as outp:
            np.save(outp, testedDistances)
        with open(
            f"{outputDirectory}/{s}-{d}-{t}/sample{s}-{d}-{t}-{w}.npy",
            "wb",
        ) as outp:
            np.save(outp, testedSampleRanges)
    return

def mapToComputeLandscape(x):
    s, d, t = neededIndices[x]
    return computeLandscape(s, d, t)
#######################
#Run Script
if __name__ == "__main__":
    with multiprocessing.Pool(threadSize) as p:
        p.map(mapToComputeLandscape, range(len(neededIndices)))
