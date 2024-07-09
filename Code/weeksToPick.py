# Due to missing data, we cannot use all weeks as a basis of comparison for
# all s,d,t tuples
# Due to time shifting, we must consider what weeks are available for all times wanted
# weeksToPickSet[s][d][t] contains all weeks with available vectors after accounting
# for time-shifting (therefore a potentially proper subset of timeToWeeks)


import pickle
import copy
import itertools
###############################
#Parameters

#iRange: set as an upper-bound on amount of time
#   shifting used when creating bags 
#   (using less is okay, but it shouldn't be egregiously large
#   in case of missing data)
#sRange: number of sensors
#dRange: number of days (a week)
#tRange: number of samples a day
#directoryOfReference: directory to find timeToWeeks
iRange = 15
outputDirectory = "Reference"
sRange = 11
dRange = 7
tRange = 288
directoryOfReference = "Reference"

############################33
#Functions



def dtGetter(d, t):
    if t < 0:
        return (d - 1) % 7, t % 288
    elif t >= 288:
        return (d + 1) % 7, t % 288
    else:
        return d, t
################################3
#Run the script

neededIndices = itertools.product(range(sRange), range(dRange), range(tRange))
sdtList = [
    [[[] for t in range(tRange)] for d in range(dRange)] for s in range(sRange)
]
with open(f"{directoryOfReference}/timeToWeeks5test.pkl", "rb") as f:
    timeToWeeks = pickle.load(f)
weeksToPickSet = copy.deepcopy(sdtList)
for s, d, t in neededIndices:
    di, ti = dtGetter(d, t + iRange)
    weeksToPick = timeToWeeks[s][di][ti]
    for i in range(iRange * 2):
        di, ti = dtGetter(d, t - iRange + i)
        weeksToPick = weeksToPick & timeToWeeks[s][di][ti]
    weeksToPickSet[s][d][t] = weeksToPick
    if len(weeksToPick) < 35:
        raise ValueError("iRange too big")
with open(f"./{outputDirectory}/weeksToPick5train-i{iRange}.pkl", "wb") as outp:
    pickle.dump(weeksToPickSet, outp, pickle.HIGHEST_PROTOCOL)

