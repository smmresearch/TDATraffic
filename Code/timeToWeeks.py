#Due to missing data, not all s,d,t tuples have all weeks
#This script provides a way to go from a s,d,t to weeks with data
#timeToWeeks[s][d][t] = {w | vector from week w included}
import numpy as np
import pickle
import csv

##############################
#Parameters
#sRange: number of sensors
sRange = 11
vectorLength = 5
outputDirectory = "Reference"
directoryOfCSV = "Reference"
#############################
#Run the script 

with open(f"{directoryOfCSV}/phase1_test.csv", newline="") as csvfile:
    data = list(csv.reader(csvfile))

testData = np.array(data)[1:,:]
numberOfDataRows = np.shape(testData)[0]
timeToWeeks = [[[set() for t in range(288)] for d in range(7)] for s in range(sRange)]
for w in range(52):
    for d in range(7):
        for t in range(288):
            # there are 288, 5 minute periods a day
            indexStart = w * 7 * 288 + d * 288 + t
            indexStopExclusive = indexStart + vectorLength
            if indexStopExclusive > numberOfDataRows:
                continue
            for s in range(sRange):
                vector = testData[indexStart:indexStopExclusive, s]
                try:
                    vector = list(map(int, vector))
                    timeToWeeks[s][d][t].add(w)
                except:
                    print(f"Error: {s}-{d}-{t}")
with open(f"{outputDirectory}/timeToWeeks5test.pkl", "wb") as outp:
    pickle.dump(timeToWeeks, outp, pickle.HIGHEST_PROTOCOL)
