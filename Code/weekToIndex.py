#Due to missing data, in the array parsedData[s][d][t] the ith entry may
#not be from the ith week. weekToIndex provides the index for the ith week
#e.g. parsedData[s][d][t][weekToIndex[s][d][t][w]] is a vector of data from
#week w

#If the week is in fact missing, weekToIndex[s][d][t][w] = -1
import numpy as np
import pickle
import csv
#####################
#Parameters
#sRange: number of sensors
sRange = 11
outputDirectory = "Output"
vectorLength = 5
directoryOfCSV = outputDirectory

################
# Run the Script

with open(f"{directoryOfCSV}/phase1_test.csv", newline="") as csvfile:
    data = list(csv.reader(csvfile))
testData = np.array(data)[1:, :]
numberOfDataRows = np.shape(testData)[0]
parsedData = [[[[] for t in range(288)] for d in range(7)] for s in range(sRange)]
weekToIndex = [[[dict() for t in range(288)] for d in range(7)] for s in range(sRange)]
for w in range(52):
    for d in range(7):
        for t in range(288):
            # there are 288, 5 minute periods a day
            indexStart = w * 7 * 288 + d * 288 + t
            indexStopExclusive = indexStart + vectorLength
            if indexStopExclusive > numberOfDataRows:
                for s in range(sRange):
                    weekToIndex[s][d][t][w] = -1
                continue
            for s in range(sRange):
                vector = testData[indexStart:indexStopExclusive, s]
                try:
                    vector = list(map(int, vector))
                    parsedData[s][d][t].append(vector)
                    weekToIndex[s][d][t][w] = len(parsedData[s][d][t])
                except:
                    print(f"{s}-{d}-{t}-{w} is missing")
                    weekToIndex[s][d][t][w] = -1
for s in range(sRange):
    for d in range(7):
        for t in range(288):
            if len(parsedData[s][d][t]) < 50:
                print(len(parsedData[s][d][t]))
                print(f"Warning: {s}-{d}-{t} only has {len(parsedData[s][d][t])} weeks")
with open(f"{outputDirectory}/weekToIndex5test.pkl", "wb") as outp:
    pickle.dump(weekToIndex, outp, pickle.HIGHEST_PROTOCOL)
