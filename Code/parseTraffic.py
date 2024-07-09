# Importing the data into a Python-friendly structure
import numpy as np
import pickle
import datetime
import time
import csv

#We assume the data is a CSV file
#One column per sensor, 11 sens
#
#This is configured for a header row and no label column

########################
# Parameters
#sRange: number of sensors
#outputDirectory: output directory
sRange = 11
outputDirectory = "Reference"
directoryOfCSV = "Reference"

########################
#Run the Scripts

with open(f"{directoryOfCSV}/phase1_test.csv", newline="") as csvfile:
    data = list(csv.reader(csvfile))

#Adjust this line for headers.
testData = np.array(data)[1:, :]
print(np.shape(testData))
numberOfDataRows = np.shape(testData)[0]
parsedData = [[[[] for t in range(288)] for d in range(7)] for s in range(sRange)]
badCounter = 0
for w in range(52):
    for d in range(7):
        for t in range(288):
            # there are 288, 5 minute periods a day
            indexStart = w * 7 * 288 + d * 288 + t
            indexStopExclusive = indexStart + 5
            stTime = datetime.datetime.strptime(
                testData[indexStart, 10], "%Y-%m-%d %H:%M:%S"
            ).timetuple()
            if stTime[3] * 60 + stTime[4] != 5 * t:
                print("bad")
            if indexStopExclusive > numberOfDataRows:
                break
            for s in range(sRange):
                vector = testData[indexStart:indexStopExclusive, s]
                try:
                    vector = list(map(int, vector))
                    parsedData[s][d][t].append(vector)
                except:
                    parsedData[s][d][t].append([np.nan, np.nan, np.nan, np.nan, np.nan])
                    badCounter += 1
        if indexStopExclusive > numberOfDataRows:
            break
    if indexStopExclusive > numberOfDataRows:
        break
with open(f"{outputDirectory}/parsedData5test.pkl", "wb") as outp:
    pickle.dump(parsedData, outp, pickle.HIGHEST_PROTOCOL)
print(f"Number of vectors skipped due to missing data:{badCounter}")
