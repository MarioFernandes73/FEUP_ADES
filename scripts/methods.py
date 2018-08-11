import pandas as pd
import numpy as np
import math as math
import copy

import heapSort as hs
import mergeSort as ms
import time

trainArraysPath = 'original_files\\train-arrays.csv'
testArraysPath = 'original_files\\test-arrays.csv'
trainTargetPath = 'original_files\\train-target.csv'

trainArraysCuttedPath = 'chunks\\train-arrays-cutted.csv'
testArraysCuttedPath = 'chunks\\test-arrays-cutted.csv'
trainTargetCuttedPath = 'chunks\\train-target-cutted.csv'

lengthProcessedPath = 'metric_length\\length_processed.csv'
lengthTestArrayPath = 'metric_length\\length_testarrays.csv'

negativeProcessedPath = 'metric_negative\\negative_processed.csv'
negativeTestArrayPath = 'metric_negative\\negative_testarrays.csv'

sequenceProcessedPath = 'metric_sequence\\sequence_processed.csv'
sequenceTestArrayPath = 'metric_sequence\\sequence_testarrays.csv'

sequence2by2ProcessedPath = 'metric_sequence2by2\\sequence2by2_processed.csv'
sequence2by2TestArrayPath = 'metric_sequence2by2\\sequence2by2_testarrays.csv'

stdProcessedPath = 'metric_std\\std_processed.csv'
stdTestArrayPath = 'metric_std\\std_testarrays.csv'

sampleProcessedPath = 'metric_sample\\sample_processed.csv'
sampleTestArrayPath = 'metric_sample\\sample_testarrays.csv'

inversionProcessedPath = 'metric_inversion\\inversion_processed.csv'
inversionTestArrayPath = 'metric_inversion\\inversion_testarrays.csv'


#MIX

lengthSequence2by2ProcessedPath = 'metric_length_sequence2by2\\length_sequence2by2_processed.csv'
lengthSequence2by2TestArrayPath = 'metric_length_sequence2by2\\length_sequence2by2_testarrays.csv'

lengthNegativeProcessedPath = 'metric_length_negative\\length_negative_processed.csv'
lengthNegativeTestArrayPath = 'metric_length_negative\\length_negative_testarrays.csv'

allProcessedPath = 'metric_all5\\all5_processed.csv'
allTestArrayPath = 'metric_all5\\all5_testarrays.csv'

finalProcessedPath = 'metric_final\\final_processed.csv'
finalTestArrayPath = 'metric_final\\final2_testarrays.csv'



#8400 rows
def processDeployCSV():
    with open("deliverable_length.csv","w") as output:
    for chunk in pd.read_csv('predicted_length.csv', sep=",", names=['Length', 'Id', 'Unlabelled','Confidence','Error','Label'], chunksize=50):
        for row in chunk.itertuples(index=True):
            outputLine = str(int(getattr(row, 'Id'))) + ","+ str(getattr(row, 'Label'))
            output.write("\n" + outputLine)
    return

def processTestArrays(targetFile, sourceFile):
    with open(targetFile,"w") as output:
        output.write("Id,LENGTH,NEGATIVE,SEQUENCE,SEQUENCE2by2,STD,MERGE1,MERGE2,MERGE3,MERGE4,MERGE5,MERGE6,MERGE7,MERGE8,HEAP1,HEAP2,HEAP3,HEAP4,HEAP5,HEAP6,HEAP7,HEAP8,MIX1,MIX2,MIX3,MIX4,MIX5,MIX6,MIXDIF,Predicted")
        for chunk in pd.read_csv(sourceFile, sep=",", names=['Id', 'Length', 'Data'], chunksize=50):
            for row in chunk.itertuples(index=True):
                data = np.fromstring(getattr(row, 'Data')[1:len(getattr(row, 'Data'))-1], dtype=int, sep=' ') #Array with data


                # LENGTH PARAMETER

                outputParameter1 = len(data)

                if(outputParameter1 < 10001):
                    continue

                # NEGATIVE % PARAMETER

                outputParameter2 = (sum(n < 0 for n in data) / len(data))  
                
                # SEQUENCE PARAMETER
                
                counter = 0
                previousNum = 0
                for num in data:
                    if(num > previousNum):
                        counter += 1
                    previousNum = num
                outputParameter3 = counter / len(data)
                
                #SEQUENCE2by2PARAMETER
                
                counter = 0
                previousNum = 0
                firstNum = True
                for num in data:
                    if(firstNum):
                        previousNum = num
                    else:
                        if(num>previousNum):
                            counter += 1
                    firstNum = not firstNum
                outputParameter4 = counter / len(data)
                
                #STD PARAMETER
                outputParameter5 = np.std(data,ddof=1)


                #INVERSIONS 
                #outputParameter6 = getInvCount(data,len(data))

                #Merge sort execution time (sum of 3 times / 3) * 1000000       
                outputParameter7 = getMsTime(0.1,data) 
                outputParameter8 = getMsTime(0.15,data) 
                outputParameter9 = getMsTime(0.20,data) 
                outputParameter10 = outputParameter9 - outputParameter7
                outputParameter11 = outputParameter8 - outputParameter7
                outputParameter12 = outputParameter9 - outputParameter8
                outputParameter13 = outputParameter7 + outputParameter8 + outputParameter9
                outputParameter14 = ( outputParameter9 / outputParameter7 ) * 1000

                #Heap sort execution time (sum of 3 times / 3) * 1000000
                outputParameter15 = getHsTime(0.1,data) 
                outputParameter16 = getHsTime(0.15,data) 
                outputParameter17 = getHsTime(0.20,data) 
                outputParameter18 = outputParameter17 - outputParameter15
                outputParameter19 = outputParameter16 - outputParameter15
                outputParameter20 = outputParameter17 - outputParameter16
                outputParameter21 = outputParameter15 + outputParameter16 + outputParameter17
                outputParameter22 = ( outputParameter17 / outputParameter15 ) * 1000
                

                #Mix of both execution times

                outputParameter23 = outputParameter15 - outputParameter7
                outputParameter24 = outputParameter16 - outputParameter8
                outputParameter25 = outputParameter17 - outputParameter9
                outputParameter26 = (outputParameter15 / outputParameter7) * 1000
                outputParameter27 = (outputParameter16 / outputParameter8) * 1000
                outputParameter28 = (outputParameter17 / outputParameter9) * 1000

                #How many differences are positive
                outputParameter29 = 0
                if outputParameter23 > 0:
                    outputParameter29 = outputParameter29 + 1
                if outputParameter24 > 0:
                    outputParameter29 = outputParameter29 + 1
                if outputParameter25 > 0:
                    outputParameter29 = outputParameter29 + 1
                
                params1 = str(outputParameter1) + "," + str(outputParameter2) + "," + str(outputParameter3) + "," + str(outputParameter4) + "," + str(outputParameter5)
                params2 = str(outputParameter7) + "," + str(outputParameter8) + "," + str(outputParameter9) + "," + str(outputParameter10)
                params3 = str(outputParameter11) + "," + str(outputParameter12) + "," + str(outputParameter13) + "," + str(outputParameter14) + "," + str(outputParameter15)
                params4 = str(outputParameter16) + "," + str(outputParameter17) + "," + str(outputParameter18) + "," + str(outputParameter19) + "," + str(outputParameter20)
                params5 = str(outputParameter21) + "," + str(outputParameter22) + "," + str(outputParameter23) + "," + str(outputParameter24) + "," + str(outputParameter25)
                params6 = str(outputParameter26) + "," + str(outputParameter27) + "," + str(outputParameter28) + "," + str(outputParameter29)

                outputLine = str(getattr(row, 'Id')) + "," + params1 + "," + params2 + "," + params3 + "," + params4 + "," + params5 + "," + params6 + ","
                output.write("\n" + outputLine)
    return


def processTrainArrays(targetFile, trainTargetFile, trainArraysFile):
    with open(targetFile,"w") as output, open(trainTargetFile,"r") as tt:
        targets = tt.readlines()
        output.write("Id,LENGTH,NEGATIVE,SEQUENCE,SEQUENCE2by2,STD,MERGE1,MERGE2,MERGE3,MERGE4,MERGE5,MERGE6,MERGE7,MERGE8,HEAP1,HEAP2,HEAP3,HEAP4,HEAP5,HEAP6,HEAP7,HEAP8,MIX1,MIX2,MIX3,MIX4,MIX5,MIX6,MIXDIF,Predicted\n")
        for chunk in pd.read_csv(trainArraysFile, sep=",", names=['Id', 'Length', 'Data'], chunksize=50):
            for row in chunk.itertuples(index=True):
                data = np.fromstring(getattr(row, 'Data')[1:len(getattr(row, 'Data'))-1], dtype=int, sep=' ') #Array with data
                currTarget = targets[getattr(row, 'Index') + 1]

                # LENGTH PARAMETER

                outputParameter1 = len(data)

                if(outputParameter1 < 10001):
                    continue

                # NEGATIVE % PARAMETER

                outputParameter2 = (sum(n < 0 for n in data) / len(data))  
                
                # SEQUENCE PARAMETER
                
                counter = 0
                previousNum = 0
                for num in data:
                    if(num > previousNum):
                        counter += 1
                    previousNum = num
                outputParameter3 = counter / len(data)
                
                #SEQUENCE2by2PARAMETER
                
                counter = 0
                previousNum = 0
                firstNum = True
                for num in data:
                    if(firstNum):
                        previousNum = num
                    else:
                        if(num>previousNum):
                            counter += 1
                    firstNum = not firstNum
                outputParameter4 = counter / len(data)
                
                #STD PARAMETER
                outputParameter5 = np.std(data,ddof=1)


                #INVERSIONS 
                #outputParameter6 = getInvCount(data,len(data))

                #Merge sort execution time (sum of 3 times / 3) * 1000000       
                outputParameter7 = getMsTime(0.1,data) 
                outputParameter8 = getMsTime(0.15,data) 
                outputParameter9 = getMsTime(0.20,data) 
                outputParameter10 = outputParameter9 - outputParameter7
                outputParameter11 = outputParameter8 - outputParameter7
                outputParameter12 = outputParameter9 - outputParameter8
                outputParameter13 = outputParameter7 + outputParameter8 + outputParameter9
                outputParameter14 = ( outputParameter9 / outputParameter7 ) * 1000

                #Heap sort execution time (sum of 3 times / 3) * 1000000
                outputParameter15 = getHsTime(0.1,data) 
                outputParameter16 = getHsTime(0.15,data) 
                outputParameter17 = getHsTime(0.20,data) 
                outputParameter18 = outputParameter17 - outputParameter15
                outputParameter19 = outputParameter16 - outputParameter15
                outputParameter20 = outputParameter17 - outputParameter16
                outputParameter21 = outputParameter15 + outputParameter16 + outputParameter17
                outputParameter22 = ( outputParameter17 / outputParameter15 ) * 1000
                

                #Mix of both execution times

                outputParameter23 = outputParameter15 - outputParameter7
                outputParameter24 = outputParameter16 - outputParameter8
                outputParameter25 = outputParameter17 - outputParameter9
                outputParameter26 = (outputParameter15 / outputParameter7) * 1000
                outputParameter27 = (outputParameter16 / outputParameter8) * 1000
                outputParameter28 = (outputParameter17 / outputParameter9) * 1000

                #How many differences are positive
                outputParameter29 = 0
                if outputParameter23 > 0:
                    outputParameter29 = outputParameter29 + 1
                if outputParameter24 > 0:
                    outputParameter29 = outputParameter29 + 1
                if outputParameter25 > 0:
                    outputParameter29 = outputParameter29 + 1
                
                params1 = str(outputParameter1) + "," + str(outputParameter2) + "," + str(outputParameter3) + "," + str(outputParameter4) + "," + str(outputParameter5)
                params2 = str(outputParameter7) + "," + str(outputParameter8) + "," + str(outputParameter9) + "," + str(outputParameter10)
                params3 = str(outputParameter11) + "," + str(outputParameter12) + "," + str(outputParameter13) + "," + str(outputParameter14) + "," + str(outputParameter15)
                params4 = str(outputParameter16) + "," + str(outputParameter17) + "," + str(outputParameter18) + "," + str(outputParameter19) + "," + str(outputParameter20)
                params5 = str(outputParameter21) + "," + str(outputParameter22) + "," + str(outputParameter23) + "," + str(outputParameter24) + "," + str(outputParameter25)
                params6 = str(outputParameter26) + "," + str(outputParameter27) + "," + str(outputParameter28) + "," + str(outputParameter29)

                outputLine = str(getattr(row, 'Id')) + "," + params1 + "," + params2 + "," + params3 + "," + params4 + "," + params5 + "," + params6 + "," + currTarget[currTarget.find(',')+1:len(currTarget)]
                output.write(outputLine)

    return


def getInvCount(arr, n):
 
    inv_count = 0
    for i in range(n):
        for j in range(i+1, n):
            if (arr[i] > arr[j]):
                inv_count += 1
 
    return inv_count


def getMsTime(percentage, data):
    sampleQt = math.ceil(percentage * len(data))
    sample = data[:sampleQt]
    totalTime = 0
    for x in range(3):
        sample2 = copy.deepcopy(sample)
        msStartTime = time.time()
        for y in range(5):
            ms.merge_sort(sample2)
            sample2 = copy.deepcopy(sample)
        totalTime = totalTime + (time.time() - msStartTime)

    return float(totalTime*1000000/3)

def getHsTime(percentage, data):
    sampleQt = math.ceil(percentage * len(data))
    sample = data[:sampleQt]
    totalTime = 0
    for x in range(3):
        sample2 = copy.deepcopy(sample)
        hsStartTime = time.time()
        for y in range(5):
            hs.heap_sort(sample2)
            sample2 = copy.deepcopy(sample)
        totalTime = totalTime + (time.time() - hsStartTime)

    return float(totalTime*1000000/3)

#processTrainArrays(finalProcessedPath,trainTargetCuttedPath,trainArraysCuttedPath)
processTestArrays(finalTestArrayPath,testArraysCuttedPath)



