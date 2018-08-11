import pandas as pd

trainArraysPath = 'original_files\\train-arrays.csv'
testArraysPath = 'original_files\\test-arrays.csv'
trainTargetPath = 'original_files\\train-target.csv'

trainArraysCuttedPath = 'chunks\\train-arrays-cutted.csv'
testArraysCuttedPath = 'chunks\\test-arrays-cutted.csv'
trainTargetCuttedPath = 'chunks\\train-target-cutted.csv'

#divide train-arrays into chunks of 10, 100, 1000, 10000, 100000, 1000000 

def divideIntoChunks():
    with open("chunks\\length10rows.csv","w") as l10, open("chunks\\length100rows.csv","w") as l100, open("chunks\\length1000rows.csv","w") as l1000, open("chunks\\length10000rows.csv","w") as l10000, open("chunks\\length100000rows.csv","w") as l100000, open("chunks\\length1000000rows.csv","w") as l1000000:
        fileArray = [l10,l100,l1000,l10000,l100000,l1000000]
        for chunkCsv in fileArray:
            chunkCsv.write("Id,Length,Data")
        for chunk in pd.read_csv('original_files\\train-arrays.csv', sep=",", names=['Id', 'Length', 'Data'], chunksize=1):
            for row in chunk.itertuples(index=True):
                currIndex = str(getattr(row, 'Length')).count('0') - 1
                fileArray[currIndex].write("\n" + str(getattr(row, 'Id')) + "," + str(getattr(row, 'Length')) + "," + str(getattr(row, 'Data')))
    return


#ids from 10 and 1000000 -> 2000 - 9999

# cut from arrays file the arrays with length 10 and 1000000

def cutFromArrays(target, source):
    counter = 0
    with open(target,"w") as output:
        for chunk in pd.read_csv(source, sep=",", names=['Id', 'Length', 'Data'], chunksize=1):
            for row in chunk.itertuples(index=True):
                if(str(getattr(row, 'Length')) == "1000000"):
                    exit()
                if(str(getattr(row, 'Length')) != "10"):
                    outputLine =  str(getattr(row, 'Id')) + "," + str(getattr(row, 'Length')) + "," + str(getattr(row, 'Data'))
                    outputWrite = ""
                    if(counter != 0):
                        outputWrite = "\n" + outputLine
                    else:
                        outputWrite = outputLine
                        counter += 1
                    output.write(outputWrite) 
    return

divideIntoChunks()
