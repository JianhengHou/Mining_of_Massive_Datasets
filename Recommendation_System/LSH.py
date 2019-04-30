from pyspark import SparkContext, SparkConf
from sys import *
import time
from itertools import combinations


class LocalSensitityHashing:
    def __init__(self, numOfBands, numOfRows, threshold, inputPath, outputPath ):
        self.numOfBands = numOfBands
        self.numOfRows = numOfRows
        self.threshold = threshold
        self.numOfMinHash = numOfBands * numOfRows
        self.inputPath = inputPath
        self.outputPath = outputPath

        self.numOfUser = 0
        self.stringUserToIndex = {}
        self.stringItemToIndex = {}
        self.IndexToStringItem = {}
        self.ItemAndUser = {}
        self.result = []
        self.pairs = set()

        # prime generator
        self.prime = []
        j = 0
        for j in range(1, 1000):
            if self.isPrime(j):
                self.prime.append(j)

    def isPrime(self, num):
        i = 2
        while (i < num):
            if (0 == num % i):
                return False
                break
            else:
                i = i + 1
        return True

    def readAndSplit(self, line):
        splited_line = line.strip().split(',')
        return (splited_line[0], splited_line[1])

    def replaceIndex(self, indexDic, usersOrItems):
        i = 0
        for userOrItem in usersOrItems:
            indexDic[userOrItem] = i
            i += 1
        return

    def create_signature(self, userline, length):
        user_signature = []
        for hashFun in range(self.numOfMinHash):
            min_hash = maxsize
            for user in userline:
                min_hash = min(min_hash, self.generateHash(user, hashFun, length))
            user_signature.append(min_hash)
        return user_signature

    def generateHash(self, row, hashNum, bins):
        return (self.prime[hashNum] * row + 5 * hashNum) % bins

    def bandHash(self, rows):
        value_string = ""
        for each in rows:
            value_string += str(each) + "-"
        return value_string

    def calculateJaccard(self, listA, listB):
        numerator = len(set(listA).intersection(set(listB)))
        denominator = len(set(listA).union(set(listB)))
        return float(numerator)/denominator

    def run(self):
        startTime = time.time()
        conf = SparkConf() \
            .setAppName("Similar_Items_LSH") \
            .set("spark.executor.memory", "4g")\
            .set("spark.driver.host", "localhost")
        sc = SparkContext(conf=conf)

        inputData = sc.textFile(self.inputPath)

        # Drop the header
        header = inputData.first()
        inputData = inputData.filter(lambda line: line != header)

        # read and split data into tuples
        Raw_RDD = inputData.map(self.readAndSplit)

        # create user index
        UniqueUser = Raw_RDD.map(lambda line: line[0]).distinct().collect()
        self.replaceIndex(self.stringUserToIndex, UniqueUser)
        self.numOfUser = len(self.stringUserToIndex)

        # create item index {"dlksa":1} and also create its reversed version {1:"dlksa"}
        UniqueItem = Raw_RDD.map(lambda line: line[1]).distinct().collect()
        self.replaceIndex(self.stringItemToIndex, UniqueItem)
        self.numOfItem = len(self.stringItemToIndex)
        self.IndexToStringItem = {value: key for key, value in self.stringItemToIndex.items()}

        # 1. replace user string with index
        # 2. replace item string with index
        # 3. group by item: business1:[user1, user3, user11]
        ItemAndUsers = Raw_RDD.map(lambda line: (self.stringItemToIndex[line[1]], self.stringUserToIndex[line[0]]))\
                                .groupByKey()

        # store the global dict {business1:[user1, user3, user11]} for computation of Jaccard similarity
        self.ItemAndUser = ItemAndUsers.collectAsMap()

        # get minhash user index for each item
        ItemAndMinHash_RDD = ItemAndUsers.map(lambda x: (x[0], self.create_signature(x[1], self.numOfUser)))

        # 1. each part (i.e. rows) of the signature of businesses as the key, business_id as value e.g. "2-3-0" : 22
        # 2. groupByKey, e.g "2-3-0": [12,22,78]
        # 3. filter single item out
        # 4. get pairs by combinations per each key
        candidatePairs = sc.emptyRDD()
        for band in range(self.numOfBands):
            bandPairs = ItemAndMinHash_RDD.map(lambda x: (self.bandHash(x[1][band * self.numOfRows:(band + 1) * self.numOfRows]), x[0]))\
                                            .groupByKey()\
                                                .filter(lambda x: len(x[1]) > 1)\
                                                    .flatMap(lambda x: combinations(sorted(x[1]), 2))
            candidatePairs = candidatePairs.union(bandPairs)

        # 1. calculate the jaccard similarity of candidate pairs
        # 2. filter those whose original similarity >= 0.5
        # 3. mapping index of business to their string id in the alphabetical order
        finalPairs = candidatePairs.map(lambda x: (x, self.calculateJaccard(self.ItemAndUser[x[0]], self.ItemAndUser[x[1]])))\
                                        .filter(lambda x: x[1] >= self.threshold)\
                                                .map(lambda x: ((self.IndexToStringItem[x[0][0]], self.IndexToStringItem[x[0][1]]), x[1]) if self.IndexToStringItem[x[0][0]] < self.IndexToStringItem[x[0][1]] else ((self.IndexToStringItem[x[0][1]],self.IndexToStringItem[x[0][0]]), x[1])).distinct().sortByKey()

        #  write result into a file
        self.result = finalPairs.collect()

        #  For evaluation
        self.pairs = set(finalPairs.map(lambda x: x[0]).collect())
        sc.stop()
        print("Finish Time:", time.time() - startTime)

    def write(self):
        with open(self.outputPath,'w') as f:
            f.write("business_id_1, business_id_2, similarity\n")
            for pair in self.result:
                f.write(pair[0][0] + "," + pair[0][1] + "," + str(round(pair[1], 2)) + "\n")

    def evaluate(self):
        groundTruth_file = "pure_jaccard_similarity.csv"

        sc = SparkContext(appName="Evaluation")
        testData = sc.textFile(groundTruth_file)

        # Drop the header
        header = testData.take(1)
        testData = set(testData.filter(lambda line: line != header).map(self.readAndSplit).map(lambda x: (x[0],x[1])).collect())

        # evaluation
        if len(self.pairs):
            TP = len(testData.intersection(self.pairs))
            Recall = TP / len(testData)
            Precision = TP / len(self.pairs)
            print("Precision:", Precision, " Recall:", Recall)
        sc.stop()

if __name__ == '__main__':
    # input_path = "yelp_train.csv"
    # output_path = "output.csv"
    if len(argv) != 3:
        print(stderr, "<train_file_name> <output_file_name>")
        exit(-1)
    input_path = argv[1]
    output_path = argv[2]

    numOfBands = 12
    numOfRows = 2
    threshold = 0.5

    lsh = LocalSensitityHashing(numOfBands, numOfRows, threshold, input_path, output_path)
    lsh.run()
    lsh.write()
    # lsh.evaluate()

