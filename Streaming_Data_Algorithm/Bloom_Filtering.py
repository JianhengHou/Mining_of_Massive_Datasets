from datetime import datetime
import binascii
import math
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import json
import sys


class Bloom_Filtering:
    def __init__(self, m, n, port, output_path):
        self.global_filter = [0 for each in range(m)]
        self.city_set = {}
        self.visited_true = {}

        self.lenOfFilter = m
        self.approxNumOfDistinct = n
        self.numOfk = int(self.lenOfFilter / self.approxNumOfDistinct * math.log(2))
        self.port = port
        self.f = open(output_path, 'w')
        self.f.write('Time,FPR\n')
        self.fp = 0
        self.tn = 0
    def writeToFile(self, stats):
        self.f = open(output_path, 'a')
        self.f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ',' + str(stats) + '\n')


    def hash_determine(self, rdd):
        a = [1, 3, 5, 7, 11, 13]
        b = [13, 11, 7, 5, 3, 1]

        for line in rdd:
            city = json.loads(line)['city'].encode('utf-8')
            inSet = None
            if city not in self.city_set:
                self.city_set[city] = 1
                inSet = False
            else:
                self.city_set[city] += 1
                inSet = True

            mappedNum = int(binascii.hexlify(city), 16)
            result = []
            for each in range(self.numOfk):
                f = (a[each] * mappedNum + b[each]) % self.lenOfFilter
                if self.global_filter[f] == 0:
                    result.append(f)
            inArray = None
            # at least one corresponding bit is 0, which means it is not in bits array so that it is a new one
            if len(result) > 0:
                inArray = False
                for each in result:
                    self.global_filter[each] = 1
            else:
                inArray = True
            if not inSet and not inArray:
                self.tn += 1
            if not inSet and inArray:
                self.fp += 1

        fp = float(self.fp) / (self.tn + self.fp)
        self.writeToFile(fp)

    def run(self):
        conf = SparkConf() \
            .setAppName("Bloom_Filtering") \
            .set("spark.executor.memory", "4g") \
            .set("spark.driver.host", "localhost")
        sc = SparkContext(conf=conf)
        sc.setLogLevel("OFF")
        ssc = StreamingContext(sc, 10)
        lines = ssc.socketTextStream("localhost", self.port)
        lines.foreachRDD(lambda rdd: self.hash_determine(rdd.collect()))

        ssc.start()
        ssc.awaitTermination()


if __name__ == "__main__":
    # for debug
    # port = 9997
    # output_path = 'task1_output.txt'

    if len(sys.argv) != 3:
        print(sys.stderr, "<port #> <output_filename>")
        exit(-1)

    lenOfFilter = 200
    approxNumOfDistinct = 100
    port = int(sys.argv[1])
    output_path = sys.argv[2]

    bf = Bloom_Filtering(lenOfFilter, approxNumOfDistinct, port, output_path)
    bf.run()