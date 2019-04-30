import os
os.environ["PYSPARK_SUBMIT_ARGS"] = ( "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell")
from pyspark import SparkContext, SparkConf, SQLContext
from sys import *
import time

from graphframes import *


class CommunityDetection:
    def __init__(self, threshold, input_path, output_path):
        self.threshold = threshold
        self.input_path = input_path
        self.output_path = output_path
        self.global_User_items = {}

    def generate_edges(self, user_items):
        currUser = user_items[0]
        currItems = user_items[1]
        edges = set()
        for otherUser, items in self.global_User_items.items():
            if currUser != otherUser and len(currItems.intersection(items)) >= self.threshold:
                edges.add((currUser, otherUser))
        return edges

    def readAndSplit(self, line):
        splited_line = line.strip().split(',')
        return (splited_line[0], splited_line[1])

    def run(self):
        startTime = time.time()
        conf = SparkConf() \
            .setAppName("Community_Detection_Based_on_GraphFrames") \
            .set("spark.executor.memory", "4g")\
            .set("spark.driver.host", "localhost")
        sc = SparkContext(conf=conf)

        inputData = sc.textFile(self.input_path)

        # Drop the header
        header = inputData.first()
        inputData = inputData.filter(lambda line: line != header)

        # read and split data into tuples
        Standard_RDD = inputData.map(self.readAndSplit)

        UserAndItems = Standard_RDD.groupByKey().map(lambda x: (x[0], set(list(x[1]))))
        self.global_User_items = UserAndItems.collectAsMap()

        edge_RDD = UserAndItems.flatMap(self.generate_edges).filter(lambda x: len(x) > 0)

        vertex_RDD = edge_RDD.flatMap(lambda x: [x[0], x[1]]).distinct().map(lambda x: (x, ))

        sqlContext = SQLContext(sc)

        vertices = sqlContext.createDataFrame(vertex_RDD.collect(), ["id",])

        edges = sqlContext.createDataFrame(edge_RDD.collect(), ["src", "dst",])

        g = GraphFrame(vertices, edges)

        result = g.labelPropagation(maxIter=5)

        verticeRDD = sc.parallelize(result.collect())
        community_list = verticeRDD.map(lambda x: (str(x.label), x.id))\
                                    .groupByKey()\
                                            .map(lambda x: sorted(list(x[1])))\
                                                    .sortBy(lambda x: (len(x), x[0]))\
                                                            .collect()
        with open(self.output_path, 'w') as f:
            for line in community_list:
                for each in line[:-1]:
                    f.write(each + ', ')
                f.write(line[-1] + '\n')
        print("Finish time:", time.time() - startTime)


if __name__ == '__main__':
    #  for debug
    # threshold = 7
    # input_path = 'ub_sample_Data.csv'
    # community_output = 'task1_result.txt'

    if len(argv) != 4:
        print(stderr, "<filter threshold> <input_file_path> <community_output_file_path>")
        exit(-1)

    threshold = int(argv[1])
    input_path = argv[2]
    community_output = argv[3]

    cd = CommunityDetection(threshold, input_path, community_output)
    cd.run()
