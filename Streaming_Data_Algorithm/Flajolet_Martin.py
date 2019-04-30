from datetime import datetime
import binascii
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import json
import sys


class Bloom_Filtering:
    def __init__(self, window, sliding, port, output_path):
        self.window_len = window
        self.sliding_interval = sliding
        self.port = port
        self.f = open(output_path, 'w')
        self.f.write('Time,Ground Truth,Estimation\n')


        self.numOfGroup = 5
        self.numOfHashPerGroup = 10
        a = [1, 3, 5, 7, 11, 13, 17, 19]
        b = [1, 3, 5, 7, 11, 13, 17, 19]
        self.hash_com = [(each1, each2) for each1 in a for each2 in b]

    def writeToFile(self, gt, estm):
        self.f = open(output_path, 'a')
        self.f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ',' + str(gt) + ',' + str(estm) + '\n')

    def count(self, bits):
        index = len(bits) - 1
        count = 0
        while index >= 0:
            if bits[index] == '0':
                count += 1
                index -= 1
            else:
                break
        return count

    def compare(self, x, y):
        result = []
        for each in list(zip(x, y)):
            if self.count(each[0]) > self.count(each[1]):
                result.append(each[0])
            else:
                result.append(each[1])
        return result

    def hash(self, city, groupNum):
        cityNum = int(binascii.hexlify(city), 16)
        bits_list = []
        for each in range(self.numOfHashPerGroup):
            index = each * groupNum
            f = (self.hash_com[index][0] * cityNum + self.hash_com[index][1]) % (2 ** 15)
            bits_list.append(bin(f))
        return bits_list

    def Flajolet_Martin(self, rdd_list):
        batch_city_set = set()
        cities = []
        for line in rdd_list:
            city = json.loads(line)["city"].encode('utf-8')
            cities.append(city)
            if city not in batch_city_set:
                batch_city_set.add(city)
        final_result = []
        for group in range(self.numOfGroup):
            max_r_bits = ['1' for each in range(self.numOfHashPerGroup)]
            for each in cities:
                current_bits_list = self.hash(each, group)
                max_r_bits = self.compare(current_bits_list, max_r_bits)
            final_result.append(int(sum([2 ** self.count(each) for each in max_r_bits]) / self.numOfHashPerGroup))
        final_result.sort()
        numofDistinct = final_result[int((self.numOfGroup + 1) / 2)]
        print(len(batch_city_set), numofDistinct)
        self.writeToFile(len(batch_city_set), numofDistinct)

    def run(self):
        conf = SparkConf() \
            .setAppName("Flajolet_Martin") \
            .set("spark.executor.memory", "4g") \
            .set("spark.driver.host", "localhost")
        sc = SparkContext(conf=conf)
        sc.setLogLevel("OFF")
        ssc = StreamingContext(sc, 5)
        stream = ssc.socketTextStream("localhost", self.port)
        # Return a new DStream which is computed based on windowed batches of the source DStream.
        # now each window is a batch(rdd) in the DStream
        stream.window(self.window_len, self.sliding_interval).foreachRDD(lambda rdd: self.Flajolet_Martin(rdd.collect()))

        ssc.start()
        ssc.awaitTermination()


if __name__ == "__main__":
    # for debug
    # port = 9995
    # output_path = 'task2_output.txt'

    if len(sys.argv) != 3:
        print(sys.stderr, "<port #> <output_filename>")
        exit(-1)

    window_len = 30
    sliding = 10
    port = int(sys.argv[1])
    output_path = sys.argv[2]

    bf = Bloom_Filtering(window_len, sliding, port, output_path)
    bf.run()