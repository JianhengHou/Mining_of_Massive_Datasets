from pyspark import SparkContext
import json
import sys
import time

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(sys.stderr, "<input_file_name> <out_file_name> n_partition")
        exit(-1)

    def count_partition(iterator):
        partition_number = []
        par = list(iterator)
        partition_number.append(len(par))
        return partition_number

    def partition_func(key):
        return hash(key)

    sc = SparkContext(appName="DataExploration1")

    lines = sc.textFile(sys.argv[1])
    # lines = sc.textFile("yelp_dataset/review.json")
    lines = lines.map(lambda x: json.loads(x))
    start1 = time.time()
    default_partition_number = lines.getNumPartitions()
    default_partition_elements = lines.mapPartitions(count_partition).collect()
    distinct_businesses = lines.map(lambda x: (x['business_id'], 1)).countByKey()
    sorted_businesses = sorted(distinct_businesses.items(), key=lambda x: x[1], reverse=True)
    task4_f = sorted_businesses[:10]  # The top 10 businesses that had the largest numbers of reviews and the number of reviews they had
    end1 = time.time()
    default_duration = end1-start1

    start2 = time.time()
    repartition_num = int(sys.argv[3])
    businesses = lines.map(lambda x: (x['business_id'], 1)).partitionBy(repartition_num, partition_func)
    customized_partition_number = businesses.getNumPartitions()
    customized_partition_elements = businesses.mapPartitions(count_partition).collect()
    distinct_businesses = businesses.countByKey()
    sorted_businesses = sorted(distinct_businesses.items(), key=lambda x: x[1], reverse=True)
    task4_ff = sorted_businesses[:10]  # The top 10 businesses that had the largest numbers of reviews and the number of reviews they had
    end2 = time.time()
    customized_duration = end2 - start2

    output = {"default": {"n_partition": default_partition_number,
                          "n_items": default_partition_elements,
                          "exe_time": default_duration},
              "customized":{"n_partition": customized_partition_number,
                          "n_items": customized_partition_elements,
                          "exe_time": customized_duration},
              "explanation":"Repartitioning with certain hash function would make data hashed distribute in less partitions and make it efficient for reduce manipulation later"}

    with open(sys.argv[2],'w') as out_file:
        out_file.write(json.dumps(output))
    # with open('task2.json','w') as out_file:
        # out_file.write(json.dumps(output))
