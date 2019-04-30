from pyspark import SparkContext
import json
import sys
import time

if __name__ == "__main__":
    # if len(sys.argv) != 5:
    #     print(sys.stderr, "<input_file_name> <out_file_name> n_partition")
    #     exit(-1)

    sc = SparkContext(appName="DataExploration1")

    lines1 = sc.textFile(sys.argv[1])
    # line1 = sc.textFile("yelp_dataset/review.json")
    review = lines1.map(lambda x: json.loads(x)).map(lambda x: (x["business_id"], x["stars"]))
    lines2 = sc.textFile(sys.argv[2])
    # line2 = sc.textFile("yelp_dataset/business.json")
    business = lines2.map(lambda x: json.loads(x)).map(lambda x: (x["business_id"], x["city"]))
    joined_rdd = review.join(business)
    avg = joined_rdd.map(lambda x: (x[1][1], (1, x[1][0]))).reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])).map(lambda x: (x[0], round(float(x[1][1])/x[1][0],2))).sortBy((lambda x: x[1]), False)
    task3_a = avg.collect()

    start1 = time.time()
    task3_b1 = avg.collect()[:10]
    end1 = time.time()


    start2 = time.time()
    task3_b2 = avg.take(10)
    end2 = time.time()


    output2 = {"m1": end1-start1, "m2": end2 - start2,
               "explanation":"For method 1,  collect() fetches the entire RDD as a Python list, and then we print the top 10. But for method 2, we only fetch top 10 elements and print them. The second method will save more time since it does not need to manipulate the entire RDD"}

    with open(sys.argv[3],'w') as out_file1:
        out_file1.write("city,stars\n")
        for each in task3_b2:
            out_file1.write(each[0]+','+str(each[1])+'\n')
    with open(sys.argv[4], 'w') as out_file2:
            out_file2.write(json.dumps(output2))
    # with open('task3_a.txt','w') as out_file1:
    #     out_file1.write("city,stars\n")
    #     for each in task3_b2:
    #         out_file1.write(each[0]+','+str(each[1])+'\n')
    # with open('task3_b.json','w') as out_file2:
    #     out_file2.write(json.dumps(output2))

