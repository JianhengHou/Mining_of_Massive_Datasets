from pyspark import SparkContext
import json
import sys


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(sys.stderr, "<input_file_name> <out_file_name>")
        exit(-1)
    sc = SparkContext(appName="DataExploration")

    lines = sc.textFile(sys.argv[1])
    # lines = sc.textFile("yelp_dataset/review.json")

    task4_a = lines.count()  # The total number of reviews

    lines = lines.map(lambda x: json.loads(x))
    task4_b = lines.filter(lambda x: x['date'].find('2018') != -1).count()  # The number of reviews in 2018

    distinct_users = lines.map(lambda x: (x['user_id'], 1)).countByKey()
    task4_c = len(distinct_users)  # The number of distinct users who wrote reviews

    sorted_users = sorted(distinct_users.items(), key=lambda x: x[1], reverse=True)
    task4_d = sorted_users[:10]  # The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote

    distinct_businesses = lines.map(lambda x: (x['business_id'], 1)).countByKey()
    task4_e = len(distinct_businesses)  # The number of distinct businesses that have been reviewed

    sorted_businesses = sorted(distinct_businesses.items(), key=lambda x: x[1], reverse=True)
    task4_f = sorted_businesses[:10]  # The top 10 businesses that had the largest numbers of reviews and the number of reviews they had


    output = {"n_review": task4_a,
              "n_review_2018": task4_b,
              "n_user": task4_c,
              "top10_user": task4_d,
              "n_business": task4_e,
              "top10_business": task4_f}
              
    with open(sys.argv[2],'w') as out_file:
        out_file.write(json.dumps(output))
    # with open('task1.json','w') as out_file:
    #     out_file.write(json.dumps(output))
