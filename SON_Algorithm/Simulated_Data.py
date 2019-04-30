from pyspark import SparkContext, SparkConf
from sys import *
import time
from itertools import combinations


def apriori(input_baskets, support):
    # For 1-itemset part
    baskets = list(input_baskets)
    num_baskets = len(baskets)
    Candidate_count = {}
    for basket in baskets:
        for item in basket:
            if item in Candidate_count.keys():
                Candidate_count[item] += 1
            else:
                Candidate_count[item] = 1

    # For 1-itemset Candidate
    Candidate1_key = Candidate_count.keys()
    Candidate1_item = []
    for i in Candidate1_key:
        Candidate1_item.append([i])

    # For Frequent 1-itemset
    Frequent1_item = set()
    for k in Candidate1_item:
        if Candidate_count[k[0]] >= support:
            Frequent1_item.add(frozenset(k))
    FrequentItemSet = Frequent1_item  # FrequentItemSet is a set

    # For k-itemset part
    AllFrequentItemSet = {}
    k = 2
    while FrequentItemSet != set([]):
        # add last frequent k-ItemSet that remain
        AllFrequentItemSet[k - 1] = FrequentItemSet
        # generate new k-Itemset
        CandidateiItemSet = getCandiateItemSet(FrequentItemSet, AllFrequentItemSet, k)
        # generate new frequent k-Itemset
        FrequentItemSet = getCurrentFrequent(baskets, CandidateiItemSet, support)
        k += 1

    output = []
    for each in AllFrequentItemSet.values():
        for item in list(each):
            output.append(str(sorted(list(item))).strip('[').strip(']').replace("'", "").replace(' ', ''))
    return output


def getCandiateItemSet(currFrequentItemSet, globalFrequentItemSet, k):
    # generate new candidates by union function
    candiateSet = []
    currFrequentItemSet_sort = list(currFrequentItemSet)
    for index1, itemset1 in enumerate(currFrequentItemSet_sort):
        for itemset2 in currFrequentItemSet_sort[index1 + 1:]:
            unionSet = itemset1.union(itemset2)

            if len(unionSet) == k and checkSubSetFrequent(unionSet, globalFrequentItemSet, k - 1):
                candiateSet.append(unionSet)
    return candiateSet


def checkSubSetFrequent(Set, globalFrequentItemSet, m):
    # check: a k-itemSet is frequent iff its k-1 itemSets are all frequent
    result = True
    comb = combinations(sorted(Set), m)
    for each_comb in comb:
        if frozenset(list(each_comb)) not in globalFrequentItemSet[m]:
            result = False
    return result


def getCurrentFrequent(baskets, CandidateItemSet, support):
    # count each candidate itemset by scanning the whole basket
    count = getCount(baskets, CandidateItemSet)
    # filter candidates that do not meet the requirement of the threshold
    return_candidate = set()
    for i, item in enumerate(CandidateItemSet):
        if count[i] >= support:
            return_candidate.add(item)
    return return_candidate


def getCount(baskets, CandidateItemSet):
    # count each candidate itemset by scanning the whole basket
    count_list = []
    for item in CandidateItemSet:
        count_each = 0
        for T in baskets:
            if len(frozenset(T).intersection(item)) == len(item):
                count_each += 1
        count_list.append(count_each)
    return count_list


def getCountSON(whole, candiate):
    # count each candidate itemset by scanning the whole basket
    count_list = []
    for item in candiate:
            count_each = 0
            for T in whole:
                new_T = str(T).strip('[').strip(']').replace(' ', '').replace("'", "").split(',')
                new_item = str(item).strip('[').strip(']').replace(' ', '').replace("'", "").split(',')
                if len(frozenset(new_T).intersection(set(new_item))) == len(new_item):
                    count_each += 1
            out = str(sorted(item[0].split(','))).strip('[').strip(']').replace(' ', '').replace("'", "")
            count_list.append((out, count_each))
    return count_list


def process_output(data):
    data_dic = {}
    for each_cand in data:
        length = len(each_cand.split(','))
        if length not in data_dic.keys():
            data_dic[length] = ['(' + each_cand.replace(',', ', ') + ')']
        else:
            data_dic[length].append('(' + each_cand.replace(',', ', ') + ')')
    return data_dic


def writeTo(Candidates_data, FrequentItemsets, output_path):
    f = open(output_path, 'w')
    Candidates_output = process_output(Candidates_data)
    FrequentItemsets_output = process_output(FrequentItemsets)
    f.write('Candidates:\n')
    for num1, items1 in Candidates_output.items():
        f.write(str(sorted(items1))[1:-1]+'\n')
    f.write('Frequent Itemsets:\n')
    for num2, items2 in FrequentItemsets_output.items():
        f.write(str(sorted(items2))[1:-1]+'\n')
    f.close()


def main():
    if len(argv) != 5:
        print(stderr, "<Filter threshold> <Support> <Input_file_path> <Output_file_path>")
        exit(-1)

    caseNo = int(argv[1])
    support = int(argv[2])
    input_file = argv[3]
    output_file = argv[4]

    # code here only for debug
    # caseNo = 1
    # support = 4
    # input_file= 'small1.csv'
    # output_file = 'task1_output.txt'

    start_time = time.time()
    conf1 = (SparkConf().setMaster("local[*]"))
    sc = SparkContext(appName="SONAlgorithm", conf=conf1)

    dataRDD_header = sc.textFile(input_file)
    dataRDD = dataRDD_header.filter(lambda string: '_' not in string)


    if caseNo == 1:
        # (user_id, business_id)
        dataRDD = dataRDD.map(lambda string: (string.strip().split(',')[0], string.strip().split(',')[1]))
    elif caseNo == 2:
        # (business_id, user_id)
        dataRDD= dataRDD.map(lambda string: (string.strip().split(',')[1], string.strip().split(',')[0]))

    # reduce the redundancy in values
    basketRDD = dataRDD.reduceByKey(lambda x, y: x + ',' + y if y not in x else x). \
        map(lambda x: sorted([each for each in x[1].split(',')]))
    basket = basketRDD.collect()

    # set threshold for each partition, or say, chunk
    numPartition = basketRDD.getNumPartitions()
    threshold = 1
    if support / numPartition > threshold:
        threshold = support / numPartition

    # SON phase 1
    tempCandidatesItemSet = basketRDD.mapPartitions(lambda chunk: apriori(chunk, threshold)).map(
        lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).map(lambda x: x[0])
    CandidatesItemSet = tempCandidatesItemSet.map(lambda x: [x])

    # SON phase 2  There are some problems here: basket should be divided into chunks when counting candiates, and use reduce to get global support
    allFrequentItemSet = CandidatesItemSet.mapPartitions(lambda candidates: getCountSON(list(basket), list(candidates))).\
                                                filter(lambda x,: x[1] >= support).map(lambda x: x[0]).collect()

    # write candidates and frequentItemSet into a file
    writeTo(tempCandidatesItemSet.collect(), allFrequentItemSet, output_file)
    print("Duration: ", time.time() - start_time)


if __name__ == "__main__":
    main()


