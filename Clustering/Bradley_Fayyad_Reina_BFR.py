from numpy import *
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
import numpy as np
import time
import sys
import copy


class Bradley_Fayyad_Reina:
    def __init__(self, k, input_path, output_path):
        self.initial_k = k * 10
        self.k = k
        self.sample_size = 0.2
        self.numOfRound = 0
        self.input_path = input_path
        self.totalRounds = int(1 / self.sample_size)
        self.distance_threshold = 0

        self.global_stat = {"DS": {}, "CS": {}, "RS": {}}
        self.global_CS_id = 0

        self.o = open(output_path, 'w')


    def readfile(self):
        raw_data = []
        with open(self.input_path) as input:
            for line in input.readlines():
                raw_data.append([float(each) for each in line.strip().split(',')[2:]])
        return raw_data

    def mahalanobis_distance(self, point, cluster):
        dist = 0
        for d in range(len(point)):
            dist += ((point[d] - cluster["SUM"][d] / cluster["N"]) ** 2) / ((cluster["SUMSQ"][d] / cluster["N"]) - ((cluster["SUM"][d] / cluster["N"]) ** 2))
        dist = sqrt(dist)
        return dist

    def finish(self):
        # merge CS into nearest cluster
        for cs_i, group in copy.deepcopy(self.global_stat["CS"]).items():
            min_CS_DS_dist = sys.maxsize
            min_DS_index = ""
            existMerge = False
            for key, values in copy.deepcopy(self.global_stat["DS"]).items():
                CS_point = group["SUM"] / group["N"]
                cur_dist = self.mahalanobis_distance(CS_point, values)
                if cur_dist < self.distance_threshold and cur_dist < min_CS_DS_dist:
                    min_CS_DS_dist = cur_dist
                    min_DS_index = key
                    existMerge = True
            if existMerge:
                self.global_stat["DS"][min_DS_index]["N"] += group["N"]
                self.global_stat["DS"][min_DS_index]["point_index"] += group["point_index"]
                self.global_stat["DS"][min_DS_index]["SUM"] += group["SUM"]
                self.global_stat["DS"][min_DS_index]["SUMSQ"] = group["SUMSQ"]
                del self.global_stat["CS"][cs_i]
        for rest_id, rest_group in copy.deepcopy(self.global_stat["CS"]).items():
            self.global_stat["RS"]["index"] += rest_group["point_index"]
            del self.global_stat["CS"][rest_id]

    def merge_CS(self):
        existMerge = False
        while True:
            temp = copy.deepcopy(self.global_stat["CS"])
            # current CS
            for key1, value1 in temp.items():
                existMerge = False
                min_CS_dist = sys.maxsize
                min_CS_index = ""
                #  target CS
                for key2, value2 in temp.items():
                    if key1 != key2:
                        point = value1["SUM"] / value1["N"]
                        current_dist = self.mahalanobis_distance(point, value2)
                        if current_dist < self.distance_threshold and current_dist < min_CS_dist:
                            min_CS_dist = current_dist
                            min_CS_index = key2
                            existMerge = True
                if existMerge:
                    #  merge value2 into the value1
                    self.global_stat["CS"][min_CS_index]["N"] += self.global_stat["CS"][key1]["N"]
                    self.global_stat["CS"][min_CS_index]["point_index"] += self.global_stat["CS"][key1]["point_index"]
                    self.global_stat["CS"][min_CS_index]["SUM"] += self.global_stat["CS"][key1]["SUM"]
                    self.global_stat["CS"][min_CS_index]["SUMSQ"] += self.global_stat["CS"][key1]["SUMSQ"]
                    del self.global_stat["CS"][key1]
                    break
            if not existMerge:
                break

    def generate_CS_RS(self):
        rs_data = self.global_stat["RS"]["data"]
        size = len(rs_data)
        if self.initial_k/2 > size:
            cumOfCluster = size
        else:
            cumOfCluster = int(self.initial_k/2)
        kmeans = KMeans(n_clusters=cumOfCluster, random_state=0, n_jobs=-1).fit(rs_data)
        rs_data_labels = kmeans.labels_
        count_dict = {}
        # count for finding outliers
        for label, i, pos in list(zip(rs_data_labels, self.global_stat["RS"]["index"], list(range(len(self.global_stat["RS"]["index"]))))):
            if label not in count_dict:
                count_dict[label] = {"index": [i], "size": 1, "position": [pos]}
            else:
                count_dict[label]["index"].append(i)
                count_dict[label]["position"].append(pos)
                count_dict[label]["size"] += 1

        #  filter points into CS and do stat for each group in CS
        for label, details in count_dict.items():
            if details["size"] > 1:
                    self.global_CS_id += 1
                    self.global_stat["CS"][self.global_CS_id] = {}
                    self.global_stat["CS"][self.global_CS_id]["N"] = details["size"]
                    self.global_stat["CS"][self.global_CS_id]["SUM"] = np.zeros(len(rs_data[0]))
                    self.global_stat["CS"][self.global_CS_id]["SUMSQ"] = np.zeros(len(rs_data[0]))
                    for p in details["position"]:
                        self.global_stat["CS"][self.global_CS_id]["SUM"] += np.array(rs_data[p])
                        self.global_stat["CS"][self.global_CS_id]["SUMSQ"] += np.array(rs_data[p]) ** 2
                    self.global_stat["CS"][self.global_CS_id]["point_index"] = details["index"]


        # index for outlier
        self.global_stat["RS"]["index"] = [cluster["index"][0] for cluster in count_dict.values() if cluster["size"] == 1]
        # data for outlier
        self.global_stat["RS"]["data"] = [rs_data[cluster["position"][0]] for cluster in count_dict.values() if cluster["size"] == 1]

    def main(self, data, index):
        #  check which cluster the coming data belongs to
        temp_count = 0

        for each, i in list(zip(data, index)):
            assign_DS = False
            min_DS_dist = sys.maxsize
            min_DS_index = ""
            #  check if it belongs to any cluster in DS, if so, find the nearest one
            for cluster_key, cluster in self.global_stat["DS"].items():
                current_dist = self.mahalanobis_distance(each, cluster)
                if current_dist < self.distance_threshold and current_dist < min_DS_dist:
                    assign_DS = True
                    min_DS_dist = current_dist
                    min_DS_index = cluster_key
            if assign_DS:
                temp_count += 1
                # update the target cluster in the DS
                self.global_stat["DS"][min_DS_index]["point_index"].append(i)
                self.global_stat["DS"][min_DS_index]["N"] += 1
                self.global_stat["DS"][min_DS_index]["SUM"] += np.array(each)
                self.global_stat["DS"][min_DS_index]["SUMSQ"] += np.array(each) ** 2
            else:
                #  no chance to get into DS, now check if it belongs to any cluster in CS
                assign_CS = False
                min_CS_dist = sys.maxsize
                min_CS_index = ""
                for cluster_key, cluster in self.global_stat["CS"].items():
                    current_dist = self.mahalanobis_distance(each, cluster)
                    if current_dist < self.distance_threshold and current_dist < min_CS_dist:
                        assign_CS = True
                        min_CS_dist = current_dist
                        min_CS_index = cluster_key
                if assign_CS:
                    # update the target cluster in the CS
                    self.global_stat["CS"][min_CS_index]["point_index"].append(i)
                    self.global_stat["CS"][min_CS_index]["N"] += 1
                    self.global_stat["CS"][min_CS_index]["SUM"] += np.array(each)
                    self.global_stat["CS"][min_CS_index]["SUMSQ"] += np.array(each) ** 2
                else:
                    #  no chance to get into CS, add it into RS
                    self.global_stat["RS"]["index"].append(i)
                    self.global_stat["RS"]["data"].append(each)
        self.generate_CS_RS()
        self.merge_CS()

    def initial_CS(self, rs_data, rs_index):
        size = len(rs_data)
        if self.initial_k / 2 > size:
            cumOfCluster = size
        else:
            cumOfCluster =int(self.initial_k / 2)
        kmeans = KMeans(n_clusters=cumOfCluster, random_state=0, n_jobs=-1).fit(rs_data)
        rs_data_labels = kmeans.labels_
        count_dict = {}

        # count for finding outliers
        for label, i, pos in list(zip(rs_data_labels, rs_index, list(range(len(rs_data))))):
            if label not in count_dict:
                count_dict[label] = {"index": [i], "size": 1, "position": [pos]}
            else:
                count_dict[label]["index"].append(i)
                count_dict[label]["position"].append(pos)
                count_dict[label]["size"] += 1

        #  filter points into CS and do stat for each group in CS
        for label, details in count_dict.items():
            if details["size"] > 1:
                self.global_CS_id += 1
                self.global_stat["CS"][self.global_CS_id] = {}
                self.global_stat["CS"][self.global_CS_id]["N"] = details["size"]
                self.global_stat["CS"][self.global_CS_id]["SUM"] = np.zeros(len(rs_data[0]))
                self.global_stat["CS"][self.global_CS_id]["SUMSQ"] = np.zeros(len(rs_data[0]))
                for p in details["position"]:
                    self.global_stat["CS"][self.global_CS_id]["SUM"] += np.array(rs_data[p])
                    self.global_stat["CS"][self.global_CS_id]["SUMSQ"] += np.array(rs_data[p]) ** 2
                self.global_stat["CS"][self.global_CS_id]["point_index"] = details["index"]

        # index for outlier
        self.global_stat["RS"]["index"] = [cluster["index"][0] for cluster in count_dict.values() if cluster["size"] == 1]
        # data for outlier
        self.global_stat["RS"]["data"] = [rs_data[cluster["position"][0]] for cluster in count_dict.values() if cluster["size"] == 1]

    def initial_DS(self, rest_data, rest_index):
        '''
        :param data:
        :param rest_index:
        '''
        kmeans = KMeans(n_clusters=self.k, random_state=0, n_jobs=-1).fit(rest_data)
        rest_labels = kmeans.labels_
        for i, label in enumerate(rest_labels):
            if label not in self.global_stat["DS"]:
                self.global_stat["DS"][label] = {}
                self.global_stat["DS"][label]["N"] = 1
                self.global_stat["DS"][label]["SUM"] = np.array(rest_data[i])
                self.global_stat["DS"][label]["SUMSQ"] = np.array(rest_data[i]) ** 2
                self.global_stat["DS"][label]["point_index"] = [rest_index[i]]
            else:
                self.global_stat["DS"][label]["N"] += 1
                self.global_stat["DS"][label]["SUM"] += np.array(rest_data[i])
                self.global_stat["DS"][label]["SUMSQ"] += np.array(rest_data[i]) ** 2
                self.global_stat["DS"][label]["point_index"].append(rest_index[i])

    def initial_RS(self, data, index):
        '''

        :param data: a list of data (first 20%)
        :param index: a list of corresponding index
        :return: index of points that are not outliers, index of outliers
        '''
        kmeans = KMeans(n_clusters=self.initial_k, random_state=0, n_jobs=-1).fit(data)
        labels = kmeans.labels_
        count_dict = {}

        # count for finding outliers
        for label, i, pos in list(zip(labels, index, list(range(len(data))))):
            if label not in count_dict:
                count_dict[label] = {"index": [i], "size": 1, "position": [pos]}
            else:
                count_dict[label]["index"].append(i)
                count_dict[label]["position"].append(pos)
                count_dict[label]["size"] += 1

        # outliers
        outliers_index = [cluster["index"][0] for cluster in count_dict.values() if cluster["size"] == 1]
        outliers_data = [data[cluster["position"][0]] for cluster in count_dict.values() if cluster["size"] == 1]
        # groups
        groups_index = [each for cluster in count_dict.values() if cluster["size"] > 1 for each in cluster["index"]]
        groups_data = [data[each] for cluster in count_dict.values() if cluster["size"] > 1 for each in cluster["position"]]
        return groups_index, groups_data, outliers_index, outliers_data

    def initial_DS_RS(self, data, index):
        rest_index, rest_data, RS_index, RS_data = self.initial_RS(data, index)
        self.initial_DS(rest_data, rest_index)
        self.initial_CS(RS_data, RS_index)

    def printStat(self):
        numOfDSpoints = 0
        numOfCSgroup = 0
        numOfCSpoints = 0
        numOfRSpoints = 0
        for cluster in self.global_stat["DS"].values():
            numOfDSpoints += cluster["N"]
        numOfCSgroup = len(self.global_stat["CS"])
        for group in self.global_stat["CS"].values():
            numOfCSpoints += group["N"]
        numOfRSpoints = len(self.global_stat["RS"]["index"])
        return str(numOfDSpoints) + "," + str(numOfCSgroup) + "," + str(numOfCSpoints) + "," + str(numOfRSpoints)

    def write_result(self):
        result = []
        for key, value in self.global_stat["DS"].items():
            for each in value["point_index"]:
                result.append((each, key))
        for each in self.global_stat["RS"]["index"]:
            result.append((each, -1))
        result = sorted(result, key=lambda x: x[0])
        self.o.write("\nThe clustering results:\n")
        for each in result:
            self.o.write(str(each[0]) + "," + str(each[1]) + "\n")
        self.o.close()

    def evaluation(self, size):
        ground_truth = {}
        with open(self.input_path) as f:
            for line in f.readlines():
                data = line.strip().split(',')[:2]
                if data[1] != '-1':
                    if data[1] not in ground_truth:
                        ground_truth[data[1]] = set(data[0])
                    else:
                        ground_truth[data[1]].add(data[0])
        result = {}
        for key, value in self.global_stat["DS"].items():
            result[key] = set([str(each) for each in value["point_index"]])

        numOfRight = 0
        for r_value in result.values():
            max_intersect = -(sys.maxsize - 1)
            for g_value in ground_truth.values():
                num = len(g_value.intersection(r_value))
                if num > max_intersect:
                    max_intersect = num
            numOfRight += max_intersect
        acc = float(numOfRight) / size
        print("Accuracy is: ", acc)

    def run(self):
        startTime = time.time()
        input_data = self.readfile()
        input_size = len(input_data)
        self.distance_threshold = 2 * sqrt(len(input_data[0]))
        input_data_index = [i for i in range(input_size)]
        shuffled_data, shuffled_index = shuffle(input_data, input_data_index, random_state=0)

        start = int(input_size * self.sample_size)
        end = start + int(input_size * self.sample_size)

        self.numOfRound += 1
        self.initial_DS_RS(shuffled_data[:start], shuffled_index[:start])
        print("The intermediate results:")
        print("Round " + str(self.numOfRound) + ": " + self.printStat())
        self.o.write("The intermediate results: \n")
        self.o.write("Round " + str(self.numOfRound) + ": " + self.printStat() + '\n')

        while start < input_size:
            self.numOfRound += 1
            self.main(shuffled_data[start:end], shuffled_index[start:end])

            start = end
            end = start + int(input_size * self.sample_size)

            if self.numOfRound == self.totalRounds - 1:
                end = input_size

            if self.numOfRound == self.totalRounds:
                self.finish()
            print("Round " + str(self.numOfRound) + ": " + self.printStat())
            self.o.write("Round " + str(self.numOfRound) + ": " + self.printStat() + '\n')

        self.write_result()
        # self.evaluation(input_size)
        print("Finish Time:", time.time() - startTime)


if __name__ == '__main__':
    #  for debug
    # k = 10
    # input_path = 'hw5_clustering.txt'
    # output_path = "result.txt"

    if len(sys.argv) != 4:
        print(sys.stderr, "<input_file_name> <n_cluster> <output_file_name>")
        exit(-1)
    k = int(sys.argv[2])
    input_path = sys.argv[1]
    output_path = sys.argv[3]

    bfr = Bradley_Fayyad_Reina(k, input_path, output_path)
    bfr.run()
