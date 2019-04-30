from LSH import LocalSensitityHashing
from pyspark.mllib.recommendation import ALS, Rating
from pyspark import SparkContext, SparkConf
import time
from sys import *

class ALS_CFRecommendationSystem:
    def __init__(self, train_file_path, test_file_path, output_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.output_Path = output_path

        self.stringUserToIndex_train = {}
        self.indexToStringUser_train = {}
        self.stringItemToIndex_train = {}
        self.indexToStringItem_train = {}

        self.stringUserToIndex_test = {}
        self.indexToStringUser_test = {}
        self.stringItemToIndex_test = {}
        self.indexToStringItem_test = {}

        self.result = []

    def replaceIndex_test(self, indexDicTrain, indexDicTest, usersOrItems):
        i = max(indexDicTrain.values()) + 1
        for userOrItem in usersOrItems:
            if not indexDicTrain.__contains__(userOrItem):
                indexDicTest[userOrItem] = i
                i += 1
            else:
                indexDicTest[userOrItem] = indexDicTrain[userOrItem]

    def readAndSplit(sefl, line):
        splited_line = line.strip().split(',')
        return (splited_line[0], splited_line[1], splited_line[2])

    def replaceIndex_train(self, indexDic, usersOrItems):
        i = 0
        for userOrItem in usersOrItems:
            indexDic[userOrItem] = i
            i += 1

    def cutPred(self, x):
        if x > 5:
            rate = 5
        elif x < 1:
            rate = 1
        else:
            rate = x
        return rate

    def write(self):
        with open(self.output_Path,'w') as f:
            f.write("user_id, business_id, prediction\n")
            for pair in self.result:
                f.write(self.indexToStringUser_test[pair[0][0]] + "," + self.indexToStringItem_test[pair[0][1]] + "," + str(pair[1]) + "\n")

    def run(self):
        startTime = time.time()
        conf = SparkConf()\
                        .setAppName("Model_based_CF_recommendation_system")\
                        .set("spark.executor.memory", "4g")\
                        .set("spark.driver.host", "localhost")
        sc = SparkContext(conf=conf)

        #  train part
        inputData1 = sc.textFile(self.train_file_path)

        # Drop the header
        header1 = inputData1.first()
        inputData1 = inputData1.filter(lambda line: line != header1)

        # read and split data into tuples
        train_RDD = inputData1.map(self.readAndSplit)

        # create user index
        UniqueUserTrain = train_RDD.map(lambda line: line[0]).distinct().collect()
        self.replaceIndex_train(self.stringUserToIndex_train, UniqueUserTrain)
        self.indexToStringUser_train = {value:key for key, value in self.stringUserToIndex_train.items()}

        # create item index {"dlksa":1} and also create its reversed version {1:"dlksa"}
        UniqueItemTrain = train_RDD.map(lambda line: line[1]).distinct().collect()
        self.replaceIndex_train(self.stringItemToIndex_train, UniqueItemTrain)
        self.indexToStringItem_train = {value: key for key, value in self.stringItemToIndex_train.items()}

        ratings_train = train_RDD.map(lambda l: Rating(self.stringUserToIndex_train[l[0]], self.stringItemToIndex_train[l[1]], float(l[2])))

        inputData2 = sc.textFile(self.test_file_path)

        # Drop the header
        header2 = inputData2.first()
        inputData2 = inputData2.filter(lambda line: line != header2)

        # read and split data into tuples
        test_RDD = inputData2.map(self.readAndSplit)

        # create user index
        UniqueUserTest = test_RDD.map(lambda line: line[0]).distinct().collect()
        self.replaceIndex_test(self.stringUserToIndex_train, self.stringUserToIndex_test, UniqueUserTest)
        self.indexToStringUser_test = {value: key for key, value in self.stringUserToIndex_test.items()}

        # create item index {"dlksa":1} and also create its reversed version {1:"dlksa"}
        UniqueItemTest = test_RDD.map(lambda line: line[1]).distinct().collect()
        self.replaceIndex_test(self.stringItemToIndex_train, self.stringItemToIndex_test, UniqueItemTest)
        self.indexToStringItem_test = {value: key for key, value in self.stringItemToIndex_test.items()}

        ratings_test = test_RDD.map(lambda l: Rating(self.stringUserToIndex_test[l[0]], self.stringItemToIndex_test[l[1]], float(l[2])))

        # Build the recommendation model using Alternating Least Squares
        model = ALS.train(ratings_train, rank=1, iterations=10, lambda_=0.1, nonnegative=True)

        # Predict for testing set
        testdata_whole = ratings_test.map(lambda p: (p[0], p[1]))
        testdata_com = testdata_whole.filter(lambda p: self.indexToStringUser_train.__contains__(p[0]) and self.indexToStringItem_train.__contains__(p[1]))
        predict_new = testdata_whole.filter(lambda p: not(self.indexToStringUser_train.__contains__(p[0]) and self.indexToStringItem_train.__contains__(p[1]))).map(lambda x: ((x[0], x[1]), 3))
        predictions = model.predictAll(testdata_com).map(lambda r: ((r[0], r[1]), self.cutPred(r[2]))).union(predict_new)
        self.result = predictions.collect()


        # Evaluation for prediction
        ratesAndPreds = ratings_test.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
        MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        RMSE = pow(MSE, 0.5)
        print("RMSE =", RMSE)
        print("Finish Time:", time.time() - startTime)
        sc.stop()


class UserBased_CFRecommendationSystem:
    def __init__(self, train_file_path, test_file_path, output_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.output_Path = output_path

        self.stringUserToIndex_train = {}
        self.indexToStringUser_train = {}
        self.stringItemToIndex_train = {}
        self.indexToStringItem_train = {}

        self.stringUserToIndex_test = {}
        self.indexToStringUser_test = {}
        self.stringItemToIndex_test = {}
        self.indexToStringItem_test = {}

        self.userToItemList = {}
        self.itemToUserList = {}
        self.result = []

    def readAndSplit(sefl, line):
        splited_line = line.strip().split(',')
        return (splited_line[0], splited_line[1], splited_line[2])

    def replaceIndex_train(self, indexDic, usersOrItems):
        i = 0
        for userOrItem in usersOrItems:
            indexDic[userOrItem] = i
            i += 1

    def replaceIndex_test(self, indexDicTrain, indexDicTest, usersOrItems):
        i = max(indexDicTrain.values()) + 1
        for userOrItem in usersOrItems:
            if not indexDicTrain.__contains__(userOrItem):
                indexDicTest[userOrItem] = i
                i += 1
            else:
                indexDicTest[userOrItem] = indexDicTrain[userOrItem]

    def cutPred(self, x):
        if x > 5:
            rate = 5
        elif x < 1:
            rate = 1
        else:
            rate = x
        return rate

    def generateUserToItemList(self, lines):
        for line in lines:
            self.userToItemList[line[0]] = {}
            self.userToItemList[line[0]]["num"] = 0
            self.userToItemList[line[0]]["sum"] = 0
            self.userToItemList[line[0]]["Items"] = {}
            self.userToItemList[line[0]]["itemList"] = set()
            for item, rate in line[1]:
                self.userToItemList[line[0]]["Items"][item] = rate
                self.userToItemList[line[0]]["itemList"].add(item)
                self.userToItemList[line[0]]["num"] += 1
                self.userToItemList[line[0]]["sum"] += rate

    def generateItemToUserList(self, lines):
        for line in lines:
            self.itemToUserList[line[0]] = {}
            for user, rate in line[1]:
                self.itemToUserList[line[0]][user] = rate

    def weights(self, user, similarUser):
        co_rate_items = self.userToItemList[user]["itemList"].intersection(self.userToItemList[similarUser]["itemList"])
        co_rate_size = len(co_rate_items)
        avg_user = 0
        avg_similarUser = 0
        w = 0
        if co_rate_size > 0:
            for each in list(co_rate_items):
                avg_user += self.userToItemList[user]["Items"][each]
                avg_similarUser += self.userToItemList[similarUser]["Items"][each]
            avg_user = float(avg_user) / co_rate_size
            avg_similarUser = float(avg_similarUser) / co_rate_size
            numerator = 0
            r_u_2 = 0
            r_su_2 = 0
            for each in list(co_rate_items):
                numerator += (self.userToItemList[user]["Items"][each] - avg_user) * (self.userToItemList[similarUser]["Items"][each] - avg_similarUser)
                r_u_2 += pow(self.userToItemList[user]["Items"][each] - avg_user, 2)
                r_su_2 += pow(self.userToItemList[similarUser]["Items"][each] - avg_similarUser, 2)
            if r_u_2 != 0 and r_su_2 != 0:
                w = float(numerator) / (pow(r_u_2, 0.5) * pow(r_su_2, 0.5))
        return w

    def prediction(self, line):
        rate = float(self.userToItemList[line[0]]["sum"]) / self.userToItemList[line[0]]["num"]
        numerator = 0
        sum_w = 0
        for userCoRate in self.itemToUserList[line[1]].keys():
            if userCoRate != line[0]:
                w = self.weights(line[0], userCoRate)
                sum_w += abs(w)
                userCoRate_avg = float(self.userToItemList[userCoRate]["sum"]) / self.userToItemList[userCoRate]["num"]
                numerator += (self.userToItemList[userCoRate]["Items"][line[1]] - userCoRate_avg) * w
        if sum_w != 0:
             rate = self.cutPred(rate + float(numerator) / sum_w)
        return ((line[0], line[1]), rate)

    def write(self):
        with open(self.output_Path,'w') as f:
            f.write("user_id, business_id, prediction\n")
            for pair in self.result:
                f.write(self.indexToStringUser_test[pair[0][0]] + "," + self.indexToStringItem_test[pair[0][1]] + "," + str(pair[1]) + "\n")

    def run(self):
        startTime = time.time()
        conf = SparkConf() \
            .setAppName("User_based_CF_recommendation_system") \
            .set("spark.executor.memory", "4g") \
            .set("spark.driver.host", "localhost")
        sc = SparkContext(conf=conf)

        #  train part
        inputData1 = sc.textFile(self.train_file_path)

        # Drop the header
        header1 = inputData1.first()
        inputData1 = inputData1.filter(lambda line: line != header1)

        # read and split data into tuples
        train_RDD = inputData1.map(self.readAndSplit)

        # create user index
        UniqueUserTrain = train_RDD.map(lambda line: line[0]).distinct().collect()
        self.replaceIndex_train(self.stringUserToIndex_train, UniqueUserTrain)
        self.indexToStringUser_train = {value: key for key, value in self.stringUserToIndex_train.items()}

        # create item index {"dlksa":1} and also create its reversed version {1:"dlksa"}
        UniqueItemTrain = train_RDD.map(lambda line: line[1]).distinct().collect()
        self.replaceIndex_train(self.stringItemToIndex_train, UniqueItemTrain)
        self.indexToStringItem_train = {value: key for key, value in self.stringItemToIndex_train.items()}

        inputData2 = sc.textFile(self.test_file_path)

        # Drop the header
        header2 = inputData2.first()
        inputData2 = inputData2.filter(lambda line: line != header2)

        # read and split data into tuples
        test_RDD = inputData2.map(self.readAndSplit)

        # create user index
        UniqueUserTest = test_RDD.map(lambda line: line[0]).distinct().collect()
        self.replaceIndex_test(self.stringUserToIndex_train, self.stringUserToIndex_test, UniqueUserTest)
        self.indexToStringUser_test = {value: key for key, value in self.stringUserToIndex_test.items()}

        # create item index {"dlksa":1} and also create its reversed version {1:"dlksa"}
        UniqueItemTest = test_RDD.map(lambda line: line[1]).distinct().collect()
        self.replaceIndex_test(self.stringItemToIndex_train, self.stringItemToIndex_test, UniqueItemTest)
        self.indexToStringItem_test = {value: key for key, value in self.stringItemToIndex_test.items()}

        ratings_train = train_RDD.map(lambda l: (self.stringUserToIndex_train[l[0]], self.stringItemToIndex_train[l[1]],float(l[2])))
        ratings_test = test_RDD.map(lambda l: Rating(self.stringUserToIndex_test[l[0]], self.stringItemToIndex_test[l[1]], float(l[2])))

        userToItem = ratings_train.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().collect()
        self.generateUserToItemList(userToItem)
        itemToUser = ratings_train.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().collect()
        self.generateItemToUserList(itemToUser)

        testdata_whole = ratings_test.map(lambda p: (p[0], p[1]))
        testdata_com = testdata_whole.filter(lambda p: self.indexToStringUser_train.__contains__(p[0]) and self.indexToStringItem_train.__contains__(p[1]))
        predict_new = testdata_whole.filter(lambda p: not(self.indexToStringUser_train.__contains__(p[0])) or not(self.indexToStringItem_train.__contains__(p[1]))).map(lambda x: ((x[0], x[1]), 3.5))

        predictions = testdata_com.map(self.prediction).union(predict_new)
        self.result = predictions.collect()

        # Evaluation for prediction
        ratesAndPreds = ratings_test.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
        MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
        RMSE = pow(MSE, 0.5)
        print("RMSE =", RMSE)
        print("Finish Time:", time.time() - startTime)
        sc.stop()
        self.write()


class ItemBased_CFRecommendationSystem:
    def __init__(self, train_file_path, test_file_path, output_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.output_Path = output_path

        self.stringUserToIndex_train = {}
        self.indexToStringUser_train = {}
        self.stringItemToIndex_train = {}
        self.indexToStringItem_train = {}

        self.stringUserToIndex_test = {}
        self.indexToStringUser_test = {}
        self.stringItemToIndex_test = {}
        self.indexToStringItem_test = {}

        self.userToItemList = {}
        self.itemToUserList = {}
        self.result = []

    def readAndSplit(sefl, line):
        splited_line = line.strip().split(',')
        return (splited_line[0], splited_line[1], splited_line[2])

    def replaceIndex_train(self, indexDic, usersOrItems):
        i = 0
        for userOrItem in usersOrItems:
            indexDic[userOrItem] = i
            i += 1

    def replaceIndex_test(self, indexDicTrain, indexDicTest, usersOrItems):
        i = max(indexDicTrain.values()) + 1
        for userOrItem in usersOrItems:
            if not indexDicTrain.__contains__(userOrItem):
                indexDicTest[userOrItem] = i
                i += 1
            else:
                indexDicTest[userOrItem] = indexDicTrain[userOrItem]

    def cutPred(self, x):
        if x > 5:
            rate = 5
        elif x < 1:
            rate = 1
        else:
            rate = x
        return rate

    def generateUserToItemList(self, lines):
        '''
        :param lines: (user_id, [(item_id1, rate1), (item_id3, rate3)])
        :return: self.userToItemList: {user_id: {item_id1: rate1, item_id3: rate3}}
        '''
        for line in lines:
            self.userToItemList[line[0]] = {}
            for item, rate in line[1]:
                self.userToItemList[line[0]][item] = rate

    def generateItemToUserList(self, lines):
        '''
        :param lines: (item_id, [(user_id1, rate1), (user_id3, rate3)])
        :return: self.itemToUserList: {item_id1:
                                                {users:{user_id1: rate1, user_id3: rate3},
                                                {userList:set(user_id1, user_id3)},
                                                {sum: 6511},
                                                {num: 77}}, }
        '''
        for line in lines:
            self.itemToUserList[line[0]] = {}
            self.itemToUserList[line[0]]["num"] = 0
            self.itemToUserList[line[0]]["sum"] = 0
            self.itemToUserList[line[0]]["users"] = {}
            self.itemToUserList[line[0]]["userList"] = set()
            for user, rate in line[1]:
                self.itemToUserList[line[0]]["users"][user] = rate
                self.itemToUserList[line[0]]["userList"].add(user)
                self.itemToUserList[line[0]]["num"] += 1
                self.itemToUserList[line[0]]["sum"] += rate

    def weights(self, item, comparedItem):
        '''

        :param item: target item
        :param comparedItem: another item that is also rated by target user
        :return: weight between them
        '''
        co_rate_users = self.itemToUserList[item]["userList"].intersection(self.itemToUserList[comparedItem]["userList"])

        avg_item = float(self.itemToUserList[item]["sum"]) / self.itemToUserList[item]["num"]
        avg_comparedItem = float(self.itemToUserList[comparedItem]["sum"]) / self.itemToUserList[comparedItem]["num"]

        w = 0
        numerator = 0
        r_u_2 = 0
        r_su_2 = 0

        for each in list(co_rate_users):
            numerator += (self.itemToUserList[item]["users"][each] - avg_item) * (self.itemToUserList[comparedItem]["users"][each] - avg_comparedItem)
            r_u_2 += pow(self.itemToUserList[item]["users"][each] - avg_item, 2)
            r_su_2 += pow(self.itemToUserList[comparedItem]["users"][each] - avg_comparedItem, 2)

        #  check if either part in the denominator is zero
        if r_u_2 != 0 and r_su_2 != 0:
            w = float(numerator) / (pow(r_u_2, 0.5) * pow(r_su_2, 0.5))
        return w

    def prediction(self, line):
        '''

        :param line: (target_user_id, target_item_id)
        :return: ((target_user_id, target_item_id), predicted_rate)
        '''
        # initial rate based on the avg rate for this item
        rate = float(self.itemToUserList[line[1]]["sum"]) / self.itemToUserList[line[1]]["num"]
        numerator = 0
        sum_w = 0

        # for each items that this user has rated
        for ratedItem in self.userToItemList[line[0]].keys():
            if ratedItem != line[1]:
                w = self.weights(line[1], ratedItem)
                if w > 0.1:
                    sum_w += abs(w)
                    itemCoRate_avg = float(self.itemToUserList[line[1]]["sum"]) / self.itemToUserList[line[1]]["num"]
                    numerator += (self.userToItemList[line[0]][ratedItem] - itemCoRate_avg) * w

        # check if the denominator is zero
        if sum_w > 0:
             rate = self.cutPred(rate + float(numerator) / sum_w)
        return ((line[0], line[1]), rate)

    def write(self):
        with open(self.output_Path,'w') as f:
            f.write("user_id, business_id, prediction\n")
            for pair in self.result:
                f.write(self.indexToStringUser_test[pair[0][0]] + "," + self.indexToStringItem_test[pair[0][1]] + "," + str(pair[1]) + "\n")

    def run(self):
        startTime = time.time()
        conf = SparkConf() \
            .setAppName("Item_based_CF_recommendation_system") \
            .set("spark.executor.memory", "4g") \
            .set("spark.driver.host", "localhost")
        sc = SparkContext(conf=conf)

        #  train set part
        inputData1 = sc.textFile(self.train_file_path)

        # Drop the header
        header1 = inputData1.first()
        inputData1 = inputData1.filter(lambda line: line != header1)

        # read and split data into tuples
        train_RDD = inputData1.map(self.readAndSplit)

        # create user index
        UniqueUserTrain = train_RDD.map(lambda line: line[0]).distinct().collect()
        self.replaceIndex_train(self.stringUserToIndex_train, UniqueUserTrain)
        self.indexToStringUser_train = {value: key for key, value in self.stringUserToIndex_train.items()}

        # create item index {"dlksa":1} and also create its reversed version {1:"dlksa"}
        UniqueItemTrain = train_RDD.map(lambda line: line[1]).distinct().collect()
        self.replaceIndex_train(self.stringItemToIndex_train, UniqueItemTrain)
        self.indexToStringItem_train = {value: key for key, value in self.stringItemToIndex_train.items()}

        ratings_train = train_RDD.map(lambda l: (self.stringUserToIndex_train[l[0]], self.stringItemToIndex_train[l[1]], float(l[2])))

        #  test set part
        inputData2 = sc.textFile(self.test_file_path)

        # Drop the header
        header2 = inputData2.first()
        inputData2 = inputData2.filter(lambda line: line != header2)

        # read and split data into tuples
        test_RDD = inputData2.map(self.readAndSplit)

        # create user index
        UniqueUserTest = test_RDD.map(lambda line: line[0]).distinct().collect()
        self.replaceIndex_test(self.stringUserToIndex_train, self.stringUserToIndex_test, UniqueUserTest)
        self.indexToStringUser_test = {value: key for key, value in self.stringUserToIndex_test.items()}

        # create item index {"dlksa":1} and also create its reversed version {1:"dlksa"}
        UniqueItemTest = test_RDD.map(lambda line: line[1]).distinct().collect()
        self.replaceIndex_test(self.stringItemToIndex_train, self.stringItemToIndex_test, UniqueItemTest)
        self.indexToStringItem_test = {value: key for key, value in self.stringItemToIndex_test.items()}

        ratings_test = test_RDD.map(lambda l: (self.stringUserToIndex_test[l[0]], self.stringItemToIndex_test[l[1]], float(l[2])))

        userToItem = ratings_train.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().collect()
        self.generateUserToItemList(userToItem)
        itemToUser = ratings_train.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().collect()
        self.generateItemToUserList(itemToUser)

        testdata_whole = ratings_test.map(lambda p: (p[0], p[1]))
        testdata_com = testdata_whole.filter(lambda p: self.indexToStringUser_train.__contains__(p[0]) and self.indexToStringItem_train.__contains__(p[1]))
        predict_new = testdata_whole.filter(lambda p: not(self.indexToStringUser_train.__contains__(p[0])) or not(self.indexToStringItem_train.__contains__(p[1]))).map(lambda x: ((x[0], x[1]), 3))

        predictions = testdata_com.map(self.prediction).union(predict_new)
        self.result = predictions.collect()

        # Evaluation for prediction
        ratesAndPreds = ratings_test.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
        MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
        RMSE = pow(MSE, 0.5)
        print("RMSE =", RMSE)
        print("Finish Time:", time.time() - startTime)
        sc.stop()
        self.write()


class ItemBased_CFRecommendationSystem_LSH:
    def __init__(self, train_file_path, test_file_path, output_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.output_Path = output_path
        self.LSH_pair = {}

        self.stringUserToIndex_train = {}
        self.indexToStringUser_train = {}
        self.stringItemToIndex_train = {}
        self.indexToStringItem_train = {}

        self.stringUserToIndex_test = {}
        self.indexToStringUser_test = {}
        self.stringItemToIndex_test = {}
        self.indexToStringItem_test = {}

        self.userToItemList = {}
        self.itemToUserList = {}
        self.result = []

    def runLSH(self):
        numOfBands = 12
        numOfRows = 2
        threshold = 0.3

        lsh = LocalSensitityHashing(numOfBands, numOfRows, threshold, self.train_file_path)
        lsh.run()
        for pair in lsh.result:
            self.LSH_pair[pair[0][0] + ',' + pair[0][1]] = pair[1]

    def readAndSplit(sefl, line):
        splited_line = line.strip().split(',')
        return (splited_line[0], splited_line[1], splited_line[2])

    def replaceIndex_train(self, indexDic, usersOrItems):
        i = 0
        for userOrItem in usersOrItems:
            indexDic[userOrItem] = i
            i += 1

    def replaceIndex_test(self, indexDicTrain, indexDicTest, usersOrItems):
        i = max(indexDicTrain.values()) + 1
        for userOrItem in usersOrItems:
            if not indexDicTrain.__contains__(userOrItem):
                indexDicTest[userOrItem] = i
                i += 1
            else:
                indexDicTest[userOrItem] = indexDicTrain[userOrItem]

    def cutPred(self, x):
        if x > 5:
            rate = 5
        elif x < 1:
            rate = 1
        else:
            rate = x
        return rate

    def generateUserToItemList(self, lines):
        '''
        :param lines: (user_id, [(item_id1, rate1), (item_id3, rate3)])
        :return: self.userToItemList: {user_id: {item_id1: rate1, item_id3: rate3}}
        '''
        for line in lines:
            self.userToItemList[line[0]] = {}
            for item, rate in line[1]:
                self.userToItemList[line[0]][item] = rate

    def generateItemToUserList(self, lines):
        '''
        :param lines: (item_id, [(user_id1, rate1), (user_id3, rate3)])
        :return: self.itemToUserList: {item_id1:
                                                {users:{user_id1: rate1, user_id3: rate3},
                                                {userList:set(user_id1, user_id3)},
                                                {sum: 6511},
                                                {num: 77}}, }
        '''
        for line in lines:
            self.itemToUserList[line[0]] = {}
            self.itemToUserList[line[0]]["num"] = 0
            self.itemToUserList[line[0]]["sum"] = 0
            self.itemToUserList[line[0]]["users"] = {}
            self.itemToUserList[line[0]]["userList"] = set()
            for user, rate in line[1]:
                self.itemToUserList[line[0]]["users"][user] = rate
                self.itemToUserList[line[0]]["userList"].add(user)
                self.itemToUserList[line[0]]["num"] += 1
                self.itemToUserList[line[0]]["sum"] += rate

    def weights(self, item, comparedItem):
        '''

        :param item: target item
        :param comparedItem: another item that is also rated by target user
        :return: weight between them
        '''
        co_rate_users = self.itemToUserList[item]["userList"].intersection(self.itemToUserList[comparedItem]["userList"])

        avg_item = float(self.itemToUserList[item]["sum"]) / self.itemToUserList[item]["num"]
        avg_comparedItem = float(self.itemToUserList[comparedItem]["sum"]) / self.itemToUserList[comparedItem]["num"]

        w = 0
        numerator = 0
        r_u_2 = 0
        r_su_2 = 0

        for each in list(co_rate_users):
            numerator += (self.itemToUserList[item]["users"][each] - avg_item) * (self.itemToUserList[comparedItem]["users"][each] - avg_comparedItem)
            r_u_2 += pow(self.itemToUserList[item]["users"][each] - avg_item, 2)
            r_su_2 += pow(self.itemToUserList[comparedItem]["users"][each] - avg_comparedItem, 2)

        #  check if either part in the denominator is zero
        if r_u_2 != 0 and r_su_2 != 0:
            w = float(numerator) / (pow(r_u_2, 0.5) * pow(r_su_2, 0.5))
        return w

    def prediction(self, line):
        '''

        :param line: (target_user_id, target_item_id)
        :return: ((target_user_id, target_item_id), predicted_rate)
        '''
        # initial rate based on the avg rate for this item
        rate = float(self.itemToUserList[line[1]]["sum"]) / self.itemToUserList[line[1]]["num"]
        numerator = 0
        sum_w = 0

        # for each items that this user has rated
        for ratedItem in self.userToItemList[line[0]].keys():
            if ratedItem != line[1]:
                sorted_key = self.indexToStringItem_train[line[1]] + ',' + self.indexToStringItem_train[ratedItem] \
                    if self.indexToStringItem_train[line[1]] < self.indexToStringItem_train[ratedItem] \
                    else self.indexToStringItem_train[ratedItem] + ',' + self.indexToStringItem_train[line[1]]
                if self.LSH_pair.__contains__(sorted_key):
                    similarity = self.LSH_pair[sorted_key]
                    w = self.weights(line[1], ratedItem)
                    sum_w += abs(similarity * w)
                    itemCoRate_avg = float(self.itemToUserList[line[1]]["sum"]) / self.itemToUserList[line[1]]["num"]
                    numerator += (self.userToItemList[line[0]][ratedItem] - itemCoRate_avg) * similarity * w

        # check if the denominator is zero
        if sum_w > 0:
             rate = self.cutPred(rate + float(numerator) / sum_w)
        return ((line[0], line[1]), rate)

    def write(self):
        with open(self.output_Path,'w') as f:
            f.write("user_id, business_id, prediction\n")
            for pair in self.result:
                f.write(self.indexToStringUser_test[pair[0][0]] + "," + self.indexToStringItem_test[pair[0][1]] + "," + str(pair[1]) + "\n")

    def run(self):
        startTime = time.time()
        conf = SparkConf() \
            .setAppName("Item_based_CF_recommendation_system") \
            .set("spark.executor.memory", "4g") \
            .set("spark.driver.host", "localhost")
        sc = SparkContext(conf=conf)

        #  train set part
        inputData1 = sc.textFile(self.train_file_path)

        # Drop the header
        header1 = inputData1.first()
        inputData1 = inputData1.filter(lambda line: line != header1)

        # read and split data into tuples
        train_RDD = inputData1.map(self.readAndSplit)

        # create user index
        UniqueUserTrain = train_RDD.map(lambda line: line[0]).distinct().collect()
        self.replaceIndex_train(self.stringUserToIndex_train, UniqueUserTrain)
        self.indexToStringUser_train = {value: key for key, value in self.stringUserToIndex_train.items()}

        # create item index {"dlksa":1} and also create its reversed version {1:"dlksa"}
        UniqueItemTrain = train_RDD.map(lambda line: line[1]).distinct().collect()
        self.replaceIndex_train(self.stringItemToIndex_train, UniqueItemTrain)
        self.indexToStringItem_train = {value: key for key, value in self.stringItemToIndex_train.items()}

        ratings_train = train_RDD.map(lambda l: (self.stringUserToIndex_train[l[0]], self.stringItemToIndex_train[l[1]], float(l[2])))

        #  test set part
        inputData2 = sc.textFile(self.test_file_path)

        # Drop the header
        header2 = inputData2.first()
        inputData2 = inputData2.filter(lambda line: line != header2)

        # read and split data into tuples
        test_RDD = inputData2.map(self.readAndSplit)

        # create user index
        UniqueUserTest = test_RDD.map(lambda line: line[0]).distinct().collect()
        self.replaceIndex_test(self.stringUserToIndex_train, self.stringUserToIndex_test, UniqueUserTest)
        self.indexToStringUser_test = {value: key for key, value in self.stringUserToIndex_test.items()}

        # create item index {"dlksa":1} and also create its reversed version {1:"dlksa"}
        UniqueItemTest = test_RDD.map(lambda line: line[1]).distinct().collect()
        self.replaceIndex_test(self.stringItemToIndex_train, self.stringItemToIndex_test, UniqueItemTest)
        self.indexToStringItem_test = {value: key for key, value in self.stringItemToIndex_test.items()}

        ratings_test = test_RDD.map(lambda l: (self.stringUserToIndex_test[l[0]], self.stringItemToIndex_test[l[1]], float(l[2])))

        userToItem = ratings_train.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().collect()
        self.generateUserToItemList(userToItem)
        itemToUser = ratings_train.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().collect()
        self.generateItemToUserList(itemToUser)

        testdata_whole = ratings_test.map(lambda p: (p[0], p[1]))
        testdata_com = testdata_whole.filter(lambda p: self.indexToStringUser_train.__contains__(p[0]) and self.indexToStringItem_train.__contains__(p[1]))
        predict_new = testdata_whole.filter(lambda p: not(self.indexToStringUser_train.__contains__(p[0])) or not(self.indexToStringItem_train.__contains__(p[1]))).map(lambda x: ((x[0], x[1]), 3))

        predictions = testdata_com.map(self.prediction).union(predict_new)
        self.result = predictions.collect()

        # Evaluation for prediction
        ratesAndPreds = ratings_test.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
        MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
        RMSE = pow(MSE, 0.5)
        print("RMSE =", RMSE)
        print("Finish Time:", time.time() - startTime)
        sc.stop()



if __name__ == '__main__':
    # train_path = "yelp_train.csv"
    # test_path = "yelp_val.csv"
    # case = 4
    # output_path = "output_case3.csv"

    if len(argv) != 5:
        print(stderr, "<train_file_name> <test_file_name> <case_id> <output_file_name>")
        exit(-1)

    train_path = argv[1]
    test_path = argv[2]
    case = int(argv[3])
    output_path = argv[4]

    if case == 1:
        ALSRS = ALS_CFRecommendationSystem(train_path, test_path, output_path)
        ALSRS.run()
        ALSRS.write()
    elif case == 2:
        UBRS = UserBased_CFRecommendationSystem(train_path, test_path, output_path)
        UBRS.run()
        UBRS.write()
    elif case == 3:
        IBRS = ItemBased_CFRecommendationSystem(train_path, test_path, output_path)
        IBRS.run()
        IBRS.write()
    elif case == 4:
        IBRSLSH = ItemBased_CFRecommendationSystem_LSH(train_path, test_path, output_path)
        IBRSLSH.run()
        IBRSLSH.write()
