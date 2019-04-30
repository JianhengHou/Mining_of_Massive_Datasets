from pyspark import SparkContext, SparkConf
import sys
import time
import copy
from queue import Queue


class TreeNode:
    def __init__(self, u_id, weight):
        self.user_id = u_id
        self.weight = weight
        self.credit = 0
        self.level = -1
        self.neighbors = set()
        self.parents = set()


class CommunityDetection:
    def __init__(self, threshold, input_path,  betweenness_out_path, community_out_path):
        self.threshold = threshold
        self.input_path = input_path
        self.betweenness_out_path = betweenness_out_path
        self.community_out_path = community_out_path
        self.vistedUser = set()
        self.global_User_items = {}

        # store the graph of user nodes (will change each iteration)
        self.global_user_treeNode = {}
        self.edge_betweenness = []
        self.community_graphs = {}
        self.community_set = {}
        self.community_index = 0
        # 2 * number of edges
        self.double_edges = 0
        self.modularity = []
        self.earlyStopping = 5

    def cal_modularity(self, user_treenode, orginal_treenode):
        global_visited = set()
        community_list = []
        for node in user_treenode.values():
            if not global_visited.__contains__(node.user_id):
                community_nodes = self.bfs(node)
                # print("number of the nodes in this community", len(community_nodes))
                community_list.append(community_nodes)
                global_visited = global_visited.union(community_nodes)
        modularity = 0

        for group in community_list:
            for i in group:
                for j in group:
                    neighbors_id = set([each.user_id for each in user_treenode[i].neighbors])
                    if i != j:
                        if j in neighbors_id:
                            modularity = modularity + (1 - (float(len(orginal_treenode[i].neighbors)\
                                                    * len(orginal_treenode[j].neighbors)) / self.double_edges))
                        else:
                            modularity = modularity - (float(len(orginal_treenode[i].neighbors)\
                                                  * len(orginal_treenode[j].neighbors)) / self.double_edges)
        modularity = modularity / self.double_edges
        return community_list, modularity

    @ staticmethod
    def bfs(root):
        visited = set()
        que = Queue()

        que.put(root)
        visited.add(root.user_id)

        while not que.empty():
            polled_node = que.get()
            for neighbor in polled_node.neighbors:
                if not visited.__contains__(neighbor.user_id):
                    visited.add(neighbor.user_id)
                    que.put(neighbor)
        return visited

    @ staticmethod
    def bfs_weight_assigment(root):
        visited = set()
        level_queue = []
        curr_level = 0
        que = Queue()
        que.put(root)
        visited.add(root.user_id)

        while not que.empty():
            # for each level
            size = que.qsize()
            dummy_queue = []
            curr_level += 1
            # collect all nodes on current level in to the dummy queue
            for index in range(size):
                node = que.get()
                node.level = curr_level
                dummy_queue.append(node)
            level_queue.append(dummy_queue)

            #  for each nodes in on current level
            for each_current in dummy_queue:
                for neighbor in each_current.neighbors:
                    # for neighbors that were not assigned level number
                    if neighbor.level < 0:
                        # set weight
                        neighbor.weight = neighbor.weight + 1
                        # set parents
                        neighbor.parents.add(each_current)

                        que.put(neighbor)
        return level_queue

    @staticmethod
    def cal_edge_credit(queue_list):
        '''

        :param queue_list: a list of each level queue
        :return: a list of edge and its betweenness: [('cyuDrrG5eEK-TZI867MUPA,l-1cva9rA8_ugLrtSdKAqA', 4234.0),...]
        '''
        edge_credit = {}
        size = len(queue_list[::-1])
        for index, each_dummy_queue in enumerate(queue_list[::-1]):
            for node in each_dummy_queue:
                #  initial the credit of each node
                node.credit = 1

                #  to calculate edges weights and update the credit of this node if this node is not leaf now
                if index != 0:
                    sumOfedgeWeights = 0
                    for neighbor in node.neighbors:
                        #  filter to get all children
                        if neighbor.level == node.level + 1:
                            edgeKey = neighbor.user_id + ',' + node.user_id
                            sumOfedgeWeights += edge_credit[edgeKey]
                    #  set the updated weights to this node
                    node.credit = node.credit + sumOfedgeWeights

                # to calculate weights of each edge above this node if this node is not root
                if index != size - 1:
                    sumOfParentWeights = 0
                    #  sum weight of all parents as denominator
                    for parent in node.parents:
                        sumOfParentWeights += parent.weight
                    #  calculate weight of each edge
                    for parent in node.parents:
                        edge_credit[node.user_id + ',' + parent.user_id] = float(parent.weight)/sumOfParentWeights * node.credit

        # store the edges betweenness into a list with lexicographical key
        edge_credit_list = []
        for key, value in edge_credit.items():
            sorted_key = sorted(key.split(','))
            edge_credit_list.append(((sorted_key[0], sorted_key[1]), value))
        return edge_credit_list

    def cal_betweenness(self, root_id, local_user_treeNode):
        TreeNode_dict = copy.deepcopy(local_user_treeNode)
        root = TreeNode_dict[root_id]
        level_queue = self.bfs_weight_assigment(root)
        edge_credit = self.cal_edge_credit(level_queue)

        return edge_credit

    def initialize_treenode(self, lines):
        '''
        initialize neighbors of each node
        :param lines: node
        :return: a list of treenode
        '''
        node_list = []
        for line in lines:
            neighbors_set = set()
            for neigh in line[1]:
                neighbors_set.add(self.global_user_treeNode[neigh])
                self.global_user_treeNode[line[0]].neighbors = neighbors_set
            node_list.append(line[0])
        return node_list

    def generate_treenode(self, lines):
        for line in lines:
            currUserId = line[0]
            node = TreeNode(currUserId, 1)
            self.global_user_treeNode[currUserId] = node

    def generate_edges(self, user_items):
        '''

        :param user_items: (user1_id: set(item1, item3))
        :return: (user1_id, [user3_id, user6_id])
        '''
        currUser = user_items[0]
        currItems = user_items[1]
        neighbors = []
        for otherUser, items in self.global_User_items.items():
            if currUser != otherUser and len(currItems.intersection(items)) >= self.threshold:
                neighbors.append(otherUser)
        return (currUser, neighbors)

    def readAndSplit(self, line):
        splited_line = line.strip().split(',')
        return (splited_line[0], splited_line[1])

    def write_betweenness(self, list):
        with open(self.betweenness_out_path, 'w') as b_out:
            for each in list:
                b_out.write(str(each[0]) + ',' + str(each[1]) + '\n')

    def write_community(self, community_list):
        community_list = list(map(lambda c: sorted(list(c)), community_list))
        community_list.sort(key=lambda c: (len(c), c[0]))
        with open(self.community_out_path, 'w') as c_out:
            for group in community_list:
                c_out.write(str(group)[1:-1] + '\n')

    def run(self):
        startTime = time.time()
        conf = SparkConf() \
            .setAppName("Community_Detection_Based_on_Girvan-Newman_algorithm") \
            .set("spark.executor.memory", "4g")\
            .set("spark.driver.host", "localhost")
        sc = SparkContext(conf=conf)

        inputData = sc.textFile(self.input_path)

        # Drop the header
        header = inputData.first()
        inputData = inputData.filter(lambda line: line != header)

        # read and split data into tuples
        Standard_RDD = inputData.map(self.readAndSplit)

        # format:  user_id1, {item_id1, item_id3} and generate global dictionary storing such key value pairs
        UserAndItems = Standard_RDD.groupByKey().map(lambda x: (x[0], set(list(x[1]))))
        self.global_User_items = UserAndItems.collectAsMap()

        # format: user_id:
        UserAndNeighbors = UserAndItems.map(self.generate_edges).filter(lambda x: len(x[1]) > 0)

        #  UserAndNeighbors_list: [(user1_id, [user3_id, user6_id]), ...]
        UserAndNeighbors_list = UserAndNeighbors.collect()

        self.generate_treenode(UserAndNeighbors_list)

        #  roots_RDD: (User1TreeNode, User2TreeNode, ...)
        roots_RDD = sc.parallelize(self.initialize_treenode(UserAndNeighbors_list), 10)

        self.edge_betweenness = roots_RDD.flatMap(lambda x: self.cal_betweenness(x, self.global_user_treeNode)) \
                                            .reduceByKey(lambda a, b: a + b) \
                                                    .map(lambda x: (x[0], float(x[1]) / 2)) \
                                                        .sortBy(lambda x: (-x[1], x[0][0]))\
                                                                .collect()
        #  write betweenness into the file
        self.write_betweenness(self.edge_betweenness)

        numOfGraphEdges = len(self.edge_betweenness)
        self.double_edges = numOfGraphEdges * 2

        last_numOfvertex = len(self.global_user_treeNode)
        updated_global_treeNode = copy.deepcopy(self.global_user_treeNode)
        # record current community set
        com = []
        max_mol = -sys.maxsize-1
        round_count = 0
        final_community_index = 0
        for index in range(numOfGraphEdges):
            endNode1_id = self.edge_betweenness[0][0][0]
            endNode2_id = self.edge_betweenness[0][0][1]

            updated_global_treeNode[endNode1_id].neighbors.remove(updated_global_treeNode[endNode2_id])
            updated_global_treeNode[endNode2_id].neighbors.remove(updated_global_treeNode[endNode1_id])

            # run bfs to get the number of vertex in the community in which endNode1_id (or endNode1_id) exists
            current_numOfvertex = len(self.bfs(updated_global_treeNode[endNode1_id]))


            # if produce a new community
            if last_numOfvertex > current_numOfvertex:

                new_communities = copy.deepcopy(updated_global_treeNode)
                self.community_index += 1

                #  cal modularity
                com, mol = self.cal_modularity(new_communities, self.global_user_treeNode)
                self.community_set[self.community_index] = com
                self.modularity.append(mol)
                if mol > max_mol:
                    max_mol = mol
                    final_community_index = self.community_index
                else:
                    round_count += 1

            if round_count > self.earlyStopping:
                    break

            #  re calculate betweenness
            updated_root_RDD = sc.parallelize(updated_global_treeNode.keys(), 20)
            self.edge_betweenness = updated_root_RDD.flatMap(lambda x: self.cal_betweenness(x, updated_global_treeNode)) \
                                                        .reduceByKey(lambda a, b: a + b) \
                                                                    .map(lambda x: (x[0], float(x[1]) / 2)) \
                                                                            .sortBy(lambda x: (-x[1], x[0][0])) \
                                                                                    .take(1)

            # assign the number of vertex in the community in which these two
            if len(self.edge_betweenness) > 0:
                waiting_node_id = self.edge_betweenness[0][0][0]
                for community in com:
                    if community.__contains__(waiting_node_id):
                        last_numOfvertex = len(community)
                        break


        self.write_community(self.community_set[final_community_index])
        print("Finish time:", time.time() - startTime)


if __name__ == '__main__':
    # for debug
    # threshold = 7
    # input_path = 'ub_sample_Data.csv'
    # between_output = "task21_out.txt"
    # community_output = "task22_out.txt"

    if len(sys.argv) != 5:
        print(sys.stderr, "<filter threshold> <input_file_path> <betweenness_output_file_path> <community_output_file_path>")
        exit(-1)

    threshold = int(sys.argv[1])
    input_path = sys.argv[2]
    between_output = sys.argv[3]
    community_output = sys.argv[4]

    cd = CommunityDetection(threshold, input_path, between_output ,community_output)
    cd.run()
