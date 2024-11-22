from get_instance import DAG
import numpy as np
import copy
import random

class Probability_model(object):
    def __init__(self, DAG, machine_num, voltage_relative_speed_pair = {1.0:1.75, 0.8:1.4, 0.6:1.2, 0.4:0.9},
                 learn_rate=0.1):
        self.DAG = DAG
        self.machines = [i for i in range(machine_num)]
        self.voltage_relative_speed_pair = voltage_relative_speed_pair
        self.learn_rate = learn_rate

        self.model_0 = np.zeros(( len(self.DAG.nodes.keys())+1, len(self.DAG.nodes.keys()) ))
        self.model_0[1:] += 0.5
        model_part = self.model_0[1:]
        for i in self.DAG.edge_list:
            model_part[int(i[0]),int(i[1])] = 1
            model_part[int(i[1]),int(i[0])] = 0
        for i in range(len(model_part)):
            model_part[i,i] = 0
        self.model_0[0] = 1

        self.model_1 = np.zeros( ( len(self.DAG.nodes.keys()), len(self.machines)) )
        self.model_1 += 1.0/len(self.machines)

        self.model_2 = np.zeros( ( len(self.DAG.nodes.keys()), len(self.voltage_relative_speed_pair.keys())) )
        self.model_2 += 1.0/len(self.voltage_relative_speed_pair.keys())

        self.model_3 = np.zeros( ( len(self.DAG.nodes.keys()), len(self.machines)) )
        self.model_3 += 1.0/len(self.machines)

    def sampling(self):
        permutation = self.sampling_0()
        processed_location = self.sampling_1(permutation)
        speed_list = self.sampling_2(permutation)
        backup_list = self.sampling_3(permutation, processed_location)

        return [permutation, processed_location, speed_list, backup_list]

    def update(self, elite_list):
        pass

    def update_3(self, elite_list):
        for elite in elite_list:
            self.model_3[:] = self.model_3[:]*(1-self.learn_rate) + (1.0/len(elite_list)) * self.learn_unit_3(elite) * self.learn_rate

    def learn_unit_3(self, elite):
        unit = np.zeros((len(self.DAG.nodes.keys()), len(self.machines)))#用0去填充nxm的矩阵
        for task, location in zip(elite[0] , elite[3]):
            unit[task, location] = 1#有备份的任务
        return unit

    def sampling_3(self, permutation, processed_location):
        backup_list = []
        for i, j in zip(permutation, processed_location):
            probability_list = copy.deepcopy(self.model_3[i])
            probability_list[j] = 0
            probability_list = self.normaliza(probability_list)
            backup = self.roulette(probability_list)
            backup_list.append(backup)
        return backup_list

    def update_2(self, elite_list):
        for elite in elite_list:
            self.model_2[:] = self.model_2[:]*(1-self.learn_rate) +  (1.0/len(elite_list)) * self.learn_unit_2(elite) * self.learn_rate

    def learn_unit_2(self, elite):
        unit = np.zeros( ( len(self.DAG.nodes.keys()), len(self.voltage_relative_speed_pair.keys())) )
        speed_list = list(self.voltage_relative_speed_pair.keys())
        for task, speed in zip(elite[0], elite[2]):
            unit[task, speed_list.index(speed)] = 1
        return unit

    def sampling_2(self, permutation):
        speed_list = []
        for i in permutation:
            probability_list = copy.deepcopy(self.model_2[i])
            probability_list = self.normaliza(probability_list)
            speed_ind = self.roulette(probability_list)
            speed = list(self.voltage_relative_speed_pair.keys())[speed_ind]
            speed_list.append(speed)
        return speed_list

    def update_1(self, elite_list):
        for elite in elite_list:
            self.model_1[:] = self.model_1[:] * (1-self.learn_rate) + (1.0/len(elite_list)) * self.learn_unit_1(elite) * self.learn_rate

    def learn_unit_1(self, elite):
        unit = np.zeros( ( len(self.DAG.nodes.keys()), len(self.machines)) )
        for task, location in zip(elite[0] , elite[1]):
            unit[task, location] = 1
        return unit

    def sampling_1(self, permutation):
        processed_location = []
        for i in permutation:
            probability_list = copy.deepcopy(self.model_1[i])
            probability_list = self.normaliza(probability_list)
            location = self.roulette(probability_list)
            processed_location.append(location)
        return processed_location

    def update_0(self, elite_list):
        for elite in elite_list:
            self.model_0[:] = self.model_0[:] * (1-self.learn_rate) + (1.0/len(elite_list)) * self.learn_unit_0(elite) * self.learn_rate

    def learn_unit_0(self, elite):
        unit = np.zeros(( len(self.DAG.nodes.keys())+1, len(self.DAG.nodes.keys()) ))
        unit[0,elite[0][0]] = 1
        for i in range(1, len(elite[0])):
            unit[ elite[0][i-1]+1,  elite[0][i]] = 1
        return unit

    def sampling_0(self):
        permutation = []
        while len(permutation) < len(self.DAG.nodes):
            alternative_task = self.check_node(self.DAG.matrix, permutation)
            probability_list = []
            if len(permutation) == 0:
                for ind, i in enumerate(self.model_0[0]):
                    if ind in alternative_task:
                        probability_list.append(i)
                    else:
                        probability_list.append(0)
            else:
                for ind,i in enumerate(self.model_0[permutation[-1]+1]):
                    if ind in alternative_task:
                        probability_list.append(i)
                    else:
                        probability_list.append(0)
            probability_list = self.normaliza(probability_list)
            enter_task = self.roulette(probability_list)
            permutation.append(enter_task)
        return permutation

    def roulette(self, probability_list):
        randomNum = random.random()
        up = 0.0
        for ind, i in enumerate(probability_list):
            down = up
            up += i
            if randomNum > down and up > randomNum:
                return ind
        raise RuntimeError

    def normaliza(self, probability_list):
        probability_list = np.array(probability_list)
        return probability_list/sum(probability_list)

    def check_node(slef, matrix, processed_nodes):
        nodes = []
        for i in range(matrix.shape[1]):
            if i in processed_nodes:
                continue
            mark = True
            for ind, j in enumerate(matrix[:, i] != 0):
                if j == True and ind not in processed_nodes:
                    mark = False
                    break
            if mark:
                nodes.append(i)
        return nodes


