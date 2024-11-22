from deap import creator
from deap import base
from deap import tools
import random
import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import product
from probability_model_0 import Probability_model
from get_instance import DAG
from functools import reduce
from deap import tools
import time

class EDA(Probability_model):
    def __init__(self,DAG, machine_num, network_domain, power_domain,
                 voltage_relative_speed_pair = {1:4, 0.8:2, 0.6:1.2, 0.4:0.5}, energy_coeffi=1,
                 alpha1=20, alpha2=20, alpha3=20, alpha4=20, simulation_num = 500, learn_rate=0.3):
        super(EDA, self).__init__(DAG, machine_num, voltage_relative_speed_pair = voltage_relative_speed_pair,
                 learn_rate=learn_rate)

        self.network_domain = network_domain
        self.power_domain = power_domain
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        self.simulation_num = simulation_num
        self.energy_coeffi = energy_coeffi

        # Define a EDA algorithm
        creator.create("MultiObjMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.MultiObjMin)

        self.toolbox = base.Toolbox()  # Get the toolbox

        # Register the function of population
        self.toolbox.register("init_individual", self.init_individual, self.DAG)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.init_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 定义评估函数
        self.toolbox.register("evaluate", self.evaluation_func, self.DAG)

        self.toolbox.register("sampling_individual", tools.initIterate, creator.Individual, self.sampling)
        self.toolbox.register("generate", self.generate, self.toolbox.sampling_individual)

        self.toolbox.register("update", self.update)
        self.toolbox.register("mate", self.myCrossover)

        self.toolbox.register("mutate", self.myMutate)

        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("select1", tools.selTournamentDCD)

    def generate(self,func,n):
        return [func() for _ in range(n)]

    def init_individual(self, DAG):
        matrix = DAG.matrix
        processing_sequence = []
        while len(processing_sequence) < matrix.shape[0]:
            nodes = self.check_node(matrix, processing_sequence)
            random.shuffle(nodes)
            processing_sequence += nodes

        processing_location = []
        for _ in range(matrix.shape[0]):
            processing_location.append(random.choice(self.machines))

        processing_speed = []
        for _ in range(matrix.shape[0]):
            processing_speed.append(random.choice(list(self.voltage_relative_speed_pair.keys())))

        backup_location = []
        for _ in range(matrix.shape[0]):
            tmp_machines = self.machines.copy()
            tmp_machines.remove(processing_location[_])
            backup_machine = random.choice(tmp_machines)
            backup_location.append(backup_machine)

        return [processing_sequence, processing_location, processing_speed, backup_location]

    def evaluation_func(self, DAG, individual):
        start_end_time = self.calculate_start_end_time(DAG, individual)
        makespan = self.eval1(start_end_time)
        return makespan, self.eval2(individual, start_end_time, makespan), self.eval3(individual, start_end_time,
                                                                                      makespan)
    def eval1(self, start_end_time):
        return max([i[1] for i in start_end_time.values()])

    def eval2(self, individual, start_end_time, makespan):
        machine_processing_time = [0.0 for i in range(len(self.machines))]
        busy_energy = 0.0
        for k, v in start_end_time.items():
            location = individual[0].index(k)
            speed = individual[2][location]
            machine_index = individual[1][location]
            busy_energy += self.energy_coeffi * (self.voltage_relative_speed_pair[speed] ** 2) * (v[1] - v[0]) / speed
            machine_processing_time[machine_index] += (v[1] - v[0])

        idle_energy = 0.0
        for i in machine_processing_time:
            assert makespan - i >= 0
            idle_energy += self.energy_coeffi * (self.voltage_relative_speed_pair[0.4] ** 2) * (makespan - i)

        return idle_energy + busy_energy

    def eval3(self, individual, start_end_time, makespan):
        simulation_unit = Simulation_unit(individual, start_end_time, makespan)
        units = [simulation_unit for _ in range(self.simulation_num)]
        with Pool(cpu_count()) as p:
            simulation_result = list(p.imap(self.encapsulation_simulation, units))
        return float(-sum(simulation_result)) / len(simulation_result)

    def calculate_start_end_time(self, DAG, individual):
        start_end_time = {}
        matrix = DAG.matrix
        nodes_wights = DAG.nodes
        machines_time = [0.0 for i in range(len(self.machines))]

        while len(start_end_time) < matrix.shape[0]:
            nodes = self.check_node(matrix, list(start_end_time.keys()))
            if len(start_end_time) == 0:
                for i in nodes:
                    start_end_time[i] = (max(0, machines_time[individual[1][individual[0].index(i)]]),
                                         nodes_wights[str(i)] / individual[2][individual[0].index(i)])
                    machines_time[individual[1][individual[0].index(i)]] = start_end_time[i][1]
                continue

            for i in nodes:
                pre_jobs = self.find_pre_jobs(matrix, i)
                location = individual[1][individual[0].index(i)]
                start_time_list = []
                for j in pre_jobs:
                    pre_job_location = individual[1][individual[0].index(j)]
                    if location == pre_job_location:
                        transmission_time = 0
                    else:
                        transmission_time = matrix[j, i]

                    start_time_list.append(start_end_time[j][1] + transmission_time)
                start_time = max(max(start_time_list), machines_time[location])
                end_time = start_time + nodes_wights[str(i)] / individual[2][individual[0].index(i)]
                start_end_time[i] = (start_time, end_time)
                machines_time[location] = start_end_time[i][1]

        return start_end_time

    def myCrossover(self, individual1, individual2):
        matrix = self.DAG.matrix

        check_interval = []
        checked = []
        end = 0
        while len(checked) < len(individual1[0]):
            start = end
            add_nodes = self.check_node(matrix, checked)
            end += len(add_nodes)
            check_interval.append((start, end))
            checked += add_nodes

        for i in check_interval:
            if random.random() < 0.5:
                tmp = individual1[0][i[0]:i[1]]
                individual1[0][i[0]:i[1]] = individual2[0][i[0]:i[1]]
                individual2[0][i[0]:i[1]] = tmp

        for ind in range(len(individual1[1])):
            if random.random() < 0.5:
                tmp = individual1[1][ind]
                individual1[1][ind] = individual2[1][ind]
                individual2[1][ind] = tmp

        for ind in range(len(individual1[2])):
            if random.random() < 0.5:
                tmp = individual1[2][ind]
                individual1[2][ind] = individual2[2][ind]
                individual2[2][ind] = tmp

        for ind in range(len(individual1[3])):
            if random.random() < 0.5:
                tmp = individual1[3][ind]
                if individual1[1][ind] != individual2[3][ind]:
                    individual1[3][ind] = individual2[3][ind]
                if individual2[1][ind] != tmp:
                    individual2[3][ind] = tmp

        return individual1, individual2

    def myMutate(self, individual):
        matrix = self.DAG.matrix

        check_interval = []
        checked = []
        end = 0
        while len(checked) < len(individual[0]):
            start = end
            add_nodes = self.check_node(matrix, checked)
            end += len(add_nodes)
            check_interval.append((start, end))
            checked += add_nodes

        for i in check_interval:
            if i[1] - i[0] > 1:
                choice_list = [i for i in range(i[0], i[1])]
                choice_num1 = random.choice(choice_list)
                choice_list.remove(choice_num1)
                choice_num2 = random.choice(choice_list)

                tmp = individual[0][choice_num1]
                individual[0][choice_num1] = individual[0][choice_num2]
                individual[0][choice_num2] = tmp

        for ind in range(len(individual[1])):
            if random.random() < 0.5:
                tmp = random.choice(self.machines)
                individual[1][ind] = tmp

        for ind in range(len(individual[2])):
            if random.random() < 0.5:
                tmp = random.choice(list(self.voltage_relative_speed_pair.keys()))
                individual[2][ind] = tmp

        for ind in range(len(individual[3])):
            if random.random() < 0.5:
                tmp = random.choice(self.machines)
                individual[3][ind] = tmp

            if individual[3][ind] == individual[1][ind]:
                machines_for_choice = [i for i in range(len(self.machines))]
                machines_for_choice.remove(individual[1][ind])
                individual[3][ind] = random.choice(machines_for_choice)

        return individual

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

    def find_pre_jobs(self, matrix, i):
        pre_jobs = []
        for ind, i in enumerate(matrix[:, i] != 0):
            if i == True:
                pre_jobs.append(ind)
        return pre_jobs

    def encapsulation_simulation(self, unit):
        return self.simulation(unit.individual, unit.start_end_time, unit.makespan)

    def simulation(self, individual, start_end_time, makespan):

        network_errors = []
        for _ in range(len(self.network_domain)):
            end = 0
            tmp = []
            while end < makespan:
                start = end + np.random.exponential(self.alpha1, size=1)[0]
                if start >= makespan:
                    break
                end = start + np.random.exponential(self.alpha3, size=1)[0]
                if end > makespan:
                    tmp.append((start, makespan))
                else:
                    tmp.append((start, end))
            network_errors.append(tmp)

        power_errors = []
        for _ in range(len(self.power_domain)):
            end = 0
            tmp = []
            while end < makespan:
                start = end + np.random.exponential(self.alpha2, size=1)[0]
                if start >= makespan:
                    break
                end = start + np.random.exponential(self.alpha4, size=1)[0]
                if end > makespan:
                    tmp.append((start, makespan))
                else:
                    tmp.append((start, end))
            power_errors.append(tmp)

        error_jobs = []
        for key, value in start_end_time.items():
            if key in error_jobs:
                continue

            job_index = individual[0].index(key)
            processing_location = individual[1][job_index]
            backup_location = individual[3][job_index]

            for ind, region in enumerate(self.network_domain):
                if processing_location in region:
                    network_region = ind
                    break
            network_error_times = network_errors[network_region]

            for ind, region in enumerate(self.power_domain):
                if processing_location in region:
                    power_region = ind
                    break
            power_error_times = power_errors[power_region]

            for ind, region in enumerate(self.network_domain):
                if backup_location in region:
                    network_region = ind
                    break
            backup_network_error_times = network_errors[network_region]

            for ind, region in enumerate(self.power_domain):
                if backup_location in region:
                    power_region = ind
                    break
            backup_power_error_times = power_errors[power_region]

            backup_error = backup_network_error_times + backup_power_error_times
            processing_error = network_error_times + power_error_times

            check_group_2 = list(product(processing_error, [value]))

            for tup in check_group_2:
                if self.intersection(tup[0], tup[1]) != None:
                    error_jobs.append(key)
                    break

            if key in error_jobs:
                check_group_3 = list(product(processing_error, [value], backup_error))
                for tup in check_group_3:
                    if self.intersectionN(tup) != None:
                        for child in self.find_child_nodes(key):
                            if child not in error_jobs:
                                error_jobs.append(child)
                        break

        assert len(error_jobs) <= len(start_end_time)

        return float(len(error_jobs)) / float(len(start_end_time))

    def intersection(self, range_a, range_b):
        if not (range_a and range_b):
            return None
        lower_a, upper_a = range_a
        lower_b, upper_b = range_b
        if lower_b < lower_a:
            if upper_b < lower_a:
                return None
            if upper_b < upper_a:
                return (lower_a, upper_b)
            return tuple(range_a)
        if lower_b < upper_a:
            if upper_b < upper_a:
                return tuple(range_b)
            return (lower_b, upper_a)
        return None

    def intersectionN(self, ranges):
        return reduce(self.intersection, ranges)

    def find_child_nodes(self, key):
        child_nodes = []
        edge_list = self.DAG.edge_list

        head = [key]

        while len(head) > 0:
            tmp = []
            for i in edge_list:
                if int(i[0]) in head:
                    if int(i[1]) not in child_nodes:
                        child_nodes.append(int(i[1]))
                    tmp.append(int(i[1]))
            head = tmp

        return child_nodes

    def update(self, pop):
        # elite_list = tools.sortNondominated(pop, len(pop), first_front_only=True)
        elite_list = tools.selNSGA2(pop, int(len(pop)*0.5))
        self.update_0(elite_list)
        self.update_1(elite_list)
        self.update_2(elite_list)
        self.update_3(elite_list)

    def go_run(self, loop_num, population_num,CXPB,MPB):
        random.seed(42)

        # Statistics computation
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        pop = self.toolbox.population(population_num)

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)
        pop = self.toolbox.select(pop,len(pop))


        for gen in range(1,loop_num+1):
            # parents = population
            # pop = self.toolbox.generate(population_num)
            offspring = self.toolbox.select1(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    self.toolbox.mate(ind1, ind2)

                if random.random() <= MPB:
                    self.toolbox.mutate(ind1)
                if random.random() <= MPB:
                    self.toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop = self.toolbox.select(pop + offspring, population_num)
            # self.toolbox.update(population+parents)
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)
            # for indivi in population:
            #     if random.random() < 0.05:
            #         self.myMutate(indivi)
            #
            # fitnesses = self.toolbox.map(self.toolbox.evaluate, population)
            # for ind, fit in zip(population, fitnesses):
            #     ind.fitness.values = fit
            #
            # self.toolbox.update(population+parents)
            #
            # record = stats.compile(population)
            # logbook.record(gen=gen, evals=len(invalid_ind), **record)
            # print(logbook.stream)

            # front = np.array([ind.fitness.values for ind in population])
            # plt.scatter(front[:, 0], front[:, 1], c="b")
            # plt.axis("tight")
            # plt.show()

        return pop, logbook
class Simulation_unit(object):
    def __init__(self,individual,start_end_time,makespan):
        self.individual = individual
        self.start_end_time = start_end_time
        self.makespan = makespan

