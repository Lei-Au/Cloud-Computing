from CDEDA import EDA
import random
from deap import tools
import numpy as np
import copy

class EDA_with_neighborhood_search(EDA):
    def __init__(self,DAG, machine_num, network_domain, power_domain,
                 voltage_relative_speed_pair = {1:4, 0.8:2, 0.6:1.2, 0.4:0.5}, energy_coeffi=1,
                 alpha1=20, alpha2=20, alpha3=20, alpha4=20, simulation_num = 500, learn_rate=0.1):
        super(EDA_with_neighborhood_search, self).__init__(DAG, machine_num=machine_num, network_domain=network_domain, power_domain=power_domain,
                 voltage_relative_speed_pair = voltage_relative_speed_pair, energy_coeffi=energy_coeffi,
                 alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, alpha4=alpha4, simulation_num = simulation_num, learn_rate=learn_rate)

    def neighborhood_search(self, individual):
        start_end_time = self.calculate_start_end_time(self.DAG,individual)
        idle_dataset_list = self.find_idle_time(individual, start_end_time)
        idle_dataset_list.sort(key=lambda x: x.idleTime, reverse=True)
        if random.random() < 0.5:
            for idle_dataset in idle_dataset_list:
                label = self.makespan_search(idle_dataset, individual, start_end_time)
                if label == True:
                    break
        if random.random() < 0.5:
            for idle_dataset in idle_dataset_list:
                label = self.TEC_search(idle_dataset, individual)
                if label == True:
                    break
        if random.random() < 0.5:
            self.risk_search(individual)

    def find_idle_time(self, individual, start_end_time):
        machine_cursor = {}
        for i in self.machines:
            machine_cursor[i] = 0
        idle_dataset_list = []
        for ind, job, location in zip(range(len(individual[0])),individual[0], individual[1]):
            if start_end_time[job][0] - machine_cursor[location] > 10**(-8):
                preOrderJob = None
                for find_pre in range(ind-1, -1, -1):
                    if individual[1][find_pre] == location:
                        preOrderJob = individual[0][find_pre]
                        break

                if preOrderJob != None:
                    idle_dataset_list.append(Idle_dataset(preOrderJob, job,
                                                      start_end_time[job][0]-start_end_time[preOrderJob][1],location))
                else:
                    idle_dataset_list.append(Idle_dataset("virtual", job,
                                                          start_end_time[job][0] - 0,
                                                          location))
            machine_cursor[location] = start_end_time[job][1]
        return idle_dataset_list

    def makespan_search(self, idle_dataset, individual, start_end_time):
        pre_order_jobs = []
        for i in self.DAG.edge_list:
            if int(i[1]) == idle_dataset.postOrderJob:
                if self.check_on_a_machine(int(i[0]) ,idle_dataset.postOrderJob, individual):
                    if start_end_time[idle_dataset.postOrderJob][0] - start_end_time[int(i[0])][1] < 10 ** (-8):
                        pre_order_jobs.append(int(i[0]))
                else:
                    if start_end_time[idle_dataset.postOrderJob][0] - (start_end_time[int(i[0])][1] + i[2]) < 10 ** (-8):
                        pre_order_jobs.append(int(i[0]))
        for i in pre_order_jobs:
            if self.get_job_speed(i, individual) == list(self.voltage_relative_speed_pair.keys())[0]:
                return False

        for i in pre_order_jobs:
            new_speed = random.choice(list(self.voltage_relative_speed_pair.keys())[:list(self.voltage_relative_speed_pair.keys()).index(self.get_job_speed(i, individual))])
            ind = list(individual[0]).index(i)
            individual[2][ind] = new_speed

        return True

    def TEC_search(self, idle_dataset, individual):
        if idle_dataset.preOrderJob == 'virtual':
            return False

        job_speed = individual[2][list(individual[0]).index(idle_dataset.preOrderJob)]
        candidate_speed = []
        for i in list(self.voltage_relative_speed_pair.keys()):
            if i < job_speed:
                node_wight = self.DAG.nodes[str(idle_dataset.preOrderJob)]
                if node_wight/i - node_wight/job_speed <= idle_dataset.idleTime:
                    candidate_speed.append(i)
        if len(candidate_speed) == 0:
            return False

        new_speed = random.choice(candidate_speed)
        individual[2][list(individual[0]).index(idle_dataset.preOrderJob)] = new_speed

        return True

    def risk_search(self, individual):
        for ind, processing_location, backup_location in zip(range(len(individual[0])),individual[1],individual[3]):
            available_processors = copy.deepcopy(self.machines)
            for domain in self.network_domain:
                if processing_location in domain:
                    for i in domain:
                        available_processors.remove(i)
            for domain in self.power_domain:
                if processing_location in domain:
                    for i in domain:
                        if i in available_processors:
                            available_processors.remove(i)
            if len(available_processors) != 0:
                individual[3][ind] = random.choice(available_processors)




    def check_on_a_machine(self,job1, job2, individual):
        job1_location = individual[1][list(individual[0]).index(job1)]
        job2_location = individual[1][list(individual[0]).index(job2)]
        if job1_location == job2_location:
            return True
        else:
            return False

    def get_job_speed(self,job, individual):
        ind = list(individual[0]).index(job)
        return individual[2][ind]

    def go_run(self, loop_num, population_num):
        random.seed(42)

        # Statistics computation
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        population = self.toolbox.population(population_num)

        for indivi in population:
            self.neighborhood_search(indivi)

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        record = stats.compile(population)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)

        for gen in range(loop_num):
            parents = population
            population = self.toolbox.generate(population_num)

            # fitnesses = self.toolbox.map(self.toolbox.evaluate, population)
            # for ind, fit in zip(population, fitnesses):
            #     ind.fitness.values = fit
            # record = stats.compile(population)
            # logbook.record(gen=0, evals=len(invalid_ind), **record)
            # print(logbook.stream)

            for indivi in population:
                if random.random() < 0.2:
                    self.neighborhood_search(indivi)

            for indivi in population:
                if random.random() < 0.05:
                    self.myMutate(indivi)

            fitnesses = self.toolbox.map(self.toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            self.toolbox.update(population + parents)

            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)

            # front = np.array([ind.fitness.values for ind in population])
            # plt.scatter(front[:, 0], front[:, 1], c="b")
            # plt.axis("tight")
            # plt.show()

        return population, logbook

class Idle_dataset(object):
    def __init__(self,preOrderJob, postOrderJob, idleTime, machine_num):
        self.preOrderJob = preOrderJob
        self.postOrderJob = postOrderJob
        self.idleTime = idleTime
        self.machine_num = machine_num

    def __str__(self):
        return str(self.preOrderJob) + \
               ' ' + str(self.postOrderJob) + \
               ' ' + str(self.idleTime) + \
               ' ' + str(self.machine_num)


 