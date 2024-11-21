from CDEDA import EDA, Simulation_unit
from itertools import product
from multiprocessing import Pool, cpu_count
import numpy as np
import time

class EDA_fixed(EDA):
    def __init__(self,DAG, machine_num, network_domain, power_domain, network_errors_list=None, power_errors_list=None,
                 voltage_relative_speed_pair = {1:4, 0.8:2, 0.6:1.2, 0.4:0.5}, energy_coeffi=1,
                 alpha1=20, alpha2=20, alpha3=20, alpha4=20, simulation_num = 500, learn_rate=0.05):

        super(EDA_fixed, self).__init__(DAG, machine_num, network_domain, power_domain,
                 voltage_relative_speed_pair = voltage_relative_speed_pair, energy_coeffi=energy_coeffi,
                 alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, alpha4=alpha4, simulation_num = simulation_num, learn_rate=learn_rate)

        self.network_errors_list = network_errors_list
        self.power_errors_list = power_errors_list

    def eval3(self, individual, start_end_time, makespan):
        units = []
        for ind in range(self.simulation_num):
            simulation_unit = Simulation_unit_fixed(individual, start_end_time, makespan, self.network_errors_list[ind], self.power_errors_list[ind])
            units.append(simulation_unit)

        simulation_result = []
        for unit in units:
            simulation_result.append(self.encapsulation_simulation(unit))
        # with Pool(cpu_count()) as p:
        #     simulation_result = list(p.imap(self.encapsulation_simulation_fixed, units))

        return float(-sum(simulation_result)) / len(simulation_result)

    def encapsulation_simulation_fixed(self, unit):
        return self.simulation_fixed(unit.individual, unit.start_end_time, unit.makespan, unit.network_errors, unit.power_errors)

    def simulation_fixed(self, individual, start_end_time, makespan, network_errors, power_errors):
        network_errors_constraint = []
        for area in network_errors:
            tmp = []
            for network_error in area:
                if network_error[0] >= makespan:
                    break
                elif network_error[1] > makespan:
                    tmp.append((network_error[0], makespan))
                    break
                tmp.append((network_error[0], network_error[1]))
            network_errors_constraint.append(tmp)
        network_errors = network_errors_constraint

        power_errors_constraint = []
        for area in power_errors:
            tmp = []
            for power_error in area:
                if power_error[0] >= makespan:
                    break
                elif power_error[1] > makespan:
                    tmp.append((power_error[0], makespan))
                    break
                else:
                    tmp.append((power_error[0], power_error[1]))
            power_errors_constraint.append(tmp)
        power_errors = power_errors_constraint

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
#
    def get_simulation_instance(self, max_makespan):
        network_errors_list, power_errors_list = [],[]

        for _ in range(self.simulation_num):

            network_errors = []
            for _ in range(len(self.network_domain)):
                end = 0
                tmp = []
                while end < max_makespan:
                    start = end + np.random.exponential(self.alpha1, size=1)[0]
                    if start >= max_makespan:
                        break
                    end = start + np.random.exponential(self.alpha3, size=1)[0]
                    if end > max_makespan:
                        tmp.append((start, max_makespan))
                    else:
                        tmp.append((start, end))
                network_errors.append(tmp)
            network_errors_list.append(network_errors)

            power_errors = []
            for _ in range(len(self.power_domain)):
                end = 0
                tmp = []
                while end < max_makespan:
                    start = end + np.random.exponential(self.alpha2, size=1)[0]
                    if start >= max_makespan:
                        break
                    end = start + np.random.exponential(self.alpha4, size=1)[0]
                    if end > max_makespan:
                        tmp.append((start, max_makespan))
                    else:
                        tmp.append((start, end))
                power_errors.append(tmp)
            power_errors_list.append(power_errors)

        return network_errors_list, power_errors_list

class Simulation_unit_fixed(object):
    def __init__(self,individual,start_end_time,makespan,network_errors,power_errors):
        self.individual = individual
        self.start_end_time = start_end_time
        self.makespan = makespan
        self.network_errors = network_errors
        self.power_errors = power_errors

 
 