from get_instance import DAG
from EDA import EDA
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap import tools
from EDA_fixed import EDA_fixed


instance = DAG()
eda_fixed = EDA_fixed(instance,4,[[0],[1,2],[3]],[[0,3],[1],[2]])
network_errors_list, power_errors_list = eda_fixed.get_simulation_instance(40)
eda_fixed.network_errors_list = network_errors_list
eda_fixed.power_errors_list = power_errors_list

start_time = time.time()
pop_fixed, stats_fixed = eda_fixed.go_run(120, 120,0.8,0.1)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Run time: {elapsed_time} seconds")

pop_fixed = tools.sortNondominated(pop_fixed , len(pop_fixed))[0]
front_fixed = np.array([ind.fitness.values for ind in pop_fixed])

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(front_fixed[:,0], front_fixed[:,1], front_fixed[:,2], c='r', marker='^')  # 点为红色三角形

print(front_fixed[:,0],'\n',front_fixed[:,1],'\n', front_fixed[:,2],'\n')

# 设置坐标轴
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图像
plt.show()

ax = plt.figure().add_subplot(111, projection='3d')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
