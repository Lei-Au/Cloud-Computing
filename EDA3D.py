from get_instance import DAG
from CDEDA import EDA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap import tools
from neighborhood_search import EDA_with_neighborhood_search
from EDA_fixed import EDA_fixed
from EDA_NS_fixed import EDA_NS_fixed

instance = DAG()
eda_fixed = EDA_fixed(instance,4,[[0],[1,2],[3]],[[0,3],[1],[2]])
network_errors_list, power_errors_list = eda_fixed.get_simulation_instance(40)
eda_fixed.network_errors_list = network_errors_list
eda_fixed.power_errors_list = power_errors_list
eda_NS_fixed = EDA_NS_fixed(instance,4,[[0],[1,2],[3]],[[0,3],[1],[2]])
eda_NS_fixed.network_errors_list = network_errors_list
eda_NS_fixed.power_errors_list = power_errors_list
pop_NS_fixed, stats_NS_fixed = eda_fixed.go_run(120, 120,0.8,0.1)
pop_NS_fixed = tools.sortNondominated(pop_NS_fixed , len(pop_NS_fixed))[0]
front_NS_fixed = np.array([ind.fitness.values for ind in pop_NS_fixed])

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(front_NS_fixed[:,0], front_NS_fixed[:,1], front_NS_fixed[:,2], c="b", marker='o')
print(front_NS_fixed[:,0], '\n',front_NS_fixed[:,1],'\n', front_NS_fixed[:,2],'\n')

# 设置坐标轴
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图像
plt.show()

ax = plt.figure().add_subplot(111, projection='3d')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
