import numpy as np

from get_instance import DAG
from myGA import GA
import numpy
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3d
from deap import tools

instance = DAG()
ga = GA(instance,4,[[0],[1,2],[3]],[[0,3],[1],[2]])
if __name__ =='__main__':
    pop, stats = ga.ga_run(120, 120,0.8,0.1)
    pop_fixed = tools.sortNondominated(pop, len(pop))[0]
    front = numpy.array([ind.fitness.values for ind in pop_fixed])
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.axis("tight")
    plt.show()
    front[:, 0] = np.array(front[:, 0])
    front[:, 1] = np.array(front[:, 1])
    front[:, 2] = np.array(front[:, 2])
    print(front[:,0],'\n',front[:,1],'\n', front[:,2],'\n')