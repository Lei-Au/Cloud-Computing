import matplotlib.pyplot as plt
import numpy as np
from get_instance import DAG
from deap import tools
from myGA import GA
import mpl_toolkits.mplot3d as Axes3d

instance = DAG()
ga = GA(instance,2,[[0],[1]],[[0],[1]])
if __name__ =='__main__':
    pop, stats = ga.ga_run(120, 120, 0.7, 0.05)
    pop = tools.sortNondominated(pop, len(pop))[0]
    front = np.array([ind.fitness.values for ind in pop])
    plt.scatter(front[:,0], front[:,1], c="b")
    plt.axis("tight")
    plt.show()


