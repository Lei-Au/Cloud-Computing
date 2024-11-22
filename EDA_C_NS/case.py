#case1
# 任务节点及其任务量
nodes = {'0': 1, '1': 2, '2': 1, '3': 2, '4': 3}
# 任务前后序关系,信息传输所需时间
edge_list = [('0', '1', 1), ('0', '2', 1), ('2', '3', 2), ('1', '4', 3), ('3', '4', 1)]
#(instance,2,[[0],[1]],[[0],[1]])


#case2
nodes = {'0': 1, '1': 2, '2': 1, '3': 1, '4': 2, '5': 3, '6': 2, '7': 2}
# 任务前后序关系,信息传输所需时间
edge_list = [('0', '1', 1), ('0', '2', 2), ('0', '3', 1), ('0', '4', 1), ('1', '5', 1), ('2', '5', 1), ('3', '5', 2),
             ('3', '6', 1), ('4', '6', 3), ('5', '7', 3), ('6', '7', 1)]
#(instance,3,[[0,1],[1,2],[2]],[[0,2],[1],[2,1]])


#case3
# 任务节点及其任务量
nodes = {'0': 1, '1': 2, '2': 1, '3': 2, '4': 3, '5': 1, '6': 2, '7': 3, '8': 1, '9': 3, '10': 5, '11': 2}
# 任务前后序关系,信息传输所需时间
edge_list = [('0', '1', 1), ('0', '2', 1), ('2', '3', 2), ('1', '4', 3), ('3', '4', 1), ('1', '5', 2), ('5', '6', 3),
             ('4', '6', 1), ('3', '6', 2), ('4', '7', 1), ('6', '8', 1), ('7', '8', 3), ('2', '9', 1), ('9', '7', 1),
             ('6', '10', 1),
             ('10', '11', 1), ('8', '11', 2), ('7', '11', 3)]
#(instance,4,[[0],[1,2],[3]],[[0,3],[1],[2]])
