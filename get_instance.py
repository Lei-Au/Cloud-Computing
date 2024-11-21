import numpy as np

class Graph_Matrix():
    def __init__(self, vertices=[], matrix=[]):

        self.matrix = matrix
        self.edges_dict = {}
        self.edges_array = []
        self.vertices = vertices
        self.num_edges = 0

        if len(matrix) > 0:
            if len(vertices) != len(matrix):
                raise IndexError
            self.edges = self.getAllEdges()
            self.num_edges = len(self.edges)

        elif len(vertices) > 0:
            self.matrix = [[0 for col in range(len(vertices))] for row in range(len(vertices))]

        self.num_vertices = len(self.matrix)

    def isOutRange(self, x):
        try:
            if x >= self.num_vertices or x <= 0:
                raise IndexError
        except IndexError:
            print("节点下标出界")

    def isEmpty(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices == 0

    def add_vertex(self, key):
        if key not in self.vertices:
            self.vertices[key] = len(self.vertices) + 1

        for i in range(self.getVerticesNumbers()):
            self.matrix[i].append(0)

        self.num_vertices += 1

        nRow = [0] * self.num_vertices
        self.matrix.append(nRow)

    def getVertex(self, key):
        pass

    def add_edges_from_list(self, edges_list):
        for i in range(len(edges_list)):
            self.add_edge(edges_list[i][0], edges_list[i][1], edges_list[i][2], )

    def add_edge(self, tail, head, cost=0):
        if tail not in self.vertices:
            self.add_vertex(tail)
        if head not in self.vertices:
            self.add_vertex(head)

        self.matrix[self.vertices.index(tail)][self.vertices.index(head)] = cost

        self.edges_dict[(tail, head)] = cost
        self.edges_array.append((tail, head, cost))
        self.num_edges = len(self.edges_dict)

    def getEdges(self, V):
        pass

    def getVerticesNumbers(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices

    def getAllVertices(self):
        return self.vertices

    def getAllEdges(self):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if 0 < self.matrix[i][j] < float('inf'):
                    self.edges_dict[self.vertices[i], self.vertices[j]] = self.matrix[i][j]
                    self.edges_array.append([self.vertices[i], self.vertices[j], self.matrix[i][j]])

        return self.edges_array

    def __repr__(self):
        return str(''.join(str(i) for i in self.matrix))

    def to_do_vertex(self, i):
        print('vertex: %s' % (self.vertices[i]))

    def to_do_edge(self, w, k):
        print('edge tail: %s, edge head: %s, weight: %s' % (self.vertices[w], self.vertices[k], str(self.matrix[w][k])))

class DAG(object):
    # 任务节点及其任务量
    nodes = {'0': 1, '1': 2, '2': 1, '3': 2, '4': 3, '5': 1, '6': 2, '7': 3, '8': 1, '9': 3, '10': 5, '11': 2}
    # 任务前后序关系,信息传输所需时间
    edge_list = [('0', '1', 1), ('0', '2', 1), ('2', '3', 2), ('1', '4', 3), ('3', '4', 1), ('1', '5', 2),
                 ('5', '6', 3),
                 ('4', '6', 1), ('3', '6', 2), ('4', '7', 1), ('6', '8', 1), ('7', '8', 3), ('2', '9', 1),
                 ('9', '7', 1),
                 ('6', '10', 1),
                 ('10', '11', 1), ('8', '11', 2), ('7', '11', 3)]
    my_graph = Graph_Matrix(list(nodes.keys()))
    my_graph.add_edges_from_list(edge_list)
    matrix = np.array(my_graph.matrix)
