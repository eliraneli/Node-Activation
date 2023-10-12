import numpy as np
import os


class Code:

    def __int__(self, H_filename, G_filename):
        if not (os.path.isfile(H_filename) and os.path.isfile(G_filename)):
            raise Exception("one of the file is not exists")

        self.H = self.load_H_filename(H_filename)
        self.G = self.load_G_filename(G_filename)
        self.num_edges = self.H.sum()
        self.edges = self.calculate_edges(self.var_degrees)

    def load_H_filename(self, H_filename):
        # parity-check matrix; Tanner graph parameters
        with open(H_filename) as file:
            # get n and m (n-k) from first line
            lines = file.readlines()
            columnNum, rowNum = np.fromstring(
                lines[0].rstrip('\n'), dtype=int, sep=' ')
            H = np.zeros((rowNum, columnNum)).astype(int)
            # store the degrees of VN and CN
            self.var_degrees, self.chk_degrees  = np.zeros(columnNum).astype(np.int), np.zeros(rowNum).astype(np.int)
            # store the edges of VN and CN
            var_edges, chk_edges = [[] for _ in range(0, columnNum)], [[] for _ in range(0, rowNum)]

            self.max_var_degree, self.max_chk_degree = np.fromstring(
                lines[1].rstrip('\n'), dtype=int, sep=' ')

            for column in range(4, 4 + columnNum):
                nonZeroEntries = np.fromstring(
                    lines[column].rstrip('\n'), dtype=int, sep=' ')
                var_edges[column - 4] = nonZeroEntries
                self.var_degrees[column - 4] = len(nonZeroEntries)
                for row in nonZeroEntries:
                    if row > 0:
                        H[row - 1, column - 4] = 1

            for row in range(column + 1, column + rowNum + 1):
                nonZeroEntries = np.fromstring(
                    lines[row].rstrip('\n'), dtype=int, sep=' ')
                chk_edges[row - column - 1] = nonZeroEntries
                self.chk_degrees[row - column - 1] = len(nonZeroEntries)

            # for each var node, collect and store edges and move to next var node
            d = self.calculate_edges_CN(columnNum, self.var_degrees)

            u = self.calculate_edges_VN(row, chk_edges, d, var_edges)
            # edges for marginalization
            self.edges_m = self.calculate_edges_marginalize(columnNum, self.var_degrees, d)

            # all edges
            self.extrinsic_edges_vc, self.edge_order_vc = self.calculate_extrinsic_edges(columnNum,self.var_degrees, d)
            self.extrinsic_edges_cv, self.edge_order_cv = self.calculate_extrinsic_edges(rowNum, self.chk_degrees, u)


            self.n, self.m = columnNum, rowNum
            # self.k = self.n - self.m
            return H

    def calculate_edges_marginalize(self, n, var_degrees, d):
        edges_m = []
        for i in range(0, n):
            temp_e = []
            for e in range(0, var_degrees[i]):
                temp_e.append(d[i][e])
            edges_m.append(temp_e)
        return edges_m

    def calculate_edges_VN(self, m, chk_edges, d, var_edges):
        u = [[] for _ in range(0, m)]
        edge = 0
        for i in range(0, m):
            nonZero = len(chk_edges[i])
            for j in range(0, nonZero):
                v = chk_edges[i][j]
                varible_nodes_num = len(var_edges[v])
                for e in range(0, varible_nodes_num):
                    if (i == var_edges[v][e]):
                        u[i].append(d[v][e])
        return u

    def calculate_edges(self, var_edges):
        edges = []
        for i in range(0, self.n):
            nonZero = len(var_edges[i])
            for j in range(0, nonZero):
                edges.append(i)
        return edges

    def calculate_extrinsic_edges(self,num_nodes,nodes_degrees, matrix):
        extrinsic_edges = []
        edge_order = []
        for i in range(0, num_nodes):
            for j in range(0, nodes_degrees[i]):
                edge_order.append(matrix[i][j])
                temp_edges = []
                for jj in range(0, nodes_degrees[i]):
                    if jj != j:  # extrinsic information only
                        temp_edges.append(matrix[i][jj])
                extrinsic_edges.append(temp_edges)
        return extrinsic_edges, edge_order



    def calculate_edges_CN(self, n, var_degrees):
        d = [[] for _ in range(0, n)]
        edge = 0
        for i in range(0, n):
            for j in range(0, var_degrees[i]):
                d[i].append(edge)
                edge += 1
        return d

    def load_G_filename(self, G_filename):
        G = np.load(G_filename).astype(np.int).transpose()
        self.k = G.shape[1]
        return G

