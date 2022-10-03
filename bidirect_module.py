import pandas as pd


class BidirectionalSearch:
    def __init__(self, vertices, cities):
        self.vertices = vertices
        self.graph = [[] for i in range(vertices)]
        self.traversal = []
        self.colors = pd.DataFrame(columns=['ID', 'color'])
        self.cities = cities

        self.src_queue = list()
        self.dest_queue = list()

        self.src_visited = [False for i in range(vertices)]
        self.dest_visited = [False for i in range(vertices)]

        self.src_parent = [None] * self.vertices
        self.dest_parent = [None] * self.vertices

    def add_edge(self, src, dest):
        self.graph[src].append(dest)
        self.graph[dest].append(src)

    def add_edges_matrix(self, arr):
        self.graph = arr

    def bfs(self, direction='forward'):
        if direction == 'forward':
            current, bfs_round = self.src_queue.pop(0)
            for neighbour in self.graph[current]:
                if not self.src_visited[neighbour]:
                    c_a = "[" + str(bfs_round) + "]" + self.cities.loc[[current]]["city_name"].item()
                    c_b = "[" + str(bfs_round + 1) + "]" + self.cities.loc[[neighbour]]["city_name"].item()
                    self.traversal.append((c_a, c_b))
                    if c_a not in self.colors["ID"].values:
                        self.colors = self.colors.append({'ID': c_a, 'color': 'blue'}, ignore_index=True)
                    if c_b not in self.colors["ID"].values:
                        self.colors = self.colors.append({'ID': c_b, 'color': 'blue'}, ignore_index=True)


                    self.src_queue.append([neighbour, bfs_round + 1])
                    self.src_visited[neighbour] = True
                    self.src_parent[neighbour] = current

        else:
            current, bfs_round = self.dest_queue.pop(0)
            for neighbour in self.graph[current]:
                if not self.dest_visited[neighbour]:
                    c_a = "[" + str(bfs_round) + "]" + self.cities.loc[[current]]["city_name"].item()
                    c_b = "[" + str(bfs_round + 1) + "]" + self.cities.loc[[neighbour]]["city_name"].item()
                    self.traversal.append((c_a, c_b))
                    if c_a not in self.colors["ID"].values:
                        self.colors = self.colors.append({'ID': c_a, 'color': 'orange'}, ignore_index=True)
                    if c_b not in self.colors["ID"].values:
                        self.colors = self.colors.append({'ID': c_b, 'color': 'orange'}, ignore_index=True)

                    self.dest_queue.append([neighbour, bfs_round + 1])
                    self.dest_visited[neighbour] = True
                    self.dest_parent[neighbour] = current

    def is_intersecting(self):
        for i in range(self.vertices):
            if (self.src_visited[i] and
                    self.dest_visited[i]):
                return i

        return -1

    def print_path(self, intersecting_node,
                   src, dest):
        path = list()
        path.append(intersecting_node)
        i = intersecting_node

        while i != src:
            path.append(self.src_parent[i])
            i = self.src_parent[i]

        path = path[::-1]
        i = intersecting_node

        while i != dest:
            path.append(self.dest_parent[i])
            i = self.dest_parent[i]

        print("*****Path*****")
        path = list(map(str, path))

        for i in path:
            c_name = self.cities.loc[[int(i)]]["city_name"].item()
            print(c_name, end=" ")
        print()

    def bidirectional_search(self, src, dest):
        self.src_queue.append([src, 0])
        self.src_visited[src] = True
        self.src_parent[src] = -1

        self.dest_queue.append([dest, 0])
        self.dest_visited[dest] = True
        self.dest_parent[dest] = -1

        bfs_round = -1

        while self.src_queue and self.dest_queue:
            bfs_round += 1
            self.bfs(direction='forward')
            self.bfs(direction='backward')
            intersecting_node = self.is_intersecting()

            if intersecting_node != -1:
                self.print_path(intersecting_node,
                                src, dest)
                return 1
        return -1


def bidirect(src, dest, edge_matrix, cities):
    graph = BidirectionalSearch(len(cities), cities)
    graph.add_edges_matrix(edge_matrix)

    out = graph.bidirectional_search(src, dest)

    if out == -1:
        print(f"Path does not exist between {src} and {dest}")
        return None
    else:
        return graph.traversal, graph.colors
