import networkx as nx
import matplotlib.pyplot as plt
from bidirect_module import *
from queue import PriorityQueue


def print_graph(g):
    for i in range(len(g)):
        print(i, end=": ")
        for j in g[i]:
            print(j, end=" ")
        print()


def print_weighted_grpah(g_w):
    for i in range(len(cities)):
        for j in range(len(cities)):
            print("%5d" % g_w[i][j], end=" ")
        print()


def print_way(way):
    lines = [[]]
    offsets = [0]
    line_id = 0
    for i in way:
        if i < 0:
            print()
            st = False
            offset = 0
            for j in reversed(range(0, line_id + 1)):
                if st:
                    break
                offset = 0
                for z in range(len(lines[j])):
                    offset += len(lines[j][z]) + 1
                    if cities.loc[[-i]]["city_name"].item() == lines[j][z]:
                        offset -= (len(lines[j][z]) + 1)
                        offset += offsets[j]
                        st = True
                        break

            for t in range(offset):
                print(" ", end="")
            offsets.append(offset)
            c_name = cities.loc[[-i]]["city_name"].item()
            lines[line_id].append(c_name)
            lines.append([])
            line_id += 1
            lines[line_id].append(c_name)
            print("└", c_name, sep="", end="")
            continue

        c_name = cities.loc[[i]]["city_name"].item()
        lines[line_id].append(c_name)
        print("─", c_name, sep="", end="")


def print_sep():
    print("-------------------------------------------")


def get_int() -> int:
    valid = False
    a = 0
    while not valid:
        a = input()
        try:
            a = int(a)
            valid = True
        except ValueError:
            print('wrong foramat')
            continue
    return a


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ALGORITHMS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
visited = []
g_cities = []
g_cities_names = []
g_cities_weighted = []
cities = []
queue = []
start = 0
finish = 0
dls_ways = []
dls_way = 0
heuristic = []


def dfs(v, dest, len_ans, way):
    way.append(v)
    visited[v] = True
    len_ans += 1
    if v == dest:
        print_way(way)
        print()
        return
    for i in g_cities[v]:
        if not visited[i]:
            dfs(i, dest, len_ans, way)
            way.append(-v)


def dls(v, dest, way, iterations, limit):
    global dls_way
    way.append(v)
    visited[v] = True
    iterations += 1
    ans = False

    if iterations >= limit:
        dls_way = way
        return False

    if v == dest:
        dls_way = way
        return True

    for i in g_cities[v]:
        if not visited[i]:
            ans = ans | dls(i, dest, way, iterations, limit)
            way.append(-v)
    return ans


def bfs(node, dest):
    visited[node] = True
    queue.append([node, 0])

    traversal = []
    while queue:
        s = queue.pop(0)
        for neighbour in g_cities[s[0]]:
            if not visited[neighbour]:
                c_a = cities.loc[[s[0]]]["city_name"].item()
                c_b = cities.loc[[neighbour]]["city_name"].item()
                traversal.append(("[" + str(s[1]) + "]" + c_a, "[" + str(s[1] + 1) + "]" + c_b))
                if neighbour == dest:
                    queue.clear()
                    break
                visited[neighbour] = True
                queue.append([neighbour, s[1] + 1])

    return traversal


def best_first(v, dest):
    pq = PriorityQueue()
    pq.put((0, v, -1, -1, 0))
    visited[start] = True
    traversal = []

    while not pq.empty():
        v_next = pq.get()

        if v_next[2] != -1:
            c_a = cities.loc[[v_next[1]]]["city_name"].item()
            c_b = cities.loc[[v_next[2]]]["city_name"].item()
            w = v_next[3]
            traversal.append((c_a, c_b, w))

        v_n = v_next[1]
        if v_n == dest:
            return True, traversal, v_next[4] + v_next[3]
        for neigh, edge in g_cities_weighted[v_n]:
            if not visited[neigh]:
                visited[neigh] = True
                l = heuristic.loc[[neigh]]["straight_dist"].item()
                ans = v_next[3] + v_next[4]
                pq.put((l, neigh, v_n, edge, ans))
    return False


def best_first_a(v, dest):
    pq = PriorityQueue()
    pq.put((0, v, -1, -1, 0))
    visited[start] = True
    traversal = []

    while not pq.empty():
        v_next = pq.get()

        if v_next[2] != -1:
            c_a = cities.loc[[v_next[1]]]["city_name"].item()
            c_b = cities.loc[[v_next[2]]]["city_name"].item()
            w = v_next[3]
            traversal.append((c_a, c_b, w))

        v_n = v_next[1]
        if v_n == dest:
            return True, traversal, v_next[4] + v_next[3]
        for neigh, edge in g_cities_weighted[v_n]:
            if not visited[neigh]:
                visited[neigh] = True
                l = heuristic.loc[[neigh]]["straight_dist"].item()
                ans = v_next[3] + v_next[4]
                pq.put((ans + l, neigh, v_n, edge, ans))
    return False


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ALGORITHMS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def init():
    global start
    global finish
    global cities
    global g_cities
    global g_cities_weighted
    global heuristic

    n_var = 15 % 10 + 1
    print("variant " + str(n_var))
    var = pd.read_csv("variants", delimiter=" ", index_col=0)
    print(var.loc[[n_var]])
    start = var.loc[[n_var]]["cityA"].item()
    finish = var.loc[[n_var]]["cityB"].item()

    df = pd.read_csv("city_dists", delimiter=" ")

    cities = set()
    for index, row in df.iterrows():
        cities.add(row["cityA"])
        cities.add(row["cityB"])

    cities = list(cities)
    cities.sort()

    cities = pd.DataFrame(cities, index=[i for i in range(len(cities))], columns=["city_name"])

    g_cities = [[] for i in range(len(cities))]
    g_cities_weighted = [[] for i in range(len(cities))]

    for index, row in df.iterrows():
        city_a = cities[cities["city_name"] == row["cityA"]].index[0]
        city_b = cities[cities["city_name"] == row["cityB"]].index[0]
        g_cities[city_a].append(city_b)
        g_cities[city_b].append(city_a)
        g_cities_names.append((row["cityA"], row["cityB"]))
        g_cities_weighted[city_a].append((city_b, row["dist"]))
        g_cities_weighted[city_b].append((city_a, row["dist"]))

    heuristic = pd.read_csv("straight_line", encoding="utf-8", delimiter=",", index_col=0)
    start = cities[cities["city_name"] == start].index[0]
    finish = cities[cities["city_name"] == finish].index[0]


def main():
    global visited
    global dls_way
    while True:
        print_sep()
        print('Uninformative methods')
        print('(1) dfs')
        print('(2) bfs')
        print('(3) dfs with limit')
        print('(4) dfs with iterate limit')
        print('(5) bidirect search')
        print('Informative methods')
        print('(6) Best-first search')
        print('(7) Minimizing the total estimate of A')
        print('(0) quit')
        a = get_int()
        if a == 1:
            visited = [False for i in range(len(cities))]
            dfs(start, finish, 0, [])
            print("Time complexity: O(n+m)")
        elif a == 2:
            visited = [False for i in range(len(cities))]
            traversal = bfs(start, finish)

            plt.figure(3, figsize=(12, 12))
            plt.subplot(121)
            G = nx.Graph()
            G.add_edges_from(traversal)
            nx.draw(G, node_size=80, font_size=16, with_labels=True, node_color='orange')

            # plt.subplot(122)
            # G2 = nx.Graph()
            # G2.add_edges_from(g_cities_names)
            # nx.draw(G2, node_size=40, font_size=8, with_labels=True, node_color='blue')
            plt.show()
            print("Time complexity: O(n+m)")
        elif a == 3:
            visited = [False for i in range(len(cities))]
            print("enter limit: ", end="")
            limit = get_int()
            print()
            way = dls(start, finish, [], 0, limit)
            if way:
                print("found way")
                print_way(dls_way)
                print()
                print("Time complexity: O(b^l), b - средний коэффициент ветвления, l - глубина")
        elif a == 4:
            st = False
            limit = 0
            while not st:
                print("limit:", limit)
                limit += 1
                visited = [False for i in range(len(cities))]
                dls_way = 0
                st = dls(start, finish, [], 0, limit)
                print_way(dls_way)
                print_sep()
            print("final limit: ", limit)
            print("Time complexity: O(b^l), b - средний коэффициент ветвления, l - глубина")
        elif a == 5:
            bid, colors = bidirect(start, finish, g_cities, cities)
            if bid is not None:
                plt.figure(3, figsize=(12, 12))
                plt.subplot(121)
                G1 = nx.Graph()
                G1.add_edges_from(bid)
                colors = colors.set_index('ID')
                colors = colors.reindex(G1.nodes())
                nx.draw(G1, node_size=40, font_size=8, with_labels=True, node_color=colors['color'])

                plt.subplot(122)
                G2 = nx.Graph()
                G2.add_edges_from(g_cities_names)
                nx.draw(G2, node_size=40, font_size=8, with_labels=True, node_color='green')
                plt.show()
                print("Time complexity: O(b^d), b - средний коэффициент ветвления, d -  расстояние между начальной и конечной")
        elif a == 6:
            visited = [False for i in range(len(cities))]
            greedy = best_first(start, finish)
            if greedy[0]:
                plt.figure(3, figsize=(12, 12))
                G1 = nx.Graph()
                for edge in greedy[1]:
                    G1.add_edge(edge[0], edge[1], weight=edge[2])

                pos = nx.spring_layout(G1)  # pos = nx.nx_agraph.graphviz_layout(G)
                nx.draw_networkx(G1, pos, node_size=40, font_size=8, node_color='orange')
                labels = nx.get_edge_attributes(G1, 'weight')
                nx.draw_networkx_edge_labels(G1, pos, edge_labels=labels)
                plt.show()
                print('Path length:', greedy[2])
            else:
                print("no path found")
        elif a == 7:
            visited = [False for i in range(len(cities))]
            greedy = best_first_a(start, finish)
            if greedy[0]:
                plt.figure(3, figsize=(12, 12))
                G1 = nx.Graph()
                for edge in greedy[1]:
                    G1.add_edge(edge[0], edge[1], weight=edge[2])

                pos = nx.spring_layout(G1)  # pos = nx.nx_agraph.graphviz_layout(G)
                nx.draw_networkx(G1, pos, node_size=40, font_size=8, node_color='orange')
                labels = nx.get_edge_attributes(G1, 'weight')
                nx.draw_networkx_edge_labels(G1, pos, edge_labels=labels)
                plt.show()
                print('Path length:', greedy[2])
        elif a == 0:
            break
        else:
            print('wrong command')


if __name__ == '__main__':
    init()
    main()
