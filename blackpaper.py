import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("straight_line", encoding="utf-8", delimiter=",", index_col=0)
    neigh = 2
    l = df.loc[[neigh]]["straight_dist"].item()
    print(l)