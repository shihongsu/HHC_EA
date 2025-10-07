import networkx as nx
import random as rd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n = 20
    p = 0.5

    g = nx.erdos_renyi_graph(n, p)

    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(n, p)

    for (u, v) in g.edges():
        g[u][v]['weight'] = rd.randint(1, 50)
    
    # set start point
    start = rd.choice(list(g.nodes()))
    print(start)
    for u in g.nodes():
        if u == start:
            g.nodes[u]["start"] = True
        else:
            g.nodes[u]["start"] = False

    # constraint should vary (as constant or related to degree of vertices and order of the graph)
    for u in g.nodes():
        g.nodes[u]["degree_constraint"] = 4 # rd.randint(1, len(g.nodes))
        # can be (3, g.degree[u] + 3), ...
    # print(g.nodes(data = True))

    plt.figure()
    pos = nx.spring_layout(g, seed = 123)
    nx.draw(g, pos = pos, with_labels=True, node_color="lightblue", edge_color="gray")
    edge_labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color="red")
    node_labels = nx.get_node_attributes(g, 'degree_constraint')
    label_pos = {node:(pos[node][0] + 0.05, pos[node][1] + 0.05) for node in g.nodes}
    nx.draw_networkx_labels(g, label_pos, labels = node_labels, font_color = 'green')
    plt.show()

    nx.write_graphml(g, "test_case2.graphml")