
import re
import json
import networkx as nx
import matplotlib.pyplot as plt
from datasets.get_datasets import *
from experiments import experiments, bk
from .utils import get_features, load_json, pjoin, save_params, to_numpy

class GRAPH:
    def __init__(self,
                 data_dir,
                 log_dir,
                 params,
                 n_splits,
                 dedup=False,
                 cached=True,
                 use_gpu=True,
                 no_logger=False,
                 progress_bar=False):
        self.cached = cached
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.params = params
        self.n_splits = n_splits
        self.dedup = dedup
        # self.device = 'cuda:0' if use_gpu else 'cpu'
        self.use_gpu = use_gpu
        self.progress_bar = progress_bar

        self.X = None
        self.y = None

        self.network = None
    
    def init_data(self):
        self.create_graph()
    
    def create_graph(self, directed=False, colors=None):
        '''
            Creates a new graph based on data passed as parameter
            Args:
                data: information to build the graph
                directed: True if the graph is directed; False if it is not
            Returns:
                graph object
        '''
        nodes = {}
        # movies, accounts, sportsteam, venues and titles are yellow
        # genre, authors, locations and levels are purple
        # persons, words, sportsleague, classes and titles are blue
        # courses, company and proteins are darkblue

        #colors = {'movie': [0.8,0.1], 'teamplaysinleague': [0.8,0.1], 'teamalsoknownas': [0.8,0.8], 'venue': [0.1,0.8], 'follows': [0.8,0.8], 'tweets': [0.8,0.1], 'publication': [0.8,0.1], 'actor': [0.2,0.2], 'director': [0.3,0.3], 'genre': [0.1,0.5], 'author': [0.1,0.5], 'interaction': [0.3,0.3], 'location': [0.3,0.5], 'courselevel': [0.3,0.5], 'acquired': [0.3,0.3], 'companyalsoknownas': [0.3,0.3], 'female': [0.8,0.8]}

        print("Creating graph...")
        g = nx.Graph()

        source = self.data_dir.split('/')[1]
        predicate = 'workedunder'

        src_total_data = datasets.load(source, bk[source], seed=441773)
        src_data  = datasets.load(source, bk[source], target=predicate, balanced=False, seed=441773)
        data = datasets.group_folds(src_data[0]) + datasets.group_folds(src_data[1])

        all_edges, relations_of_nodes, count_relations = [], {}, {}

        for dt in data:

            if 'recursion' in dt or 'ta' in dt:
                continue

            # Check if it is binary relation
            if ',' in dt:
                from_node, to_node = dt.replace(').','').split('(')[1].split(',')
                relation = dt.replace(').','').split('(')[0]

                if from_node not in relations_of_nodes:
                    relations_of_nodes[from_node] = {relation: 0}
                else:
                    if relation not in relations_of_nodes[from_node]:
                        relations_of_nodes[from_node][relation] = 0
                relations_of_nodes[from_node][relation] += 1

                if to_node not in relations_of_nodes:
                    relations_of_nodes[to_node] = {relation: 0}
                else:
                    if relation not in relations_of_nodes[to_node]:
                        relations_of_nodes[to_node][relation] = 0
                relations_of_nodes[to_node][relation] += 1
            #else:
            #    continue
                # If it is a unary relation, define a loop
            #    from_node = dt.replace(').', '').split('(')[1]
            #    to_node = from_node

            #    relation = dt.replace(').','').split('(')[0]

            #    if to_node not in relations_of_nodes:
            #        relations_of_nodes[from_node] = {relation: 0}
            #    else:
            #        if relation not in relations_of_nodes[to_node]:
            #            relations_of_nodes[to_node][relation] = 0
            #    relations_of_nodes[to_node][relation] += 1

            all_edges.append((from_node, to_node))

            if relation not in count_relations:
                count_relations[relation] = 0
            count_relations[relation] += 1

        g.add_edges_from(all_edges)

        print("Graph successfully created.")
        print("  It contains %d vertices and %d edges." % (g.number_of_nodes(), g.number_of_edges()))

        communities = list(nx.community.girvan_newman(g))

        fig, ax = plt.subplots(3, figsize=(15, 20))

        # Plot graph with colouring based on communities
        self.visualize_communities(g, communities[0], 1)
        self.visualize_communities(g, communities[3], 2)

        plt.show()

        return g, relations_of_nodes, count_relations

    # function to create node colour list
    def create_community_node_colors(self, graph, communities):
        number_of_colors = len(communities[0])
        colors = ["#D4FCB1", "#CDC5FC", "#FFC2C4", "#F2D140", "#BCC6C8"]*number_of_colors
        colors = colors[:number_of_colors]
        print(number_of_colors)
        KOSAKOPSKAPS
        node_colors = []
        for node in graph:
            current_community_index = 0
            for community in communities:
                if node in community:
                    node_colors.append(colors[current_community_index])
                    break
                current_community_index += 1
        return node_colors


    # function to plot graph with node colouring based on communities
    def visualize_communities(self, graph, communities, i):
        node_colors = self.create_community_node_colors(graph, communities)
        modularity = round(nx.community.modularity(graph, communities), 6)
        title = f"Community Visualization of {len(communities)} communities with modularity of {modularity}"
        pos = nx.spring_layout(graph, k=0.3, iterations=50, seed=2)
        plt.subplot(3, 1, i)
        plt.title(title)
        nx.draw(
            graph,
            pos=pos,
            node_size=1000,
            node_color=node_colors,
            with_labels=True,
            font_size=20,
            font_color="black",
        )       
