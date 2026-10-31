import networkx as nx
from .entity_extractor import extract_entities

def build_graph(chunks):
    G = nx.Graph()

    for chunk in chunks:
        entities = extract_entities(chunk)

        for e in entities:
            G.add_node(e)

        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                G.add_edge(entities[i], entities[j], text=chunk)

    return G