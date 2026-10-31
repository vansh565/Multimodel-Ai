import community as community_louvain

def detect_communities(G):
    partition = community_louvain.best_partition(G)

    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    return communities