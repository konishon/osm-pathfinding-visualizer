import heapq
import networkx as nx

def heuristic(G, u, v):
    """Heuristic function for A* (Euclidean distance)."""
    return ((G.nodes[u]['x'] - G.nodes[v]['x']) ** 2 + 
            (G.nodes[u]['y'] - G.nodes[v]['y']) ** 2) ** 0.5

def astar_traversal(G, start_node, end_node):
    """Yield edges as they are explored by A*."""
    count = 0
    open_set = []
    heapq.heappush(open_set, (0, count, start_node))
    
    g_score = {start_node: 0}
    visited = set()
    
    while open_set:
        _, _, current = heapq.heappop(open_set)
        
        if current == end_node:
            break
        
        if current in visited:
            continue
        visited.add(current)
            
        for neighbor in G.neighbors(current):
            edge_data = G.get_edge_data(current, neighbor)[0]
            weight = edge_data.get('length', 1)
            
            tentative_g_score = g_score[current] + weight
            
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(G, neighbor, end_node)
                count += 1
                heapq.heappush(open_set, (f_score, count, neighbor))
                yield (current, neighbor)

def dijkstra_traversal(G, start_node, end_node):
    """Yield edges as they are explored by Dijkstra's algorithm."""
    count = 0
    open_set = []
    heapq.heappush(open_set, (0, count, start_node))
    
    g_score = {start_node: 0}
    visited = set()
    
    while open_set:
        _, _, current = heapq.heappop(open_set)
        
        if current == end_node:
            break
        
        if current in visited:
            continue
        visited.add(current)
            
        for neighbor in G.neighbors(current):
            edge_data = G.get_edge_data(current, neighbor)[0]
            weight = edge_data.get('length', 1)
            
            tentative_g_score = g_score[current] + weight
            
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g_score
                count += 1
                heapq.heappush(open_set, (tentative_g_score, count, neighbor))
                yield (current, neighbor)

def greedy_bfs_traversal(G, start_node, end_node):
    """Yield edges as they are explored by Greedy Best-First Search."""
    count = 0
    open_set = []
    h_start = heuristic(G, start_node, end_node)
    heapq.heappush(open_set, (h_start, count, start_node))
    
    visited = set()
    visited.add(start_node)
    
    while open_set:
        _, _, current = heapq.heappop(open_set)
        
        if current == end_node:
            break
            
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                h_score = heuristic(G, neighbor, end_node)
                count += 1
                heapq.heappush(open_set, (h_score, count, neighbor))
                yield (current, neighbor)

def bfs_traversal(G, start_node, end_node=None):
    """Yield edges as they are explored by BFS."""
    # end_node is ignored for standard BFS but kept for signature consistency
    for edge in nx.bfs_edges(G, source=start_node):
        yield edge
        if end_node and edge[1] == end_node:
            break

ALGORITHMS = {
    'bfs': bfs_traversal,
    'astar': astar_traversal,
    'dijkstra': dijkstra_traversal,
    'greedy': greedy_bfs_traversal
}
