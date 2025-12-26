import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from sound_effects import init_sound_system, create_search_sounds, create_path_found_sound

# --- CONFIGURATION ---
START_COORD = (27.7172, 85.3240)  # Kathmandu Center
END_COORD = (27.6800, 85.3500)    # Kathmandu South-East
BBOX_BUFFER = 0.005               # Buffer in degrees around the points
TILT_ANGLE = 45                   # Elevation angle for 3D
ROTATION_ANGLE = -45              # Azimuth angle for 3D

SEARCH_COLOR = '#1e90ff'          # Neon Blue
PATH_COLOR = '#adff2f'            # Neon Green
BG_COLOR = '#0b0b0b'              # Near black

print("=" * 60)
print("Search & Path Visualization - 3D Dijkstra Algorithm")
print("=" * 60)

# Initialize sound system
print("\n[0/4] Initializing sound system...")
sound_enabled = init_sound_system()
if sound_enabled:
    search_sounds = create_search_sounds()
    path_found_sound = create_path_found_sound()
    print("    ✓ Sound effects loaded")
else:
    print("    ⚠ Sound effects disabled")
    search_sounds = []
    path_found_sound = None

# 1. Load the map
print("\n[1/4] Downloading map data (this may take a moment)...")
north = max(START_COORD[0], END_COORD[0]) + BBOX_BUFFER
south = min(START_COORD[0], END_COORD[0]) - BBOX_BUFFER
east = max(START_COORD[1], END_COORD[1]) + BBOX_BUFFER
west = min(START_COORD[1], END_COORD[1]) - BBOX_BUFFER

G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type='drive')

# --- NEW: Fetch Contextual Features (Buildings, Water, Parks) ---
print("    → Fetching buildings, water, and parks...")
# Simplified tags to avoid potential type errors
tags = {'building': True}
try:
    features = ox.features_from_bbox(bbox=(north, south, east, west), tags=tags)
    # Separate features
    buildings = features[features['building'].notna()] if 'building' in features.columns else None
    water = None 
    waterways = None
    parks = None
    print(f"    ✓ Loaded features")
except Exception as e:
    print(f"    ⚠ Could not load features: {e}")
    buildings = water = waterways = parks = None

start_node = ox.distance.nearest_nodes(G, START_COORD[1], START_COORD[0])
end_node = ox.distance.nearest_nodes(G, END_COORD[1], END_COORD[0])

print(f"    ✓ Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
print(f"    ✓ Start node: {start_node}")
print(f"    ✓ End node: {end_node}")

# 2. Simulate the Search (Breadth-First Search for visualization)
print("\n[2/4] Calculating search expansion...")
explored_edges = []
for edge in nx.bfs_edges(G, source=start_node):
    u, v = edge
    data = G.get_edge_data(u, v)[0]
    if 'geometry' in data:
        coords = list(data['geometry'].coords)
    else:
        coords = [(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])]
    
    # Convert to 3D (x, y, z=0)
    coords_3d = np.array([(c[0], c[1], 0) for c in coords])
    explored_edges.append(coords_3d)
    if v == end_node: 
        break

print(f"    ✓ Explored {len(explored_edges)} edges during search")

# 3. Get the Final Shortest Path
print("\n[3/4] Computing shortest path...")
route = nx.shortest_path(G, start_node, end_node, weight='length')
route_coords = []
for u, v in zip(route[:-1], route[1:]):
    data = G.get_edge_data(u, v)[0]
    if 'geometry' in data:
        route_coords.extend(list(data['geometry'].coords))
    else:
        route_coords.extend([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])

# Convert to 3D
route_coords_3d = np.array([(c[0], c[1], 0.001) for c in route_coords]) # Slightly above ground

print(f"    ✓ Final route: {len(route)} nodes, {len(route_coords)} coordinates")

# 4. Setup Plotting
print("\n[4/4] Setting up animation...")

# Create figure with TikTok aspect ratio (9:16 portrait)
fig_width = 9
fig_height = 16
fig = plt.figure(figsize=(fig_width, fig_height), dpi=100, facecolor=BG_COLOR)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor(BG_COLOR)

# Set 3D view
ax.view_init(elev=TILT_ANGLE, azim=ROTATION_ANGLE)

# Hide axes for clean look
ax.set_axis_off()

# Set limits based on bbox
ax.set_xlim(west, east)
ax.set_ylim(south, north)
ax.set_zlim(0, 0.01)

# --- Plot Contextual Layers in 3D ---
def plot_polygon_3d(ax, poly, color, alpha, z=0):
    try:
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            verts = [list(zip(x, y, [z]*len(x)))]
            poly_coll = Poly3DCollection(verts, color=color, alpha=alpha)
            ax.add_collection(poly_coll)
        elif poly.geom_type == 'MultiPolygon':
            for p in poly.geoms:
                x, y = p.exterior.xy
                verts = [list(zip(x, y, [z]*len(x)))]
                poly_coll = Poly3DCollection(verts, color=color, alpha=alpha)
                ax.add_collection(poly_coll)
    except:
        pass

if buildings is not None:
    print("    → Rendering 3D features...")
    # Plot water
    if water is not None and not water.empty:
        for poly in water.geometry:
            plot_polygon_3d(ax, poly, '#1a2634', 0.7, z=0)
    
    # Plot buildings (with height simulation)
    if buildings is not None and not buildings.empty:
        # Only plot a subset if too many for performance
        for poly in buildings.geometry[:1000]:
            plot_polygon_3d(ax, poly, '#1f1f24', 0.5, z=0)

# Plot the graph edges (Roads)
road_lines = []
for u, v, data in G.edges(data=True):
    if 'geometry' in data:
        xs, ys = data['geometry'].xy
        road_lines.append(np.array(list(zip(xs, ys, [0]*len(xs)))))
    else:
        road_lines.append(np.array([(G.nodes[u]['x'], G.nodes[u]['y'], 0), (G.nodes[v]['x'], G.nodes[v]['y'], 0)]))

road_coll = Line3DCollection(road_lines, colors='#2a2a2a', linewidths=0.8, alpha=0.5)
ax.add_collection(road_coll)

# HUD Elements (Cyberpunk Text) - Using 2D text on 3D axis
hud_font = {'family': 'monospace', 'weight': 'bold', 'size': 14}
stats_text = fig.text(0.05, 0.95, "SYSTEM: ONLINE", color='#00ff00', alpha=0.8, ha='left', va='top', **hud_font)
scan_text = fig.text(0.05, 0.92, "SCANNED: 0", color=SEARCH_COLOR, alpha=0.8, ha='left', va='top', **hud_font)
loc_text = fig.text(0.95, 0.02, "LOC: KATHMANDU", color='white', alpha=0.5, ha='right', va='bottom', fontsize=10)

# Create Line3DCollection for search visualization
search_lc = Line3DCollection([], colors=SEARCH_COLOR, linewidths=1.5, alpha=0.8)
ax.add_collection(search_lc)

# Create path lines for the glow effect
path_glow_outer = Line3DCollection([], colors=PATH_COLOR, linewidths=8, alpha=0.2)
path_glow_inner = Line3DCollection([], colors=PATH_COLOR, linewidths=5, alpha=0.4)
path_line = Line3DCollection([], colors=PATH_COLOR, linewidths=2.5, alpha=1)

ax.add_collection(path_glow_outer)
ax.add_collection(path_glow_inner)
ax.add_collection(path_line)

# Set limits based on bbox
ax.set_xlim(west, east)
ax.set_ylim(south, north)
ax.set_zlim(0, 0.02)

# 5. Animation Function
# --- TIMING CONFIGURATION ---
TARGET_DURATION = 10  # Total video duration in seconds
FPS = 30              # Frames per second
TOTAL_FRAMES = TARGET_DURATION * FPS
PHASE_2_FRAMES = 60   # 2 seconds for path highlight/fade-in
PHASE_1_FRAMES = TOTAL_FRAMES - PHASE_2_FRAMES # Remaining frames for search

def update(frame):
    # Phase 1: Growing the Blue Web
    if frame < PHASE_1_FRAMES:
        progress = (frame + 1) / PHASE_1_FRAMES
        idx = int(progress * len(explored_edges))
        search_lc.set_segments(explored_edges[:idx])
        
        # Update HUD
        stats_text.set_text("STATUS: SEARCHING...")
        stats_text.set_color(SEARCH_COLOR)
        scan_text.set_text(f"SCANNED: {idx}")
        
        path_glow_outer.set_segments([])
        path_glow_inner.set_segments([])
        path_line.set_segments([])
        
        # Play search sound every few frames
        if sound_enabled and frame % 5 == 0 and search_sounds:
            sound = search_sounds[frame % len(search_sounds)]
            sound.play()
            
    # Phase 2: Highlighting the Green Route
    else:
        search_lc.set_segments(explored_edges)
        
        # Update HUD
        stats_text.set_text("STATUS: TARGET ACQUIRED")
        stats_text.set_color(PATH_COLOR)
        scan_text.set_text(f"SCANNED: {len(explored_edges)}")
        
        fade_progress = (frame - PHASE_1_FRAMES) / (PHASE_2_FRAMES * 0.8)
        fade_progress = min(fade_progress, 1.0)
        
        if frame == PHASE_1_FRAMES and sound_enabled and path_found_sound:
            path_found_sound.play()
        
        # Animate the path
        # For Line3DCollection, we need a list of segments
        path_segments = [route_coords_3d]
        path_glow_outer.set_segments(path_segments)
        path_glow_inner.set_segments(path_segments)
        path_line.set_segments(path_segments)
        
        path_glow_outer.set_alpha(0.2 * fade_progress)
        path_glow_inner.set_alpha(0.4 * fade_progress)
        path_line.set_alpha(fade_progress)
        
    return search_lc, path_glow_outer, path_glow_inner, path_line, stats_text, scan_text

ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000/FPS, blit=False)

print(f"    ✓ Animation ready: {TOTAL_FRAMES} frames")
print(f"\nStarting visualization...")
print("Close the window to exit.\n")

plt.show()

print("\nAnimation complete!")
print("=" * 60)

