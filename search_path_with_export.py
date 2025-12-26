import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection
import numpy as np
import os
from sound_effects import init_sound_system, create_search_sounds, create_path_found_sound

# --- CONFIGURATION ---
CITY_POINT = (27.7172, 85.3240)   # Kathmandu, Nepal (city center)
DISTANCE = 2000                   # Search radius in meters
SEARCH_COLOR = '#1e90ff'          # Neon Blue
PATH_COLOR = '#adff2f'            # Neon Green
BG_COLOR = '#0b0b0b'              # Near black
OUTPUT_FILE = 'search_path_animation.mp4'

print("=" * 60)
print("Search & Path Visualization - WITH VIDEO EXPORT")
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
G = ox.graph_from_point(CITY_POINT, dist=DISTANCE, network_type='drive')

# --- NEW: Fetch Contextual Features (Buildings, Water, Parks) ---
print("    → Fetching buildings, water, and parks...")
tags = {
    'building': True,
    'natural': ['water', 'wood', 'tree_row', 'scrub'],
    'leisure': ['park', 'garden', 'pitch'],
    'waterway': True
}
try:
    features = ox.features_from_point(CITY_POINT, tags, dist=DISTANCE)
    # Separate features
    buildings = features[features['building'].notna()]
    water = features[features['natural'] == 'water']
    waterways = features[features['waterway'].notna()]
    parks = features[features['leisure'].isin(['park', 'garden'])]
    print(f"    ✓ Loaded {len(buildings)} buildings, {len(water)+len(waterways)} water features")
except Exception as e:
    print(f"    ⚠ Could not load features: {e}")
    buildings = water = waterways = parks = None

start_node = ox.distance.nearest_nodes(G, CITY_POINT[1], CITY_POINT[0])
end_node = ox.distance.nearest_nodes(G, 85.3500, 27.6800)

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
    
    explored_edges.append(coords)
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

print(f"    ✓ Final route: {len(route)} nodes, {len(route_coords)} coordinates")

# 4. Setup Plotting
print("\n[4/4] Setting up animation...")

# Create figure with TikTok aspect ratio (9:16 portrait)
fig_width = 9
fig_height = 16
fig = plt.figure(figsize=(fig_width, fig_height), dpi=100, facecolor=BG_COLOR)
ax = fig.add_subplot(111)
ax.set_facecolor(BG_COLOR)

# --- NEW: Plot Contextual Layers ---
if buildings is not None:
    # Plot water (Dark Blue-Grey)
    if not water.empty:
        water.plot(ax=ax, color='#1a2634', alpha=0.7, zorder=1)
    if not waterways.empty:
        waterways.plot(ax=ax, color='#1a2634', linewidth=2, alpha=0.7, zorder=1)
    
    # Plot parks (Dark Green-Grey)
    if not parks.empty:
        parks.plot(ax=ax, color='#1a332a', alpha=0.6, zorder=1)

    # Plot buildings (Dark Grey with slight purple tint)
    if not buildings.empty:
        buildings.plot(ax=ax, color='#1f1f24', alpha=0.5, zorder=2)

# Plot the graph edges (Roads) - Dimmer to let the search pop
ox.plot_graph(G, ax=ax, show=False, close=False, bgcolor=BG_COLOR, 
              edge_color='#2a2a2a', edge_linewidth=0.8, node_size=0)

# --- NEW: Add Grid Background ---
ax.grid(True, which='both', color='#1a1a1a', linestyle='-', linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

# --- NEW: Add Scanline Effect ---
for i in range(0, 100, 2):
    ax.axhline(y=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * i / 100, 
               color='white', alpha=0.03, linewidth=0.5, zorder=10)

# HUD Elements (Cyberpunk Text)
hud_font = {'family': 'monospace', 'weight': 'bold', 'size': 14}
stats_text = ax.text(0.05, 0.95, "SYSTEM: ONLINE", transform=ax.transAxes, 
                    color='#00ff00', alpha=0.8, ha='left', va='top', **hud_font)
scan_text = ax.text(0.05, 0.92, "SCANNED: 0", transform=ax.transAxes, 
                   color=SEARCH_COLOR, alpha=0.8, ha='left', va='top', **hud_font)
loc_text = ax.text(0.95, 0.02, "LOC: KATHMANDU", transform=ax.transAxes, 
                  color='white', alpha=0.5, ha='right', va='bottom', fontsize=10)

# Create LineCollection for search visualization
search_lc = LineCollection([], colors=SEARCH_COLOR, linewidths=1.5, alpha=0.8, zorder=4)
ax.add_collection(search_lc)

# Create multiple path lines for the glow effect
path_glow_outer, = ax.plot([], [], color=PATH_COLOR, linewidth=8, alpha=0.2, zorder=4)
path_glow_inner, = ax.plot([], [], color=PATH_COLOR, linewidth=5, alpha=0.4, zorder=4)
path_line, = ax.plot([], [], color=PATH_COLOR, linewidth=2.5, alpha=1, zorder=5)

# Add start and end markers
start_coords = (G.nodes[start_node]['x'], G.nodes[start_node]['y'])
end_coords = (G.nodes[end_node]['x'], G.nodes[end_node]['y'])

ax.scatter(*start_coords, color='#00ff00', s=150, marker='o', 
          edgecolors='#00ff00', linewidths=2, zorder=10, label='Start')
ax.scatter(*end_coords, color='#ff0000', s=150, marker='o', 
          edgecolors='#ff0000', linewidths=2, zorder=10, label='End')

ax.legend(loc='upper right', fancybox=True, shadow=True)
fig.suptitle('Dijkstra Search & Path Visualization', fontsize=14, color='white', 
             fontweight='bold', y=0.98)

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
        # Calculate how many edges to show based on progress
        progress = (frame + 1) / PHASE_1_FRAMES
        idx = int(progress * len(explored_edges))
        search_lc.set_segments(explored_edges[:idx])
        
        # Update HUD
        stats_text.set_text("STATUS: SEARCHING...")
        stats_text.set_color(SEARCH_COLOR)
        scan_text.set_text(f"SCANNED: {idx}")
        
        path_glow_outer.set_data([], [])
        path_glow_inner.set_data([], [])
        path_line.set_data([], [])
        
        # Play search sound every few frames (pinball bumper sound)
        if sound_enabled and frame % 5 == 0 and search_sounds:
            sound = search_sounds[frame % len(search_sounds)]
            sound.play()
    else:
        search_lc.set_segments(explored_edges)
        
        # Update HUD
        stats_text.set_text("STATUS: TARGET ACQUIRED")
        stats_text.set_color(PATH_COLOR)
        scan_text.set_text(f"SCANNED: {len(explored_edges)}")
        
        # Calculate fade-in effect
        fade_progress = (frame - PHASE_1_FRAMES) / (PHASE_2_FRAMES * 0.8)
        fade_progress = min(fade_progress, 1.0)
        
        # Play path found sound on first frame of phase 2
        if frame == PHASE_1_FRAMES and sound_enabled and path_found_sound:
            path_found_sound.play()
        
        path_x = [c[0] for c in route_coords]
        path_y = [c[1] for c in route_coords]
        
        path_glow_outer.set_data(path_x, path_y)
        path_glow_inner.set_data(path_x, path_y)
        path_line.set_data(path_x, path_y)
        
        path_glow_outer.set_alpha(0.2 * fade_progress)
        path_glow_inner.set_alpha(0.4 * fade_progress)
        path_line.set_alpha(fade_progress)
        
    return search_lc, path_glow_outer, path_glow_inner, path_line, stats_text, scan_text

ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000/FPS, blit=True)

print(f"    ✓ Animation ready: {TOTAL_FRAMES} frames")

# 6. Export to MP4 (requires FFmpeg)
print(f"\n[EXPORT] Saving animation to {OUTPUT_FILE}...")
try:
    writer = FFMpegWriter(fps=30, metadata=dict(artist='TikTok PF'), bitrate=1800)
    ani.save(OUTPUT_FILE, writer=writer, dpi=150)
    print(f"    ✓ Successfully saved to {OUTPUT_FILE}")
    print(f"    ✓ File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")
except Exception as e:
    print(f"    ✗ Error saving video: {e}")
    print(f"    → Make sure FFmpeg is installed: `apt install ffmpeg` (Linux) or `brew install ffmpeg` (Mac)")

print("\n" + "=" * 60)
