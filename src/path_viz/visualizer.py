import argparse
import sys
import os
import pickle
import numpy as np
import osmnx as ox
import networkx as nx
import wave
import subprocess
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import pyproj
from shapely.geometry import Point

try:
    from .sound_effects import init_sound_system, create_search_sounds, create_path_found_sound, get_search_waveforms, get_path_found_waveform
except ImportError:
    # Fallback for direct execution or if module not found
    try:
        from sound_effects import init_sound_system, create_search_sounds, create_path_found_sound, get_search_waveforms, get_path_found_waveform
    except ImportError:
        print("Warning: sound_effects module not found. Audio will be disabled.")
        def init_sound_system(): return False
        def create_search_sounds(*args, **kwargs): return []
        def create_path_found_sound(): return None
        def get_search_waveforms(*args, **kwargs): return []
        def get_path_found_waveform(): return np.array([])

@dataclass
class Config:
    """Configuration for the visualization."""
    start_coord: Tuple[float, float] = (27.7172, 85.3240)  # Kathmandu Center
    end_coord: Tuple[float, float] = (27.6800, 85.3500)    # Kathmandu South-East
    bbox_buffer: float = 0.025     # Increased buffer to ensure map fills the frame (was 0.005)
    tilt_angle: float = 45
    rotation_angle: float = -45
    search_color: str = '#1e90ff'  # Neon Blue
    path_color: str = '#adff2f'    # Neon Green
    bg_color: str = '#0b0b0b'      # Near black
    duration: int = 10
    fps: int = 30
    cache_dir: str = 'cache_data'
    dimension: str = '3d'
    algorithm: str = 'bfs'
    
    @property
    def total_frames(self):
        return self.duration * self.fps
    
    @property
    def phase_2_frames(self):
        return 60  # 2 seconds for path highlight
        
    @property
    def phase_1_frames(self):
        return self.total_frames - self.phase_2_frames

class PathVisualizer:
    def __init__(self, config: Config):
        self.config = config
        self.G = None
        self.features = {}
        self.explored_edges = []
        self.route_coords_3d = None
        self.fig = None
        self.ax = None
        self.collections = {}
        self.texts = {}
        self.sounds = {'search': [], 'found': None}
        self.waveforms = {'search': [], 'found': None}
        self.sound_enabled = False
        self.mode = 'view'
        self.audio_buffer = None
        self.sample_rate = 22050
        
        # Calculate center
        cy = (config.start_coord[0] + config.end_coord[0]) / 2
        cx = (config.start_coord[1] + config.end_coord[1]) / 2
        
        # Calculate raw spans required to cover the path + buffer
        dy = abs(config.start_coord[0] - config.end_coord[0]) + 2 * config.bbox_buffer
        dx = abs(config.start_coord[1] - config.end_coord[1]) + 2 * config.bbox_buffer
        
        # Enforce 9:16 aspect ratio for the map data
        # We want the map area to be proportional to 9:16 so it fills the screen
        target_ratio = 9 / 16
        current_ratio = dx / dy
        
        if current_ratio > target_ratio:
            # Too wide, need to increase height
            dy = dx / target_ratio
        else:
            # Too tall, need to increase width
            dx = dy * target_ratio
            
        # Set boundaries
        self.north = cy + dy / 2
        self.south = cy - dy / 2
        self.east = cx + dx / 2
        self.west = cx - dx / 2
        
        # Ensure cache directory exists
        os.makedirs(config.cache_dir, exist_ok=True)

    def init_audio(self):
        """Initialize audio system."""
        print("[0/5] Initializing sound system...")
        
        # Calculate number of search beeps needed
        # We play a beep every 5 frames during phase 1
        num_beeps = int(self.config.phase_1_frames / 5) + 2  # Add buffer
        
        if self.mode == 'view':
            self.sound_enabled = init_sound_system()
            if self.sound_enabled:
                self.sounds['search'] = create_search_sounds(num_steps=num_beeps)
                self.sounds['found'] = create_path_found_sound()
                print("    ✓ Sound effects loaded")
            else:
                print("    ⚠ Sound effects disabled")
        elif self.mode == 'export':
            # Load waveforms for export
            self.waveforms['search'] = get_search_waveforms(num_steps=num_beeps)
            self.waveforms['found'] = get_path_found_waveform()
            
            # Initialize audio buffer
            total_samples = int(self.config.total_frames / self.config.fps * self.sample_rate)
            self.audio_buffer = np.zeros(total_samples, dtype=np.float32)
            print("    ✓ Audio waveforms loaded for export")

    def add_sound_to_buffer(self, waveform, frame_idx):
        """Mix a sound into the audio buffer at the given frame."""
        if self.audio_buffer is None:
            return
            
        start_sample = int(frame_idx / self.config.fps * self.sample_rate)
        end_sample = start_sample + len(waveform)
        
        # Handle boundary conditions
        if start_sample >= len(self.audio_buffer):
            return
            
        if end_sample > len(self.audio_buffer):
            # Truncate waveform
            waveform = waveform[:len(self.audio_buffer) - start_sample]
            end_sample = len(self.audio_buffer)
            
        # Mix (add) the sound
        # Convert int16 waveform to float for mixing
        self.audio_buffer[start_sample:end_sample] += waveform.astype(np.float32) / 32767.0

    def save_audio(self, filename):
        """Save the accumulated audio buffer to a WAV file."""
        if self.audio_buffer is None:
            return
            
        # Normalize and clip
        max_val = np.max(np.abs(self.audio_buffer))
        if max_val > 1.0:
            self.audio_buffer /= max_val
            
        # Convert back to int16
        audio_int16 = (self.audio_buffer * 32767).astype(np.int16)
        
        print(f"    → Saving audio track to {filename}...")
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono for simplicity
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        print(f"    ✓ Audio saved")

    def load_data(self):
        """Download and prepare map data with caching."""
        print("\n[1/5] Loading map data...")
        
        # Generate cache keys based on bbox
        bbox_str = f"{self.north:.4f}_{self.south:.4f}_{self.east:.4f}_{self.west:.4f}"
        graph_cache_path = os.path.join(self.config.cache_dir, f"graph_{bbox_str}.graphml")
        features_cache_path = os.path.join(self.config.cache_dir, f"features_{bbox_str}.pkl")
        
        # 1. Load Graph
        if os.path.exists(graph_cache_path):
            print(f"    → Loading graph from cache: {graph_cache_path}")
            self.G = ox.load_graphml(graph_cache_path)
        else:
            print("    → Downloading graph from OSM...")
            self.G = ox.graph_from_bbox(
                bbox=(self.north, self.south, self.east, self.west), 
                network_type='drive'
            )
            print(f"    → Saving graph to cache...")
            ox.save_graphml(self.G, graph_cache_path)
            
        print(f"    ✓ Graph loaded: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")

        # Project graph to UTM (Meters)
        print("    → Projecting graph to UTM...")
        self.G = ox.project_graph(self.G)
        
        # Update bounds based on projected graph
        nodes = ox.graph_to_gdfs(self.G, edges=False)
        self.west = nodes.geometry.x.min()
        self.east = nodes.geometry.x.max()
        self.south = nodes.geometry.y.min()
        self.north = nodes.geometry.y.max()
        
        # Project start/end coordinates
        crs = self.G.graph['crs']
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        
        # Config coords are (Lat, Lon) -> (Y, X)
        # Transformer expects (Lon, Lat) -> (X, Y)
        start_lon, start_lat = self.config.start_coord[1], self.config.start_coord[0]
        end_lon, end_lat = self.config.end_coord[1], self.config.end_coord[0]
        
        self.start_x, self.start_y = transformer.transform(start_lon, start_lat)
        self.end_x, self.end_y = transformer.transform(end_lon, end_lat)

        # 2. Load Features
        if os.path.exists(features_cache_path):
            print(f"    → Loading features from cache: {features_cache_path}")
            with open(features_cache_path, 'rb') as f:
                self.features = pickle.load(f)
            print(f"    ✓ Loaded features")
        else:
            print("    → Fetching buildings from OSM...")
            tags = {'building': True}
            try:
                # Note: features_from_bbox uses Lat/Lon even if we want projected later
                # We use the original Lat/Lon bounds from __init__ (stored in config or recalculated?)
                # Wait, self.north/south/east/west were overwritten above!
                # We need to use the original Lat/Lon bounds for fetching features if not cached.
                # But we already overwrote them.
                # Let's recalculate original bounds for fetching.
                
                cy = (self.config.start_coord[0] + self.config.end_coord[0]) / 2
                cx = (self.config.start_coord[1] + self.config.end_coord[1]) / 2
                dy = abs(self.config.start_coord[0] - self.config.end_coord[0]) + 2 * self.config.bbox_buffer
                dx = abs(self.config.start_coord[1] - self.config.end_coord[1]) + 2 * self.config.bbox_buffer
                
                # Re-apply aspect ratio logic for fetching bounds
                target_ratio = 9 / 16
                current_ratio = dx / dy
                if current_ratio > target_ratio:
                    dy = dx / target_ratio
                else:
                    dx = dy * target_ratio
                
                orig_north = cy + dy / 2
                orig_south = cy - dy / 2
                orig_east = cx + dx / 2
                orig_west = cx - dx / 2

                gdf = ox.features_from_bbox(bbox=(orig_north, orig_south, orig_east, orig_west), tags=tags)
                if 'building' in gdf.columns:
                    self.features['buildings'] = gdf[gdf['building'].notna()]
                
                print(f"    → Saving features to cache...")
                with open(features_cache_path, 'wb') as f:
                    pickle.dump(self.features, f)
                print(f"    ✓ Loaded features")
            except Exception as e:
                print(f"    ⚠ Could not load features: {e}")

        # Project features to match graph
        if 'buildings' in self.features:
             self.features['buildings'] = self.features['buildings'].to_crs(crs)

    def _heuristic(self, u, v):
        """Heuristic function for A* (Euclidean distance)."""
        return ((self.G.nodes[u]['x'] - self.G.nodes[v]['x']) ** 2 + 
                (self.G.nodes[u]['y'] - self.G.nodes[v]['y']) ** 2) ** 0.5

    def _astar_traversal(self, start_node, end_node):
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
                
            for neighbor in self.G.neighbors(current):
                edge_data = self.G.get_edge_data(current, neighbor)[0]
                weight = edge_data.get('length', 1)
                
                tentative_g_score = g_score[current] + weight
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor, end_node)
                    count += 1
                    heapq.heappush(open_set, (f_score, count, neighbor))
                    yield (current, neighbor)

    def _dijkstra_traversal(self, start_node, end_node):
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
                
            for neighbor in self.G.neighbors(current):
                edge_data = self.G.get_edge_data(current, neighbor)[0]
                weight = edge_data.get('length', 1)
                
                tentative_g_score = g_score[current] + weight
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g_score
                    count += 1
                    heapq.heappush(open_set, (tentative_g_score, count, neighbor))
                    yield (current, neighbor)

    def _greedy_bfs_traversal(self, start_node, end_node):
        """Yield edges as they are explored by Greedy Best-First Search."""
        count = 0
        open_set = []
        # Priority is only heuristic
        h_start = self._heuristic(start_node, end_node)
        heapq.heappush(open_set, (h_start, count, start_node))
        
        visited = set()
        visited.add(start_node)
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current == end_node:
                break
                
            for neighbor in self.G.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    h_score = self._heuristic(neighbor, end_node)
                    count += 1
                    heapq.heappush(open_set, (h_score, count, neighbor))
                    yield (current, neighbor)

    def run_search(self):
        """Run Search and Shortest Path algorithms."""
        print(f"\n[2/5] Calculating search expansion ({self.config.algorithm.upper()})...")
        
        # Use projected coordinates
        start_node = ox.distance.nearest_nodes(self.G, self.start_x, self.start_y)
        end_node = ox.distance.nearest_nodes(self.G, self.end_x, self.end_y)
        
        # Search Algorithm
        if self.config.algorithm == 'astar':
            iterator = self._astar_traversal(start_node, end_node)
        elif self.config.algorithm == 'dijkstra':
            iterator = self._dijkstra_traversal(start_node, end_node)
        elif self.config.algorithm == 'greedy':
            iterator = self._greedy_bfs_traversal(start_node, end_node)
        else:
            iterator = nx.bfs_edges(self.G, source=start_node)

        for edge in iterator:
            u, v = edge
            data = self.G.get_edge_data(u, v)[0]
            if 'geometry' in data:
                coords = list(data['geometry'].coords)
            else:
                coords = [(self.G.nodes[u]['x'], self.G.nodes[u]['y']), 
                          (self.G.nodes[v]['x'], self.G.nodes[v]['y'])]
            
            # Convert to 3D numpy array (N, 3)
            # Search layer below roads (z=-100)
            if len(coords) >= 2:
                if self.config.dimension == '3d':
                    coords_3d = np.array([(c[0], c[1], -100) for c in coords])
                else:
                    coords_3d = np.array([(c[0], c[1]) for c in coords])
                self.explored_edges.append(coords_3d)
            
            if v == end_node:
                break
        
        print(f"    ✓ Explored {len(self.explored_edges)} edges")

        # Shortest Path
        print("\n[3/5] Computing shortest path...")
        try:
            route = nx.shortest_path(self.G, start_node, end_node, weight='length')
            route_coords = []
            for u, v in zip(route[:-1], route[1:]):
                data = self.G.get_edge_data(u, v)[0]
                if 'geometry' in data:
                    route_coords.extend(list(data['geometry'].coords))
                else:
                    route_coords.extend([(self.G.nodes[u]['x'], self.G.nodes[u]['y']), 
                                       (self.G.nodes[v]['x'], self.G.nodes[v]['y'])])
            
            # Convert to 3D numpy array (N, 3) with lower Z offset (z=-200)
            if len(route_coords) >= 2:
                if self.config.dimension == '3d':
                    self.route_coords_3d = np.array([(c[0], c[1], -200) for c in route_coords])
                else:
                    self.route_coords_3d = np.array([(c[0], c[1]) for c in route_coords])
            else:
                self.route_coords_3d = np.empty((0, 3 if self.config.dimension == '3d' else 2))
            print(f"    ✓ Final route: {len(route)} nodes")
        except nx.NetworkXNoPath:
            print("    ⚠ NO PATH FOUND! Check coordinates.")
            self.route_coords_3d = np.empty((0, 3 if self.config.dimension == '3d' else 2))

    def _plot_polygon_3d(self, poly, color, alpha, z=-0.0001):
        """Helper to plot 3D polygons."""
        try:
            if poly.geom_type == 'Polygon':
                x, y = poly.exterior.xy
                verts = [list(zip(x, y, [z]*len(x)))]
                poly_coll = Poly3DCollection(verts, color=color, alpha=alpha, zorder=1)
                self.ax.add_collection(poly_coll)
            elif poly.geom_type == 'MultiPolygon':
                for p in poly.geoms:
                    x, y = p.exterior.xy
                    verts = [list(zip(x, y, [z]*len(x)))]
                    poly_coll = Poly3DCollection(verts, color=color, alpha=alpha, zorder=1)
                    self.ax.add_collection(poly_coll)
        except Exception:
            pass

    def setup_scene(self):
        """Setup the 3D scene and static elements."""
        print("\n[4/5] Setting up scene...")
        
        # Figure setup
        self.fig = plt.figure(figsize=(9, 16), dpi=100, facecolor=self.config.bg_color)
        
        if self.config.dimension == '3d':
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.view_init(elev=self.config.tilt_angle, azim=self.config.rotation_angle)
            self.ax.set_zlim(-500, 100)
            self.ax.set_box_aspect((9, 16, 5))
            self.ax.dist = 6
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_aspect('equal')
            
        self.ax.set_facecolor(self.config.bg_color)
        self.ax.set_axis_off()
        
        # Set limits
        self.ax.set_xlim(self.west, self.east)
        self.ax.set_ylim(self.south, self.north)

        # Render Buildings
        # if 'buildings' in self.features:
        #     print("    → Rendering buildings...")
        #     for poly in self.features['buildings'].geometry[:1000]:  # Limit for performance
        #         self._plot_polygon_3d(poly, '#1f1f24', 0.5, z=-50)

        # Render Roads
        print("    → Rendering roads...")
        road_lines = []
        for u, v, data in self.G.edges(data=True):
            if 'geometry' in data:
                xs, ys = data['geometry'].xy
                # Ensure homogeneous shape (N, 3)
                if len(xs) >= 2:
                    if self.config.dimension == '3d':
                        points = np.column_stack((xs, ys, np.zeros(len(xs))))
                    else:
                        points = np.column_stack((xs, ys))
                    road_lines.append(points)
            else:
                if self.config.dimension == '3d':
                    p1 = (self.G.nodes[u]['x'], self.G.nodes[u]['y'], 0)
                    p2 = (self.G.nodes[v]['x'], self.G.nodes[v]['y'], 0)
                else:
                    p1 = (self.G.nodes[u]['x'], self.G.nodes[u]['y'])
                    p2 = (self.G.nodes[v]['x'], self.G.nodes[v]['y'])
                road_lines.append(np.array([p1, p2]))
        
        # Create collection manually to avoid "inhomogeneous shape" error
        # Line3DCollection expects a list of (N, 3) arrays
        # Increased visibility for roads to ensure they fill the video
        if self.config.dimension == '3d':
            road_coll = Line3DCollection(road_lines, colors='#555555', linewidths=1.0, alpha=0.8, zorder=5)
        else:
            road_coll = LineCollection(road_lines, colors='#555555', linewidths=1.0, alpha=0.8, zorder=1)
        self.ax.add_collection(road_coll)

        # Markers (Start/End)
        start_node = ox.distance.nearest_nodes(self.G, self.start_x, self.start_y)
        end_node = ox.distance.nearest_nodes(self.G, self.end_x, self.end_y)
        
        sx, sy = self.G.nodes[start_node]['x'], self.G.nodes[start_node]['y']
        ex, ey = self.G.nodes[end_node]['x'], self.G.nodes[end_node]['y']
        
        # Plot markers (using scatter for 3D points)
        if self.config.dimension == '3d':
            # Z=-300 to sit below everything
            self.ax.scatter([sx], [sy], [-300], c='#00ff00', s=200, label='Start', edgecolors='white', alpha=1.0, zorder=2)
            self.ax.scatter([ex], [ey], [-300], c='#ff0000', s=200, label='End', edgecolors='white', alpha=1.0, zorder=2)
        else:
            self.ax.scatter([sx], [sy], c='#00ff00', s=200, label='Start', edgecolors='white', alpha=1.0, zorder=5)
            self.ax.scatter([ex], [ey], c='#ff0000', s=200, label='End', edgecolors='white', alpha=1.0, zorder=5)

        # Dynamic Elements (Search & Path)
        # Search lines
        if self.config.dimension == '3d':
            self.collections['search'] = Line3DCollection([], colors=self.config.search_color, linewidths=1.5, alpha=0.8, zorder=4)
        else:
            self.collections['search'] = LineCollection([], colors=self.config.search_color, linewidths=1.5, alpha=0.8, zorder=2)
        self.ax.add_collection(self.collections['search'])

        # Path lines
        if self.config.dimension == '3d':
            self.collections['path_glow_outer'] = Line3DCollection([], colors=self.config.path_color, linewidths=8, alpha=0.2, zorder=3)
            self.collections['path_glow_inner'] = Line3DCollection([], colors=self.config.path_color, linewidths=5, alpha=0.4, zorder=3)
            self.collections['path_line'] = Line3DCollection([], colors=self.config.path_color, linewidths=2.5, alpha=1, zorder=3)
        else:
            self.collections['path_glow_outer'] = LineCollection([], colors=self.config.path_color, linewidths=8, alpha=0.2, zorder=3)
            self.collections['path_glow_inner'] = LineCollection([], colors=self.config.path_color, linewidths=5, alpha=0.4, zorder=3)
            self.collections['path_line'] = LineCollection([], colors=self.config.path_color, linewidths=2.5, alpha=1, zorder=4)
        
        self.ax.add_collection(self.collections['path_glow_outer'])
        self.ax.add_collection(self.collections['path_glow_inner'])
        self.ax.add_collection(self.collections['path_line'])

        # HUD
        hud_font = {'family': 'monospace', 'weight': 'bold', 'size': 14}
        self.texts['status'] = self.fig.text(0.05, 0.95, "SYSTEM: ONLINE", color='#00ff00', alpha=0.8, ha='left', va='top', **hud_font)
        self.texts['scan'] = self.fig.text(0.05, 0.92, "SCANNED: 0", color=self.config.search_color, alpha=0.8, ha='left', va='top', **hud_font)
        self.fig.text(0.95, 0.02, "LOC: KATHMANDU", color='white', alpha=0.5, ha='right', va='bottom', fontsize=10)

    def update_frame(self, frame):
        """Animation update function."""
        # Phase 1: Search
        if frame < self.config.phase_1_frames:
            # Calculate progress with "Fast In, Slow Out" interpolation (Cubic Ease Out)
            t = (frame + 1) / self.config.phase_1_frames
            progress = 1 - (1 - t) ** 3
            
            idx = int(progress * len(self.explored_edges))
            
            # Update search web
            current_edges = self.explored_edges[:idx]
            self.collections['search'].set_segments(current_edges)
            
            # Update HUD
            self.texts['status'].set_text("STATUS: SEARCHING...")
            self.texts['status'].set_color(self.config.search_color)
            self.texts['scan'].set_text(f"SCANNED: {idx}")
            
            # Audio
            if frame % 5 == 0:
                if self.mode == 'view' and self.sound_enabled and self.sounds['search']:
                    sound_idx = min(int(progress * len(self.sounds['search'])), len(self.sounds['search']) - 1)
                    sound = self.sounds['search'][sound_idx]
                    sound.play()
                elif self.mode == 'export' and self.waveforms['search']:
                    sound_idx = min(int(progress * len(self.waveforms['search'])), len(self.waveforms['search']) - 1)
                    waveform = self.waveforms['search'][sound_idx]
                    self.add_sound_to_buffer(waveform, frame)
                
        # Phase 2: Path Found
        else:
            self.collections['search'].set_segments(self.explored_edges)
            
            # Update HUD
            self.texts['status'].set_text("STATUS: TARGET ACQUIRED")
            self.texts['status'].set_color(self.config.path_color)
            self.texts['scan'].set_text(f"SCANNED: {len(self.explored_edges)}")
            
            # Fade in path
            fade_progress = (frame - self.config.phase_1_frames) / (self.config.phase_2_frames * 0.8)
            fade_progress = min(fade_progress, 1.0)
            
            path_segments = [self.route_coords_3d]
            self.collections['path_glow_outer'].set_segments(path_segments)
            self.collections['path_glow_inner'].set_segments(path_segments)
            self.collections['path_line'].set_segments(path_segments)
            
            self.collections['path_glow_outer'].set_alpha(0.2 * fade_progress)
            self.collections['path_glow_inner'].set_alpha(0.4 * fade_progress)
            self.collections['path_line'].set_alpha(fade_progress)
            
            # Play found sound once
            if frame == self.config.phase_1_frames:
                if self.mode == 'view' and self.sound_enabled and self.sounds['found']:
                    self.sounds['found'].play()
                elif self.mode == 'export' and self.waveforms['found'] is not None:
                    self.add_sound_to_buffer(self.waveforms['found'], frame)

        return (self.collections['search'], self.collections['path_line'], 
                self.texts['status'], self.texts['scan'])

    def run(self, mode='view', output_file='output.mp4'):
        """Execute the visualization."""
        self.mode = mode
        self.init_audio()
        self.load_data()
        self.run_search()
        self.setup_scene()
        
        if mode == 'preview':
            print("\n[5/5] Showing preview (static)...")
            # Show full search and path for preview
            self.collections['search'].set_segments(self.explored_edges)
            path_segments = [self.route_coords_3d]
            self.collections['path_glow_outer'].set_segments(path_segments)
            self.collections['path_glow_inner'].set_segments(path_segments)
            self.collections['path_line'].set_segments(path_segments)
            
            # Update HUD for preview
            self.texts['status'].set_text("STATUS: PREVIEW MODE")
            self.texts['status'].set_color(self.config.path_color)
            self.texts['scan'].set_text(f"SCANNED: {len(self.explored_edges)}")
            
            plt.show()
            return

        print(f"\n[5/5] Starting animation ({mode} mode)...")
        ani = FuncAnimation(
            self.fig, 
            self.update_frame, 
            frames=self.config.total_frames, 
            interval=1000/self.config.fps, 
            blit=False
        )
        
        if mode == 'view':
            plt.show()
        elif mode == 'export':
            print(f"    → Rendering to {output_file}...")
            writer = FFMpegWriter(fps=self.config.fps, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(output_file, writer=writer)
            print(f"    ✓ Video saved to {output_file}")
            
            # Save audio and merge
            audio_file = output_file.replace('.mp4', '.wav')
            self.save_audio(audio_file)
            
            # Merge using ffmpeg
            final_output = output_file.replace('.mp4', '_with_audio.mp4')
            print(f"    → Merging audio and video to {final_output}...")
            try:
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', output_file,
                    '-i', audio_file,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-strict', 'experimental',
                    final_output
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"    ✓ Final video saved: {final_output}")
                
                # Cleanup
                os.remove(audio_file)
                # Optionally replace original
                os.replace(final_output, output_file)
                print(f"    ✓ Replaced original file")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"    ⚠ Could not merge audio: {e}")
                print(f"    ⚠ Audio saved separately as {audio_file}")

def main():
    parser = argparse.ArgumentParser(description="3D Pathfinding Visualization")
    parser.add_argument('--mode', choices=['view', 'export', 'preview'], default='view', help="Run mode: 'view' for interactive, 'export' for video file, 'preview' for static check")
    parser.add_argument('--output', default='path_viz.mp4', help="Output filename for export mode")
    parser.add_argument('--dim', choices=['2d', '3d'], default='3d', help="Dimension mode: '2d' or '3d'")
    parser.add_argument('--algo', choices=['bfs', 'astar', 'dijkstra', 'greedy'], default='bfs', help="Search algorithm: 'bfs', 'astar', 'dijkstra', 'greedy'")
    args = parser.parse_args()

    config = Config(dimension=args.dim, algorithm=args.algo)
    viz = PathVisualizer(config)
    viz.run(mode=args.mode, output_file=args.output)

if __name__ == "__main__":
    main()
