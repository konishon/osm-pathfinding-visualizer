import argparse
import sys
import os
import pickle
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

# Import local modules
try:
    from sound_effects import init_sound_system, create_search_sounds, create_path_found_sound
except ImportError:
    print("Warning: sound_effects module not found. Audio will be disabled.")
    def init_sound_system(): return False
    def create_search_sounds(): return []
    def create_path_found_sound(): return None

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
        self.sound_enabled = False
        
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
        self.sound_enabled = init_sound_system()
        if self.sound_enabled:
            self.sounds['search'] = create_search_sounds()
            self.sounds['found'] = create_path_found_sound()
            print("    ✓ Sound effects loaded")
        else:
            print("    ⚠ Sound effects disabled")

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
                gdf = ox.features_from_bbox(bbox=(self.north, self.south, self.east, self.west), tags=tags)
                if 'building' in gdf.columns:
                    self.features['buildings'] = gdf[gdf['building'].notna()]
                
                print(f"    → Saving features to cache...")
                with open(features_cache_path, 'wb') as f:
                    pickle.dump(self.features, f)
                print(f"    ✓ Loaded features")
            except Exception as e:
                print(f"    ⚠ Could not load features: {e}")

    def run_search(self):
        """Run BFS and Shortest Path algorithms."""
        print("\n[2/5] Calculating search expansion...")
        
        start_node = ox.distance.nearest_nodes(self.G, self.config.start_coord[1], self.config.start_coord[0])
        end_node = ox.distance.nearest_nodes(self.G, self.config.end_coord[1], self.config.end_coord[0])
        
        # BFS for visualization
        for edge in nx.bfs_edges(self.G, source=start_node):
            u, v = edge
            data = self.G.get_edge_data(u, v)[0]
            if 'geometry' in data:
                coords = list(data['geometry'].coords)
            else:
                coords = [(self.G.nodes[u]['x'], self.G.nodes[u]['y']), 
                          (self.G.nodes[v]['x'], self.G.nodes[v]['y'])]
            
            # Convert to 3D numpy array (N, 3)
            # Lift search layer slightly (z=0.0002) to be just above roads
            coords_3d = np.array([(c[0], c[1], 0.0002) for c in coords])
            self.explored_edges.append(coords_3d)
            
            if v == end_node:
                break
        
        print(f"    ✓ Explored {len(self.explored_edges)} edges")

        # Shortest Path
        print("\n[3/5] Computing shortest path...")
        route = nx.shortest_path(self.G, start_node, end_node, weight='length')
        route_coords = []
        for u, v in zip(route[:-1], route[1:]):
            data = self.G.get_edge_data(u, v)[0]
            if 'geometry' in data:
                route_coords.extend(list(data['geometry'].coords))
            else:
                route_coords.extend([(self.G.nodes[u]['x'], self.G.nodes[u]['y']), 
                                   (self.G.nodes[v]['x'], self.G.nodes[v]['y'])])
        
        # Convert to 3D numpy array (N, 3) with higher Z offset (z=0.0004)
        self.route_coords_3d = np.array([(c[0], c[1], 0.0004) for c in route_coords])
        print(f"    ✓ Final route: {len(route)} nodes")

    def _plot_polygon_3d(self, poly, color, alpha, z=-0.0001):
        """Helper to plot 3D polygons."""
        try:
            if poly.geom_type == 'Polygon':
                x, y = poly.exterior.xy
                verts = [list(zip(x, y, [z]*len(x)))]
                poly_coll = Poly3DCollection(verts, color=color, alpha=alpha)
                self.ax.add_collection(poly_coll)
            elif poly.geom_type == 'MultiPolygon':
                for p in poly.geoms:
                    x, y = p.exterior.xy
                    verts = [list(zip(x, y, [z]*len(x)))]
                    poly_coll = Poly3DCollection(verts, color=color, alpha=alpha)
                    self.ax.add_collection(poly_coll)
        except Exception:
            pass

    def setup_scene(self):
        """Setup the 3D scene and static elements."""
        print("\n[4/5] Setting up scene...")
        
        # Figure setup
        self.fig = plt.figure(figsize=(9, 16), dpi=100, facecolor=self.config.bg_color)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor(self.config.bg_color)
        self.ax.view_init(elev=self.config.tilt_angle, azim=self.config.rotation_angle)
        self.ax.set_axis_off()
        
        # Set limits
        self.ax.set_xlim(self.west, self.east)
        self.ax.set_ylim(self.south, self.north)
        self.ax.set_zlim(-2, 8)
        
        # Force the 3D box to match the figure aspect ratio
        # Increase Z aspect to make layers distinct (Diorama effect)
        self.ax.set_box_aspect((9, 16, 5))
        
        # Zoom in to minimize margins
        self.ax.dist = 6  # Default is 10, lower is closer

        # Render Buildings
        if 'buildings' in self.features:
            print("    → Rendering buildings...")
            for poly in self.features['buildings'].geometry[:1000]:  # Limit for performance
                self._plot_polygon_3d(poly, '#1f1f24', 0.5, z=-2)

        # Render Roads
        print("    → Rendering roads...")
        road_lines = []
        for u, v, data in self.G.edges(data=True):
            if 'geometry' in data:
                xs, ys = data['geometry'].xy
                # Ensure homogeneous shape (N, 3)
                points = np.column_stack((xs, ys, np.zeros(len(xs))))
                road_lines.append(points)
            else:
                p1 = (self.G.nodes[u]['x'], self.G.nodes[u]['y'], 0)
                p2 = (self.G.nodes[v]['x'], self.G.nodes[v]['y'], 0)
                road_lines.append(np.array([p1, p2]))
        
        # Create collection manually to avoid "inhomogeneous shape" error
        # Line3DCollection expects a list of (N, 3) arrays
        # Increased visibility for roads to ensure they fill the video
        road_coll = Line3DCollection(road_lines, colors='#555555', linewidths=1.0, alpha=0.8)
        self.ax.add_collection(road_coll)

        # Markers (Start/End)
        start_node = ox.distance.nearest_nodes(self.G, self.config.start_coord[1], self.config.start_coord[0])
        end_node = ox.distance.nearest_nodes(self.G, self.config.end_coord[1], self.config.end_coord[0])
        
        sx, sy = self.G.nodes[start_node]['x'], self.G.nodes[start_node]['y']
        ex, ey = self.G.nodes[end_node]['x'], self.G.nodes[end_node]['y']
        
        # Plot markers (using scatter for 3D points)
        # Z=6 to sit above everything
        self.ax.scatter([sx], [sy], [6], c='#00ff00', s=200, label='Start', edgecolors='white', alpha=1.0)
        self.ax.scatter([ex], [ey], [6], c='#ff0000', s=200, label='End', edgecolors='white', alpha=1.0)

        # Dynamic Elements (Search & Path)
        # Search lines
        self.collections['search'] = Line3DCollection([], colors=self.config.search_color, linewidths=1.5, alpha=0.8)
        self.ax.add_collection(self.collections['search'])

        # Path lines
        self.collections['path_glow_outer'] = Line3DCollection([], colors=self.config.path_color, linewidths=8, alpha=0.2)
        self.collections['path_glow_inner'] = Line3DCollection([], colors=self.config.path_color, linewidths=5, alpha=0.4)
        self.collections['path_line'] = Line3DCollection([], colors=self.config.path_color, linewidths=2.5, alpha=1)
        
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
            progress = (frame + 1) / self.config.phase_1_frames
            idx = int(progress * len(self.explored_edges))
            
            # Update search web
            current_edges = self.explored_edges[:idx]
            self.collections['search'].set_segments(current_edges)
            
            # Update HUD
            self.texts['status'].set_text("STATUS: SEARCHING...")
            self.texts['status'].set_color(self.config.search_color)
            self.texts['scan'].set_text(f"SCANNED: {idx}")
            
            # Audio
            if self.sound_enabled and frame % 5 == 0 and self.sounds['search']:
                sound = self.sounds['search'][frame % len(self.sounds['search'])]
                sound.play()
                
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
            if frame == self.config.phase_1_frames and self.sound_enabled and self.sounds['found']:
                self.sounds['found'].play()

        return (self.collections['search'], self.collections['path_line'], 
                self.texts['status'], self.texts['scan'])

    def run(self, mode='view', output_file='output.mp4'):
        """Execute the visualization."""
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
            print(f"    ✓ Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="3D Pathfinding Visualization")
    parser.add_argument('--mode', choices=['view', 'export', 'preview'], default='view', help="Run mode: 'view' for interactive, 'export' for video file, 'preview' for static check")
    parser.add_argument('--output', default='tiktok_path.mp4', help="Output filename for export mode")
    args = parser.parse_args()

    config = Config()
    viz = PathVisualizer(config)
    viz.run(mode=args.mode, output_file=args.output)

if __name__ == "__main__":
    main()
