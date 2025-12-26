import argparse
import sys
import pickle
import numpy as np
import osmnx as ox
import networkx as nx
import wave
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import pyproj

from .sound_effects import (
    init_sound_system, create_search_sounds, create_path_found_sound, 
    get_search_waveforms, get_path_found_waveform
)
from .algorithms import ALGORITHMS

@dataclass
class Config:
    """Configuration for the visualization."""
    start_coord: Tuple[float, float] = (27.7172, 85.3240)
    end_coord: Tuple[float, float] = (27.6800, 85.3500)
    bbox_buffer: float = 0.025
    tilt_angle: float = 45
    rotation_angle: float = -45
    search_color: str = '#1e90ff'
    path_color: str = '#adff2f'
    bg_color: str = '#0b0b0b'
    duration: int = 10
    fps: int = 30
    cache_dir: Path = field(default_factory=lambda: Path('cache'))
    output_dir: Path = field(default_factory=lambda: Path('output'))
    dimension: str = '3d'
    algorithm: str = 'bfs'
    
    def __post_init__(self):
        self.cache_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    @property
    def total_frames(self) -> int:
        return self.duration * self.fps
    
    @property
    def phase_2_frames(self) -> int:
        return 60
        
    @property
    def phase_1_frames(self) -> int:
        return self.total_frames - self.phase_2_frames

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Calculate 9:16 aspect ratio bounding box."""
        cy = (self.start_coord[0] + self.end_coord[0]) / 2
        cx = (self.start_coord[1] + self.end_coord[1]) / 2
        
        dy = abs(self.start_coord[0] - self.end_coord[0]) + 2 * self.bbox_buffer
        dx = abs(self.start_coord[1] - self.end_coord[1]) + 2 * self.bbox_buffer
        
        target_ratio = 9 / 16
        if dx / dy > target_ratio:
            dy = dx / target_ratio
        else:
            dx = dy * target_ratio
            
        return cy + dy / 2, cy - dy / 2, cx + dx / 2, cx - dx / 2

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
        
        self.north, self.south, self.east, self.west = config.get_bbox()

    def init_audio(self):
        """Initialize audio system."""
        print("[0/5] Initializing sound system...")
        num_beeps = int(self.config.phase_1_frames / 5) + 2
        
        if self.mode == 'view':
            self.sound_enabled = init_sound_system()
            if self.sound_enabled:
                self.sounds['search'] = create_search_sounds(num_steps=num_beeps)
                self.sounds['found'] = create_path_found_sound()
                print("    ✓ Sound effects loaded")
        elif self.mode == 'export':
            self.waveforms['search'] = get_search_waveforms(num_steps=num_beeps)
            self.waveforms['found'] = get_path_found_waveform()
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
        
        bbox_str = f"{self.north:.4f}_{self.south:.4f}_{self.east:.4f}_{self.west:.4f}"
        graph_cache_path = self.config.cache_dir / f"graph_{bbox_str}.graphml"
        features_cache_path = self.config.cache_dir / f"features_{bbox_str}.pkl"
        
        if graph_cache_path.exists():
            print(f"    → Loading graph from cache: {graph_cache_path}")
            self.G = ox.load_graphml(graph_cache_path)
        else:
            print("    → Downloading graph from OSM...")
            self.G = ox.graph_from_bbox(
                bbox=(self.north, self.south, self.east, self.west), 
                network_type='drive'
            )
            ox.save_graphml(self.G, graph_cache_path)
            
        self.G = ox.project_graph(self.G)
        nodes = ox.graph_to_gdfs(self.G, edges=False)
        self.west, self.east = nodes.geometry.x.min(), nodes.geometry.x.max()
        self.south, self.north = nodes.geometry.y.min(), nodes.geometry.y.max()
        
        crs = self.G.graph['crs']
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self.start_x, self.start_y = transformer.transform(self.config.start_coord[1], self.config.start_coord[0])
        self.end_x, self.end_y = transformer.transform(self.config.end_coord[1], self.config.end_coord[0])

        if features_cache_path.exists():
            with open(features_cache_path, 'rb') as f:
                self.features = pickle.load(f)
        else:
            try:
                n, s, e, w = self.config.get_bbox()
                gdf = ox.features_from_bbox(bbox=(n, s, e, w), tags={'building': True})
                if 'building' in gdf.columns:
                    self.features['buildings'] = gdf[gdf['building'].notna()]
                with open(features_cache_path, 'wb') as f:
                    pickle.dump(self.features, f)
            except Exception as e:
                print(f"    ⚠ Could not load features: {e}")

        if 'buildings' in self.features:
             self.features['buildings'] = self.features['buildings'].to_crs(crs)

    def run_search(self):
        """Run Search and Shortest Path algorithms."""
        print(f"\n[2/5] Calculating search expansion ({self.config.algorithm.upper()})...")
        
        start_node = ox.distance.nearest_nodes(self.G, self.start_x, self.start_y)
        end_node = ox.distance.nearest_nodes(self.G, self.end_x, self.end_y)
        
        search_func = ALGORITHMS.get(self.config.algorithm, ALGORITHMS['bfs'])
        iterator = search_func(self.G, start_node, end_node)

        for u, v in iterator:
            data = self.G.get_edge_data(u, v)[0]
            coords = list(data['geometry'].coords) if 'geometry' in data else [
                (self.G.nodes[u]['x'], self.G.nodes[u]['y']), 
                (self.G.nodes[v]['x'], self.G.nodes[v]['y'])
            ]
            
            if len(coords) >= 2:
                z = -100 if self.config.dimension == '3d' else 0
                coords_3d = np.array([(c[0], c[1], z) for c in coords]) if self.config.dimension == '3d' else np.array([(c[0], c[1]) for c in coords])
                self.explored_edges.append(coords_3d)
            
            if v == end_node:
                break
        
        print(f"    ✓ Explored {len(self.explored_edges)} edges")

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
            
            if len(route_coords) >= 2:
                z = -200 if self.config.dimension == '3d' else 0
                self.route_coords_3d = np.array([(c[0], c[1], z) for c in route_coords]) if self.config.dimension == '3d' else np.array([(c[0], c[1]) for c in route_coords])
            else:
                self.route_coords_3d = np.empty((0, 3 if self.config.dimension == '3d' else 2))
        except nx.NetworkXNoPath:
            print("    ⚠ NO PATH FOUND!")
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

        hud_font = {'family': 'monospace', 'weight': 'bold', 'size': 14}
        self.texts['status'] = self.fig.text(0.05, 0.95, "SYSTEM: ONLINE", color='#00ff00', alpha=0.8, ha='left', va='top', **hud_font)
        self.texts['scan'] = self.fig.text(0.05, 0.92, "SCANNED: 0", color=self.config.search_color, alpha=0.8, ha='left', va='top', **hud_font)
  
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

    def run(self, mode='view', output_file=None):
        """Execute the visualization."""
        self.mode = mode
        
        if output_file is None:
            bbox_str = f"{self.north:.4f}_{self.south:.4f}_{self.east:.4f}_{self.west:.4f}"
            output_file = f"path_viz_{self.config.algorithm}_{bbox_str}.mp4"
            
        output_path = self.config.output_dir / Path(output_file).name
        
        self.init_audio()
        self.load_data()
        self.run_search()
        self.setup_scene()
        
        if mode == 'preview':
            print("\n[5/5] Showing preview (static)...")
            self.collections['search'].set_segments(self.explored_edges)
            path_segments = [self.route_coords_3d]
            self.collections['path_glow_outer'].set_segments(path_segments)
            self.collections['path_glow_inner'].set_segments(path_segments)
            self.collections['path_line'].set_segments(path_segments)
            
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
            print(f"    → Rendering to {output_path}...")
            writer = FFMpegWriter(fps=self.config.fps, bitrate=1800)
            ani.save(str(output_path), writer=writer)
            
            audio_path = output_path.with_suffix('.wav')
            self.save_audio(str(audio_path))
            
            final_output = output_path.parent / f"{output_path.stem}_with_audio.mp4"
            print(f"    → Merging audio and video...")
            try:
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', str(output_path),
                    '-i', str(audio_path),
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-strict', 'experimental',
                    str(final_output)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                audio_path.unlink(missing_ok=True)
                final_output.replace(output_path)
                print(f"    ✓ Final video saved: {output_path}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"    ⚠ Could not merge audio: {e}")

def main():
    parser = argparse.ArgumentParser(description="3D Pathfinding Visualization")
    parser.add_argument('--mode', choices=['view', 'export', 'preview'], default='view')
    parser.add_argument('--output', default=None)
    parser.add_argument('--dim', choices=['2d', '3d'], default='3d')
    parser.add_argument('--algo', choices=['bfs', 'astar', 'dijkstra', 'greedy'], default='bfs')
    args = parser.parse_args()

    config = Config(dimension=args.dim, algorithm=args.algo)
    viz = PathVisualizer(config)
    viz.run(mode=args.mode, output_file=args.output)

if __name__ == "__main__":
    main()
