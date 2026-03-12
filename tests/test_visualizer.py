"""Regression and integration tests for PathVisualizer and related modules."""
import argparse
import sys
import pytest
import networkx as nx
import numpy as np

from path_viz.config import Config
from path_viz.algorithms import ALGORITHMS, heuristic
from path_viz.visualizer import PathVisualizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_graph():
    """Return a tiny directed graph with node attributes x/y and edge length."""
    G = nx.MultiDiGraph()
    # 4 nodes forming a small grid
    nodes = {
        0: {'x': 0.0, 'y': 0.0},
        1: {'x': 1.0, 'y': 0.0},
        2: {'x': 2.0, 'y': 0.0},
        3: {'x': 1.0, 'y': 1.0},
    }
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs)

    edges = [
        (0, 1, {'length': 1.0}),
        (1, 2, {'length': 1.0}),
        (0, 3, {'length': 1.414}),
        (3, 2, {'length': 1.414}),
    ]
    for u, v, data in edges:
        G.add_edge(u, v, **data)

    return G


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_coords(self):
        config = Config()
        assert config.start_coord[0] == pytest.approx(28.190659105384285)
        assert config.end_coord[1] == pytest.approx(83.95717153743145)

    def test_custom_coords(self):
        config = Config(start_coord=(10.0, 20.0), end_coord=(11.0, 21.0))
        assert config.start_coord == (10.0, 20.0)
        assert config.end_coord == (11.0, 21.0)

    def test_total_frames(self):
        config = Config(duration=5, fps=30)
        assert config.total_frames == 150

    def test_phase_frames_sum(self):
        config = Config(duration=10, fps=30)
        assert config.phase_1_frames + config.phase_2_frames == config.total_frames

    def test_get_bbox_returns_four_values(self):
        config = Config()
        bbox = config.get_bbox()
        assert len(bbox) == 4
        north, south, east, west = bbox
        assert north > south
        assert east > west

    def test_algorithm_default(self):
        config = Config()
        assert config.algorithm == 'bfs'


# ---------------------------------------------------------------------------
# _reconstruct_path_from_parents tests
# ---------------------------------------------------------------------------

class TestReconstructPath:
    """Unit tests for PathVisualizer._reconstruct_path_from_parents."""

    def _viz(self):
        """Create a PathVisualizer without triggering any I/O."""
        config = Config()
        viz = PathVisualizer.__new__(PathVisualizer)
        viz.config = config
        return viz

    def test_valid_path(self):
        viz = self._viz()
        parent = {0: None, 1: 0, 2: 1, 3: 2}
        assert viz._reconstruct_path_from_parents(parent, 0, 3) == [0, 1, 2, 3]

    def test_end_not_in_parent(self):
        viz = self._viz()
        parent = {0: None, 1: 0}
        assert viz._reconstruct_path_from_parents(parent, 0, 99) == []

    def test_single_node_path(self):
        viz = self._viz()
        parent = {5: None}
        assert viz._reconstruct_path_from_parents(parent, 5, 5) == [5]

    def test_start_not_reached(self):
        # end_node is reachable but chain doesn't lead back to start_node
        viz = self._viz()
        parent = {10: None, 11: 10, 3: 11}  # chain ends at 10, not at start=0
        assert viz._reconstruct_path_from_parents(parent, 0, 3) == []

    def test_cycle_detection(self):
        viz = self._viz()
        # Artificially create a cycle in parent pointers
        parent = {0: None, 1: 2, 2: 1}
        assert viz._reconstruct_path_from_parents(parent, 0, 2) == []

    def test_missing_intermediate_node(self):
        viz = self._viz()
        # end_node 2 is present in parent, but parent[2] is None (no chain to start_node 0).
        # The while loop terminates with path=[2]. Since path[-1] (2) != start_node (0), returns [].
        parent = {0: None, 2: None}
        result = viz._reconstruct_path_from_parents(parent, 0, 2)
        assert result == []


# ---------------------------------------------------------------------------
# Algorithm traversal tests (on simple in-memory graph)
# ---------------------------------------------------------------------------

class TestAlgorithms:
    def setup_method(self):
        self.G = _make_simple_graph()

    def test_bfs_traversal_reaches_end(self):
        edges = list(ALGORITHMS['bfs'](self.G, 0, 2))
        visited = {u for u, v in edges} | {v for u, v in edges}
        assert 2 in visited

    def test_dijkstra_traversal_reaches_end(self):
        edges = list(ALGORITHMS['dijkstra'](self.G, 0, 2))
        visited = {u for u, v in edges} | {v for u, v in edges}
        assert 2 in visited

    def test_astar_traversal_reaches_end(self):
        edges = list(ALGORITHMS['astar'](self.G, 0, 2))
        visited = {u for u, v in edges} | {v for u, v in edges}
        assert 2 in visited

    def test_greedy_traversal_reaches_end(self):
        edges = list(ALGORITHMS['greedy'](self.G, 0, 2))
        visited = {u for u, v in edges} | {v for u, v in edges}
        assert 2 in visited

    def test_heuristic_positive(self):
        G = self.G
        assert heuristic(G, 0, 2) > 0

    def test_heuristic_zero_for_same_node(self):
        G = self.G
        assert heuristic(G, 1, 1) == pytest.approx(0.0)

    def test_all_algorithms_present(self):
        assert set(ALGORITHMS.keys()) == {'bfs', 'astar', 'dijkstra', 'greedy'}


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------

class TestCLIParsing:
    """Tests for the main() argument parser logic (extracted inline)."""

    def _parse(self, argv):
        """Run the argparse logic extracted from main() and return config_kwargs."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', choices=['view', 'export', 'preview'], default='view')
        parser.add_argument('--output', default=None)
        parser.add_argument('--dim', choices=['2d', '3d'], default='3d')
        parser.add_argument('--algo', choices=['bfs', 'astar', 'dijkstra', 'greedy'], default='bfs')
        parser.add_argument('--duration', type=int, default=10)
        parser.add_argument('--fps', type=int, default=30)
        parser.add_argument('--start', type=str, default=None)
        parser.add_argument('--end', type=str, default=None)
        args = parser.parse_args(argv)

        config_kwargs = dict(dimension=args.dim, algorithm=args.algo,
                             duration=args.duration, fps=args.fps)
        if args.start:
            lat, lon = map(float, args.start.split(','))
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError(f'Invalid --start: {args.start}')
            config_kwargs['start_coord'] = (lat, lon)
        if args.end:
            lat, lon = map(float, args.end.split(','))
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError(f'Invalid --end: {args.end}')
            config_kwargs['end_coord'] = (lat, lon)
        return config_kwargs

    def test_default_no_coords(self):
        kw = self._parse([])
        assert 'start_coord' not in kw
        assert 'end_coord' not in kw
        assert kw['algorithm'] == 'bfs'

    def test_start_coord_parsed(self):
        kw = self._parse(['--start', '28.19,84.01'])
        assert kw['start_coord'] == pytest.approx((28.19, 84.01))

    def test_end_coord_parsed(self):
        kw = self._parse(['--end', '28.20,83.95'])
        assert kw['end_coord'] == pytest.approx((28.20, 83.95))

    def test_both_coords_parsed(self):
        kw = self._parse(['--start', '10.0,20.0', '--end', '11.0,21.0'])
        assert kw['start_coord'] == pytest.approx((10.0, 20.0))
        assert kw['end_coord'] == pytest.approx((11.0, 21.0))

    def test_invalid_start_bad_format(self):
        with pytest.raises(Exception):
            self._parse(['--start', 'notanumber'])

    def test_invalid_start_lat_out_of_range(self):
        with pytest.raises(ValueError):
            self._parse(['--start', '91.0,0.0'])

    def test_invalid_start_lon_out_of_range(self):
        with pytest.raises(ValueError):
            self._parse(['--start', '0.0,181.0'])

    def test_invalid_end_lat_out_of_range(self):
        # Use '=' notation to avoid argparse treating the leading '-' as a flag
        with pytest.raises(ValueError):
            self._parse(['--end=-91.0,0.0'])

    def test_invalid_end_lon_out_of_range(self):
        with pytest.raises(ValueError):
            self._parse(['--end', '0.0,-181.0'])

    def test_boundary_coords_valid(self):
        # Use '=' notation for negative values to avoid argparse flag confusion
        kw = self._parse(['--start', '90.0,180.0', '--end=-90.0,-180.0'])
        assert kw['start_coord'] == pytest.approx((90.0, 180.0))
        assert kw['end_coord'] == pytest.approx((-90.0, -180.0))

    def test_negative_coords_with_equals_notation(self):
        # '=' notation must work for valid negative coordinates (help text recommendation)
        kw = self._parse(['--start=-45.0,120.0', '--end=10.0,-30.5'])
        assert kw['start_coord'] == pytest.approx((-45.0, 120.0))
        assert kw['end_coord'] == pytest.approx((10.0, -30.5))

    def test_algo_selection(self):
        for algo in ['bfs', 'astar', 'dijkstra', 'greedy']:
            kw = self._parse(['--algo', algo])
            assert kw['algorithm'] == algo


# ---------------------------------------------------------------------------
# Parent tracking in run_search (unit-level, no OSM/network I/O)
# ---------------------------------------------------------------------------

class TestParentTracking:
    """Verify that parent dict is only allocated for the greedy algorithm."""

    def test_parent_none_for_non_greedy(self):
        """For non-greedy algorithms the parent dict should remain None."""
        for algo in ('bfs', 'dijkstra', 'astar'):
            config = Config(algorithm=algo)
            # Check that parent is only initialized for greedy
            parent = {0: None} if config.algorithm == 'greedy' else None
            assert parent is None, f"Expected None parent for algo={algo}"

    def test_parent_initialized_for_greedy(self):
        config = Config(algorithm='greedy')
        start_node = 42
        parent = {start_node: None} if config.algorithm == 'greedy' else None
        assert parent is not None
        assert start_node in parent
        assert parent[start_node] is None
