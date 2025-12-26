# Path Finder Visualizer

A Python tool to visualize pathfinding algorithms on real-world maps with 3D effects and dynamic sound.

![Demo](demo1.gif)

## Features
- **Multiple Algorithms**: BFS, A*, Dijkstra, and Greedy Best-First Search.
- **3D/2D Visualization**: Toggle between 3D perspective and 2D top-down views.
- **Rich 2D Maps**: 2D mode includes rendering for rivers, lakes, forests, parks, and buildings.
- **Dynamic Sound**: Growing pitch sound effects during search and a triumphant "found" sound.
- **Real Map Data**: Fetches street networks from OpenStreetMap.
- **Video/GIF Export**: Export animations to high-quality MP4 with audio or silent GIFs.

## Requirements
- **Python 3.10+**
- **FFmpeg**: Required for video export (`--mode export`).

## Quick Start

1. **Install Dependencies**:
   ```bash
   poetry install
   ```

2. **Run Visualization**:
   ```bash
   poetry run path-viz
   ```

## CLI Usage

```bash
path-viz --mode [view|export|preview] --algo [bfs|astar|dijkstra|greedy] --dim [2d|3d] --output demo.gif
```

- `--mode`: `view` (interactive), `export` (save to mp4/gif), `preview` (static check).
- `--algo`: Choose the search algorithm.
- `--dim`: `2d` or `3d` visualization.
- `--output`: Optional filename (e.g., `result.mp4` or `demo.gif`).

## Output
- **Videos**: Saved in the `output/` directory.
- **Cache**: Map data and features are cached in the `cache/` directory.
