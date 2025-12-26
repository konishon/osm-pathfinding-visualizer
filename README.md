# Dijkstra Search & Path Visualization

A Python visualization tool that animates the Dijkstra/BFS algorithm search process with an expanding blue web, followed by the shortest path highlighted in neon green. Perfect for viral-style algorithm visualizations!

## Features

✨ **Blue Expanding Web**: Visualizes the search exploration in real-time  
✨ **Neon Green Path**: Highlights the final shortest route  
✨ **Glow Effects**: Multi-layered path rendering for that viral aesthetic  
✨ **Dark Mode**: Sleek dark background with neon colors  
✨ **MP4 Export**: Save animations directly as video files  
✨ **Real Map Data**: Uses OpenStreetMap via OSMnx for accurate street networks  

## Installation

### Step 1: Install Poetry

Poetry is a modern dependency and environment manager for Python. Install it first:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or if you already have Poetry:
```bash
poetry self update
```

### Step 2: Install Python Dependencies

```bash
poetry install
```

This will:
- Create a virtual environment automatically
- Install all dependencies from `pyproject.toml`
- Generate a `poetry.lock` file for reproducible builds

To activate the Poetry shell:
```bash
poetry shell
```

### Step 3: Install FFmpeg (for video export)

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html or use:
```bash
choco install ffmpeg
```

## Usage

### Option 1: Interactive Visualization (Recommended for Testing)

```bash
poetry run path-viz
```

Or if you've activated the Poetry shell:
```bash
path-viz
```

This will:
1. Download map data for Istanbul (Taksim area)
2. Calculate the search expansion
3. Show an interactive matplotlib window
4. Display the animation in real-time

### Option 2: Export to MP4 Video

```bash
poetry run path-viz --mode export
```

Or with shell activated:
```bash
path-viz --mode export
```

This will:
1. Generate the same animation
2. Export to `path_viz.mp4` (about 30 seconds)
3. Show progress during export

## Configuration

You can configure the visualization using command line arguments:

```bash
path-viz --mode [view|export|preview] --dim [2d|3d] --algo [bfs|astar|dijkstra|greedy]
```

Or edit the `Config` class in `src/path_viz/visualizer.py` to customize:

```python
@dataclass
class Config:
    start_coord: Tuple[float, float] = (27.7172, 85.3240)  # Kathmandu Center
    end_coord: Tuple[float, float] = (27.6800, 85.3500)    # Kathmandu South-East
    search_color: str = '#1e90ff'  # Neon Blue
    path_color: str = '#adff2f'    # Neon Green
    bg_color: str = '#0b0b0b'      # Near black
```

### Popular Cities (Lat, Lon)

| City | Coordinates |
|------|------------|
| Istanbul | `(41.0370, 28.9850)` |
| New York | `(40.7128, -74.0060)` |
| London | `(51.5074, -0.1278)` |
| Tokyo | `(35.6762, 139.6503)` |
| Paris | `(48.8566, 2.3522)` |
| San Francisco | `(37.7749, -122.4194)` |

### Color Codes

| Effect | Recommended Hex |
|--------|-----------------|
| Neon Blue | `#1e90ff`, `#00bfff` |
| Neon Green | `#adff2f`, `#00ff00` |
| Neon Purple | `#da70d6`, `#ff00ff` |
| Neon Pink | `#ff1493`, `#ff69b4` |

## Output Files

- **`src/path_viz/visualizer.py`** - Main visualization script
- **`path_viz.mp4`** - Generated video file (after running export mode)

## Performance Tips

### For Faster Execution:
```python
DISTANCE = 1000  # Reduce from 2000 meters
```

### For Better Quality:
```python
# In ani.save() call, increase dpi:
ani.save(OUTPUT_FILE, writer=writer, dpi=300)  # Higher resolution
```

### For More Search Frames:
```python
# In update() function, change:
if frame < search_phase_frames:
    idx = frame * 10  # Show 10 edges per frame instead of 20
```

## Troubleshooting

### FFmpeg Not Found
```
Error: ffmpeg not found
```
**Solution:** Install FFmpeg (see Installation section above)

### No Map Data Available
```
Error downloading map data for that location
```
**Solution:** Some areas may not have complete OpenStreetMap data. Try a different city or larger DISTANCE value.

### Animation is Slow
```
Animation playback is laggy
```
**Solution:** Reduce DISTANCE or run `search_path_visualization.py` first to see performance without export overhead.

### Out of Memory
```
MemoryError during execution
```
**Solution:** Reduce DISTANCE to 1000 or less, or use a simpler network_type (already set to 'drive').

## Technical Details

### Algorithm: Multiple Search Algorithms
- **BFS**: Breadth-First Search (Unweighted) - Explores equally in all directions
- **Dijkstra**: Weighted Shortest Path - Explores based on actual road distance
- **A***: Weighted + Heuristic - Uses distance to target to guide search
- **Greedy**: Heuristic Only - Moves directly towards target (not guaranteed shortest)

### Performance Optimization
- Uses `LineCollection` instead of individual Line2D objects
- Handles 2000+ line segments efficiently
- Lazy rendering with matplotlib's blit=True

### Map Data Source
- **OpenStreetMap** (via OSMnx)
- **Network Type**: `drive` (drivable roads)
- **Resolution**: 2000 meters (configurable)

## Advanced: Custom Styling

To create your own glow effect with different colors:

```python
# Create custom glow layers
path_glow_outer, = ax.plot([], [], color='#ff00ff', linewidth=12, alpha=0.15)
path_glow_mid, = ax.plot([], [], color='#ff00ff', linewidth=8, alpha=0.3)
path_glow_inner, = ax.plot([], [], color='#ff00ff', linewidth=3, alpha=0.6)
path_line, = ax.plot([], [], color='#ffffff', linewidth=1.5, alpha=1)

# In update(), set all of them:
path_glow_outer.set_data(path_x, path_y)
path_glow_mid.set_data(path_x, path_y)
path_glow_inner.set_data(path_x, path_y)
path_line.set_data(path_x, path_y)
```

## License

This project is provided as-is for educational and entertainment purposes.

## Dependencies

- **osmnx** - OpenStreetMap network data
- **networkx** - Graph algorithms
- **matplotlib** - Visualization and animation
- **numpy** - Numerical operations
- **FFmpeg** - Video encoding (optional, for export)

## Poetry Commands Cheat Sheet

```bash
# Install all dependencies
poetry install

# Activate the virtual environment
poetry shell

# Run a script without activating shell
poetry run python script.py

# Add a new dependency
poetry add package-name

# Add a dev dependency
poetry add --group dev package-name

# Update all dependencies
poetry update

# Lock dependencies (creates poetry.lock)
poetry lock

# Show installed packages
poetry show

# Export requirements format (if needed)
poetry export -f requirements.txt --output requirements.txt
```

---

**Ready to create viral algorithm visualizations!** 🎬✨
