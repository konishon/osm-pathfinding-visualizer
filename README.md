# Dijkstra Search & Path Visualization

A Python visualization tool that animates the Dijkstra/BFS algorithm search process with an expanding blue web, followed by the shortest path highlighted in neon green. Perfect for TikTok-style algorithm visualizations!

## Features

✨ **Blue Expanding Web**: Visualizes the search exploration in real-time  
✨ **Neon Green Path**: Highlights the final shortest route  
✨ **Glow Effects**: Multi-layered path rendering for that viral TikTok aesthetic  
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
poetry run python search_path_visualization.py
```

Or if you've activated the Poetry shell:
```bash
python search_path_visualization.py
```

This will:
1. Download map data for Istanbul (Taksim area)
2. Calculate the search expansion
3. Show an interactive matplotlib window
4. Display the animation in real-time

### Option 2: Export to MP4 Video

```bash
poetry run python search_path_with_export.py
```

Or with shell activated:
```bash
python search_path_with_export.py
```

This will:
1. Generate the same animation
2. Export to `search_path_animation.mp4` (about 30 seconds)
3. Show progress during export

## Configuration

Edit the configuration section in either script to customize:

```python
# --- CONFIGURATION ---
CITY_POINT = (41.0370, 28.9850)  # Change latitude/longitude
DISTANCE = 2000                   # Search radius in meters
SEARCH_COLOR = '#1e90ff'          # Change the blue color
PATH_COLOR = '#adff2f'            # Change the green color
BG_COLOR = '#0b0b0b'              # Background darkness
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

- **`search_path_visualization.py`** - Main visualization script (interactive only)
- **`search_path_with_export.py`** - Includes MP4 export functionality
- **`search_path_animation.mp4`** - Generated video file (after running export version)

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

### Algorithm: Breadth-First Search (BFS)
- Explores nodes in order of distance from start
- More visually interesting than Dijkstra for this animation
- Guarantees finding shortest path in unweighted graphs

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
