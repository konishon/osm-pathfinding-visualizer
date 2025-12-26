from dataclasses import dataclass, field
from typing import Tuple
from pathlib import Path

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
