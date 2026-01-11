#!/usr/bin/env python3
"""
Batch Occupancy Grid Generator
Generates training grids with various obstacle patterns.

Cell values:
  0 = Free
  1 = Occupied  
  2 = Unknown
  3 = Goal
"""

import numpy as np
import json
import os
import argparse
from datetime import datetime
from typing import Tuple, List


class GridGenerator:
    def __init__(self, grid_size: int = 50, seed: int = None):
        self.grid_size = grid_size
        if seed is not None:
            np.random.seed(seed)
    
    def empty_grid(self) -> np.ndarray:
        """Create empty grid with border walls."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        return grid
    
    def add_goal(self, grid: np.ndarray, pos: Tuple[int, int] = None) -> np.ndarray:
        """Add goal position. Random if not specified."""
        if pos is None:
            # Find free cells (not border)
            free = np.argwhere(grid == 0)
            if len(free) > 0:
                idx = np.random.randint(len(free))
                pos = tuple(free[idx])
        if pos and grid[pos] == 0:
            grid[pos] = 3
        return grid
    
    def random_obstacles(self, density: float = 0.15) -> np.ndarray:
        """Generate grid with random scattered obstacles."""
        grid = self.empty_grid()
        inner = grid[1:-1, 1:-1]
        mask = np.random.random(inner.shape) < density
        inner[mask] = 1
        return self.add_goal(grid)
    
    def rectangular_obstacles(self, n_rects: int = 3) -> np.ndarray:
        """Generate grid with rectangular obstacles."""
        grid = self.empty_grid()
        
        for _ in range(n_rects):
            w = np.random.randint(2, 5)
            h = np.random.randint(2, 5)
            x = np.random.randint(1, self.grid_size - w - 1)
            y = np.random.randint(1, self.grid_size - h - 1)
            grid[y:y+h, x:x+w] = 1
        
        return self.add_goal(grid)
    
    def corridor(self, width: int = 3) -> np.ndarray:
        """Generate corridor-like environment."""
        grid = np.ones((self.grid_size, self.grid_size), dtype=np.uint8)
        
        # Horizontal corridor
        mid = self.grid_size // 2
        start = mid - width // 2
        end = start + width
        grid[start:end, 1:-1] = 0
        
        # Add some openings
        for _ in range(2):
            x = np.random.randint(2, self.grid_size - 2)
            grid[1:start, x] = 0
            grid[end:, x] = 0
        
        return self.add_goal(grid)
    
    def maze_like(self, wall_prob: float = 0.3) -> np.ndarray:
        """Generate maze-like pattern with connected paths."""
        grid = self.empty_grid()
        
        # Add vertical walls with gaps
        for x in range(3, self.grid_size - 3, 3):
            gap = np.random.randint(1, self.grid_size - 1)
            for y in range(1, self.grid_size - 1):
                if abs(y - gap) > 1 and np.random.random() < wall_prob:
                    grid[y, x] = 1
        
        # Add horizontal walls with gaps
        for y in range(3, self.grid_size - 3, 3):
            gap = np.random.randint(1, self.grid_size - 1)
            for x in range(1, self.grid_size - 1):
                if abs(x - gap) > 1 and np.random.random() < wall_prob:
                    grid[y, x] = 1
        
        return self.add_goal(grid)
    
    def room_with_door(self) -> np.ndarray:
        """Generate room with single door opening."""
        grid = self.empty_grid()
        
        # Inner room walls
        room_start = 4
        room_end = self.grid_size - 4
        
        grid[room_start, room_start:room_end] = 1
        grid[room_end, room_start:room_end] = 1
        grid[room_start:room_end+1, room_start] = 1
        grid[room_start:room_end+1, room_end] = 1
        
        # Add door
        wall = np.random.choice(['top', 'bottom', 'left', 'right'])
        door_pos = np.random.randint(room_start + 1, room_end)
        
        if wall == 'top':
            grid[room_start, door_pos] = 0
        elif wall == 'bottom':
            grid[room_end, door_pos] = 0
        elif wall == 'left':
            grid[door_pos, room_start] = 0
        else:
            grid[door_pos, room_end] = 0
        
        return self.add_goal(grid)
    
    def l_shaped_obstacle(self) -> np.ndarray:
        """Generate L-shaped obstacles."""
        grid = self.empty_grid()
        
        for _ in range(2):
            x = np.random.randint(2, self.grid_size - 5)
            y = np.random.randint(2, self.grid_size - 5)
            length = np.random.randint(3, 5)
            
            # Horizontal part
            grid[y, x:x+length] = 1
            # Vertical part
            grid[y:y+length, x] = 1
        
        return self.add_goal(grid)
    
    def partial_unknown(self, unknown_ratio: float = 0.3) -> np.ndarray:
        """Generate grid with unknown regions (simulating unexplored areas)."""
        grid = self.random_obstacles(density=0.1)
        
        # Add unknown regions
        inner = grid[1:-1, 1:-1]
        free_cells = inner == 0
        unknown_mask = np.random.random(inner.shape) < unknown_ratio
        inner[free_cells & unknown_mask] = 2
        
        return grid
    
    def generate(self, pattern: str = 'random') -> np.ndarray:
        """Generate grid with specified pattern."""
        patterns = {
            'random': self.random_obstacles,
            'rectangles': self.rectangular_obstacles,
            'corridor': self.corridor,
            'maze': self.maze_like,
            'room': self.room_with_door,
            'l_shaped': self.l_shaped_obstacle,
            'unknown': self.partial_unknown
        }
        
        if pattern == 'mixed':
            pattern = np.random.choice(list(patterns.keys()))
        
        return patterns.get(pattern, self.random_obstacles)()


def save_grids(grids: List[np.ndarray], output_dir: str, format: str = 'json'):
    """Save generated grids to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, grid in enumerate(grids):
        if format == 'json':
            filepath = os.path.join(output_dir, f"grid_{i:04d}.json")
            data = {
                "grid_size": grid.shape[0],
                "resolution_cm": 10,
                "grid": grid.tolist()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f)
        else:  # numpy
            filepath = os.path.join(output_dir, f"grid_{i:04d}.npy")
            np.save(filepath, grid)
    
    # Also save combined dataset
    combined = np.stack(grids)
    if format == 'json':
        combined_path = os.path.join(output_dir, "all_grids.json")
        with open(combined_path, 'w') as f:
            json.dump({"grids": combined.tolist()}, f)
    else:
        np.save(os.path.join(output_dir, "all_grids.npy"), combined)


def main():
    parser = argparse.ArgumentParser(description="Generate occupancy grids for training")
    parser.add_argument("-n", "--num", type=int, default=100, help="Number of grids to generate")
    parser.add_argument("-s", "--size", type=int, default=15, help="Grid size (default: 15)")
    parser.add_argument("-p", "--pattern", type=str, default="mixed",
                        choices=['random', 'rectangles', 'corridor', 'maze', 'room', 'l_shaped', 'unknown', 'mixed'],
                        help="Obstacle pattern")
    parser.add_argument("-o", "--output", type=str, default="./grids", help="Output directory")
    parser.add_argument("-f", "--format", type=str, default="json", choices=['json', 'npy'], help="Output format")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"Generating {args.num} grids ({args.size}x{args.size}, pattern: {args.pattern})")
    
    generator = GridGenerator(args.size, args.seed)
    grids = [generator.generate(args.pattern) for _ in range(args.num)]
    
    save_grids(grids, args.output, args.format)
    print(f"Saved to {args.output}/")
    
    # Print stats
    all_grids = np.stack(grids)
    print(f"\nStats:")
    print(f"  Free cells avg: {(all_grids == 0).sum(axis=(1,2)).mean():.1f}")
    print(f"  Occupied avg: {(all_grids == 1).sum(axis=(1,2)).mean():.1f}")
    print(f"  Unknown avg: {(all_grids == 2).sum(axis=(1,2)).mean():.1f}")
    print(f"  Goals: {(all_grids == 3).sum()}")


if __name__ == "__main__":
    main()