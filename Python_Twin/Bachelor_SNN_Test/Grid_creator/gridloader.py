#!/usr/bin/env python3
"""
PyTorch Dataset for Occupancy Grids
Compatible with snnTorch for SNN training.

Encoding options:
  - flat: 225 neurons (15x15), values 0-3 normalized
  - binary: 450 neurons (2 bits per cell)
  - onehot: 900 neurons (4 channels per cell)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional, Literal, Tuple


class OccupancyGridDataset(Dataset):
    """Dataset for loading occupancy grids."""
    
    def __init__(
        self,
        data_path: str,
        encoding: Literal['flat', 'binary', 'onehot'] = 'flat',
        normalize: bool = True,
        dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            data_path: Path to directory with grid files, single .npy, or .json file
            encoding: How to encode cell values for neural network input
                - 'flat': Direct values (225 neurons for 15x15)
                - 'binary': 2-bit encoding (450 neurons)
                - 'onehot': One-hot per cell (900 neurons, 4 channels)
            normalize: Normalize values to [0, 1] range
            dtype: Output tensor dtype
        """
        self.encoding = encoding
        self.normalize = normalize
        self.dtype = dtype
        self.grids = self._load_grids(data_path)
        
    def _load_grids(self, path: str) -> np.ndarray:
        """Load grids from various formats."""
        path = Path(path)
        
        if path.is_file():
            if path.suffix == '.npy':
                data = np.load(path)
                # Handle single grid vs batch
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                return data
            elif path.suffix == '.json':
                with open(path) as f:
                    content = json.load(f)
                if 'grids' in content:
                    return np.array(content['grids'], dtype=np.uint8)
                else:
                    return np.array([content['grid']], dtype=np.uint8)
        
        elif path.is_dir():
            grids = []
            for file in sorted(path.glob('grid_*.json')):
                with open(file) as f:
                    data = json.load(f)
                grids.append(data['grid'])
            for file in sorted(path.glob('grid_*.npy')):
                grids.append(np.load(file))
            
            if not grids:
                raise ValueError(f"No grid files found in {path}")
            return np.array(grids, dtype=np.uint8)
        
        raise ValueError(f"Invalid path: {path}")
    
    def _encode(self, grid: np.ndarray) -> torch.Tensor:
        """Encode grid for neural network input."""
        
        if self.encoding == 'flat':
            # Direct flattening: 225 values
            out = grid.flatten().astype(np.float32)
            if self.normalize:
                out = out / 3.0  # Max value is 3
            return torch.tensor(out, dtype=self.dtype)
        
        elif self.encoding == 'binary':
            # 2-bit encoding: 450 values (2 per cell)
            flat = grid.flatten()
            bit0 = (flat & 1).astype(np.float32)
            bit1 = ((flat >> 1) & 1).astype(np.float32)
            out = np.stack([bit0, bit1], axis=-1).flatten()
            return torch.tensor(out, dtype=self.dtype)
        
        elif self.encoding == 'onehot':
            # One-hot: 900 values (4 per cell)
            flat = grid.flatten()
            onehot = np.zeros((flat.size, 4), dtype=np.float32)
            onehot[np.arange(flat.size), flat] = 1.0
            return torch.tensor(onehot.flatten(), dtype=self.dtype)
        
        raise ValueError(f"Unknown encoding: {self.encoding}")
    
    def __len__(self) -> int:
        return len(self.grids)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._encode(self.grids[idx])
    
    def get_raw(self, idx: int) -> np.ndarray:
        """Get raw grid without encoding."""
        return self.grids[idx]
    
    @property
    def input_size(self) -> int:
        """Return input size based on encoding."""
        grid_cells = self.grids[0].size
        if self.encoding == 'flat':
            return grid_cells
        elif self.encoding == 'binary':
            return grid_cells * 2
        elif self.encoding == 'onehot':
            return grid_cells * 4


class NavigationDataset(Dataset):
    """
    Extended dataset with goal vector and velocity inputs.
    Matches your SNN architecture: grid + goal + velocity + prev_action
    """
    
    def __init__(
        self,
        grid_path: str,
        encoding: Literal['flat', 'binary', 'onehot'] = 'flat',
        include_goal_vector: bool = True,
        include_velocity: bool = True,
        include_prev_action: bool = True,
        n_actions: int = 5  # e.g., forward, back, left, right, stop
    ):
        self.grid_dataset = OccupancyGridDataset(grid_path, encoding)
        self.include_goal_vector = include_goal_vector
        self.include_velocity = include_velocity
        self.include_prev_action = include_prev_action
        self.n_actions = n_actions
        
    def __len__(self) -> int:
        return len(self.grid_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Returns encoded input and metadata dict.
        In actual training, goal/velocity come from simulation.
        """
        grid_encoded = self.grid_dataset[idx]
        raw_grid = self.grid_dataset.get_raw(idx)
        
        # Find goal position in grid (value 3)
        goal_pos = np.argwhere(raw_grid == 3)
        if len(goal_pos) > 0:
            goal_y, goal_x = goal_pos[0]
        else:
            goal_y, goal_x = raw_grid.shape[0] // 2, raw_grid.shape[1] // 2
        
        # Compute relative goal vector (normalized)
        center = np.array(raw_grid.shape) / 2
        goal_vec = np.array([goal_x - center[1], goal_y - center[0]])
        goal_dist = np.linalg.norm(goal_vec) + 1e-6
        goal_vec_norm = goal_vec / goal_dist
        
        components = [grid_encoded]
        
        if self.include_goal_vector:
            # 2 neurons: normalized direction to goal
            components.append(torch.tensor(goal_vec_norm, dtype=torch.float32))
        
        if self.include_velocity:
            # 2 neurons: current velocity (placeholder - comes from sim)
            components.append(torch.zeros(2, dtype=torch.float32))
        
        if self.include_prev_action:
            # n_actions neurons: one-hot previous action (placeholder)
            prev_action = torch.zeros(self.n_actions, dtype=torch.float32)
            components.append(prev_action)
        
        full_input = torch.cat(components)
        
        metadata = {
            'goal_pos': (goal_y, goal_x),
            'goal_vector': goal_vec_norm,
            'grid_shape': raw_grid.shape
        }
        
        return full_input, metadata
    
    @property
    def input_size(self) -> int:
        size = self.grid_dataset.input_size
        if self.include_goal_vector:
            size += 2
        if self.include_velocity:
            size += 2
        if self.include_prev_action:
            size += self.n_actions
        return size


# ============ Usage Examples ============

def example_basic_loading():
    """Basic grid loading."""
    print("=== Basic Loading ===")
    
    # From directory
    dataset = OccupancyGridDataset('./grids', encoding='flat')
    print(f"Loaded {len(dataset)} grids")
    print(f"Input size: {dataset.input_size}")
    print(f"Sample shape: {dataset[0].shape}")
    
    # DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print(f"Batch shape: {batch.shape}")


def example_snn_integration():
    """Example with snnTorch SNN."""
    print("\n=== snnTorch Integration ===")
    
    try:
        import snntorch as snn
        from snntorch import surrogate
    except ImportError:
        print("snnTorch not installed. Install with: pip install snntorch")
        return
    
    # Load dataset
    dataset = OccupancyGridDataset('./grids', encoding='binary')
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Simple SNN
    class NavigationSNN(torch.nn.Module):
        def __init__(self, input_size, hidden_size=128, output_size=5):
            super().__init__()
            beta = 0.9
            spike_grad = surrogate.fast_sigmoid()
            
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            
            self.fc2 = torch.nn.Linear(hidden_size, output_size)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        def forward(self, x, num_steps=25):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            
            spk_out = []
            for _ in range(num_steps):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)
                
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk_out.append(spk2)
            
            return torch.stack(spk_out)  # [T, B, output_size]
    
    model = NavigationSNN(dataset.input_size)
    print(f"Model input size: {dataset.input_size}")
    
    # Forward pass
    batch = next(iter(loader))
    output = model(batch)
    print(f"Output shape: {output.shape}  # [timesteps, batch, actions]")
    
    # Decode action (spike count)
    action_counts = output.sum(dim=0)  # Sum over time
    actions = action_counts.argmax(dim=1)
    print(f"Decoded actions: {actions[:5]}")


def example_full_navigation():
    """Full navigation input with goal/velocity."""
    print("\n=== Full Navigation Dataset ===")
    
    dataset = NavigationDataset(
        './grids',
        encoding='binary',
        include_goal_vector=True,
        include_velocity=True,
        include_prev_action=True,
        n_actions=5
    )
    
    print(f"Total input size: {dataset.input_size}")
    print(f"  Grid (binary): {dataset.grid_dataset.input_size}")
    print(f"  Goal vector: 2")
    print(f"  Velocity: 2")
    print(f"  Prev action: 5")
    
    sample, metadata = dataset[0]
    print(f"\nSample input shape: {sample.shape}")
    print(f"Goal position: {metadata['goal_pos']}")
    print(f"Goal vector: {metadata['goal_vector']}")


if __name__ == "__main__":
    # Generate some test grids first
    import subprocess
    subprocess.run(['python3', 'grid_generator.py', '-n', '50', '-o', './grids', '-p', 'mixed'])
    
    example_basic_loading()
    example_snn_integration()
    example_full_navigation()