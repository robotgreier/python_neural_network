#!/usr/bin/env python3
"""
Visualization for SNN Navigation Decisions
Shows grid, robot position, and action selection over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import torch
from IPython.display import HTML

# Action definitions
ACTIONS = {
    0: ('Forward', (0, -1), '↑'),
    1: ('Backward', (0, 1), '↓'),
    2: ('Left', (-1, 0), '←'),
    3: ('Right', (1, 0), '→'),
}

# Grid colormap: Free=white, Occupied=black, Unknown=gray, Goal=green
GRID_CMAP = ListedColormap(['#FFFFFF', '#1a1a1a', '#888888', '#4CAF50'])


def visualize_decision(grid: np.ndarray, spike_counts: torch.Tensor, 
                       robot_pos: tuple = None, figsize=(10, 4)):
    """
    Show grid and action decision side by side.
    
    Args:
        grid: Raw occupancy grid (15x15) with values 0-3
        spike_counts: Output spike counts per action [4]
        robot_pos: (x, y) robot position, defaults to center
    """
    if robot_pos is None:
        robot_pos = (grid.shape[1] // 2, grid.shape[0] // 2)
    
    spike_counts = spike_counts.detach().cpu().numpy().flatten()
    action_idx = spike_counts.argmax()
    action_name, (dx, dy), arrow = ACTIONS[action_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Grid visualization
    ax1 = axes[0]
    ax1.imshow(grid, cmap=GRID_CMAP, vmin=0, vmax=3)
    ax1.plot(robot_pos[0], robot_pos[1], 'bo', markersize=15, label='Robot')
    ax1.arrow(robot_pos[0], robot_pos[1], dx*0.8, dy*0.8, 
              head_width=0.3, head_length=0.2, fc='blue', ec='blue')
    ax1.set_title('Occupancy Grid')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend(loc='upper right')
    
    # Spike counts bar chart
    ax2 = axes[1]
    colors = ['#2196F3' if i == action_idx else '#90CAF9' for i in range(4)]
    bars = ax2.bar([ACTIONS[i][0] for i in range(4)], spike_counts, color=colors)
    ax2.set_ylabel('Spike Count')
    ax2.set_title(f'Action Selection: {action_name}')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, spike_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{int(count)}', ha='center', va='bottom')
    
    # Action arrows legend
    ax3 = axes[2]
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Selected Action')
    
    # Draw large arrow
    ax3.annotate('', xy=(dx*0.5, -dy*0.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=4))
    ax3.text(0, -0.8, f'{action_name}\n{arrow}', ha='center', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return action_idx


def visualize_trajectory(grid: np.ndarray, actions: list, 
                        start_pos: tuple = None, figsize=(6, 6)):
    """
    Show trajectory of robot over multiple timesteps.
    
    Args:
        grid: Raw occupancy grid
        actions: List of action indices over time
        start_pos: Starting position (x, y)
    """
    if start_pos is None:
        start_pos = (grid.shape[1] // 2, grid.shape[0] // 2)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid, cmap=GRID_CMAP, vmin=0, vmax=3)
    
    # Trace path
    x, y = start_pos
    path_x, path_y = [x], [y]
    
    for action_idx in actions:
        _, (dx, dy), _ = ACTIONS[action_idx]
        x = np.clip(x + dx, 0, grid.shape[1] - 1)
        y = np.clip(y + dy, 0, grid.shape[0] - 1)
        
        # Stop if hit obstacle
        if grid[int(y), int(x)] == 1:
            x, y = path_x[-1], path_y[-1]  # Revert
        
        path_x.append(x)
        path_y.append(y)
    
    # Plot path
    ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
    ax.plot(path_x[0], path_y[0], 'go', markersize=12, label='Start')
    ax.plot(path_x[-1], path_y[-1], 'bs', markersize=12, label='End')
    
    # Arrows along path
    for i in range(len(path_x) - 1):
        dx = path_x[i+1] - path_x[i]
        dy = path_y[i+1] - path_y[i]
        if dx != 0 or dy != 0:
            ax.arrow(path_x[i], path_y[i], dx*0.6, dy*0.6,
                    head_width=0.2, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
    
    ax.set_title(f'Robot Trajectory ({len(actions)} steps)')
    ax.legend(loc='upper right')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()


def animate_navigation(grid: np.ndarray, actions: list, 
                       start_pos: tuple = None, interval: int = 500):
    """
    Animate robot navigation step by step.
    Returns HTML animation for Jupyter.
    
    Args:
        grid: Raw occupancy grid
        actions: List of action indices
        start_pos: Starting (x, y) position
        interval: Milliseconds between frames
    """
    if start_pos is None:
        start_pos = (grid.shape[1] // 2, grid.shape[0] // 2)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap=GRID_CMAP, vmin=0, vmax=3)
    
    robot_marker, = ax.plot([], [], 'bo', markersize=20)
    path_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.5)
    action_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                          fontsize=12, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat'))
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Precompute path
    x, y = start_pos
    positions = [(x, y)]
    for action_idx in actions:
        _, (dx, dy), _ = ACTIONS[action_idx]
        new_x = np.clip(x + dx, 0, grid.shape[1] - 1)
        new_y = np.clip(y + dy, 0, grid.shape[0] - 1)
        
        if grid[int(new_y), int(new_x)] != 1:
            x, y = new_x, new_y
        
        positions.append((x, y))
    
    def init():
        robot_marker.set_data([], [])
        path_line.set_data([], [])
        action_text.set_text('')
        return robot_marker, path_line, action_text
    
    def animate(frame):
        path_x = [p[0] for p in positions[:frame+1]]
        path_y = [p[1] for p in positions[:frame+1]]
        
        robot_marker.set_data([positions[frame][0]], [positions[frame][1]])
        path_line.set_data(path_x, path_y)
        
        if frame > 0:
            action_name = ACTIONS[actions[frame-1]][0]
            action_text.set_text(f'Step {frame}: {action_name}')
        else:
            action_text.set_text('Start')
        
        return robot_marker, path_line, action_text
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                         frames=len(positions), interval=interval, blit=True)
    plt.close()
    
    return HTML(anim.to_jshtml())


def visualize_membrane_and_spikes(mem_rec: torch.Tensor, spk_rec: torch.Tensor,
                                   figsize=(12, 4)):
    """
    Plot membrane potential and spike raster for output neurons.
    
    Args:
        mem_rec: Membrane recordings [timesteps, batch, neurons]
        spk_rec: Spike recordings [timesteps, batch, neurons]
    """
    mem = mem_rec.detach().cpu().numpy().squeeze()  # [timesteps, neurons]
    spk = spk_rec.detach().cpu().numpy().squeeze()
    
    n_steps, n_neurons = mem.shape
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Membrane potential over time
    ax1 = axes[0]
    for i in range(n_neurons):
        ax1.plot(mem[:, i], label=ACTIONS[i][0], alpha=0.8)
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Membrane Potential')
    ax1.set_title('Output Neuron Membrane Potentials')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Spike raster
    ax2 = axes[1]
    for i in range(n_neurons):
        spike_times = np.where(spk[:, i] > 0)[0]
        ax2.scatter(spike_times, [i]*len(spike_times), marker='|', s=100)
    ax2.set_yticks(range(n_neurons))
    ax2.set_yticklabels([ACTIONS[i][0] for i in range(n_neurons)])
    ax2.set_xlabel('Timestep')
    ax2.set_title('Output Spike Raster')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()


# Convenience function for notebook use
def run_and_visualize(model_fn, grid_raw: np.ndarray, spike_input: torch.Tensor,
                      n_steps: int = 100):
    """
    Run simulation and visualize results.
    
    Args:
        model_fn: Function that takes (spike_input, step) and returns (mem_rec, spk_rec)
        grid_raw: Raw occupancy grid for visualization
        spike_input: Spike-encoded input [n_steps, batch, n_inputs]
        n_steps: Number of simulation steps
    """
    mem2_rec, spk2_rec = model_fn(spike_input, n_steps)
    
    spike_counts = spk2_rec.sum(dim=0)
    
    print("=== Simulation Results ===")
    print(f"Total output spikes: {spk2_rec.sum().item()}")
    print(f"Spike counts per action: {spike_counts.squeeze().tolist()}")
    
    visualize_membrane_and_spikes(mem2_rec, spk2_rec)
    action = visualize_decision(grid_raw, spike_counts)
    
    return action, spike_counts