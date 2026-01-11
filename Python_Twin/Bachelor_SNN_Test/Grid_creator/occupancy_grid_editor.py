#!/usr/bin/env python3
"""
Interactive Occupancy Grid Editor
Creates 15x15 grids with 2-bit cell values:
  0 = Free (white)
  1 = Occupied (black)
  2 = Unknown (gray)
  3 = Reserved/Goal (green)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import json
import os
from datetime import datetime


class OccupancyGridEditor:
    CELL_VALUES = {
        0: ("Free", "#FFFFFF"),
        1: ("Occupied", "#1a1a1a"),
        2: ("Unknown", "#888888"),
        3: ("Goal", "#4CAF50")
    }
    
    def __init__(self, root, grid_size=15, cell_px=35):
        self.root = root
        self.grid_size = grid_size
        self.cell_px = cell_px
        self.current_brush = 1  # Default: Occupied
        self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.cells = {}
        self.is_drawing = False
        
        self.setup_ui()
        self.draw_grid()
    
    def setup_ui(self):
        self.root.title("Occupancy Grid Editor")
        self.root.resizable(False, False)
        
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        # Brush selection
        ttk.Label(toolbar, text="Brush:").pack(side=tk.LEFT, padx=(0, 5))
        self.brush_var = tk.IntVar(value=1)
        
        for val, (name, color) in self.CELL_VALUES.items():
            rb = ttk.Radiobutton(
                toolbar, text=name, variable=self.brush_var, 
                value=val, command=self.update_brush
            )
            rb.pack(side=tk.LEFT, padx=3)
        
        # Spacer
        ttk.Frame(toolbar, width=20).pack(side=tk.LEFT)
        
        # Action buttons
        ttk.Button(toolbar, text="Clear", command=self.clear_grid).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Fill Unknown", command=self.fill_unknown).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Add Border", command=self.add_border).pack(side=tk.LEFT, padx=2)
        
        # Canvas for grid
        canvas_size = self.grid_size * self.cell_px
        self.canvas = tk.Canvas(
            main_frame, 
            width=canvas_size, 
            height=canvas_size,
            bg="#CCCCCC",
            highlightthickness=1,
            highlightbackground="#333"
        )
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Keyboard shortcuts
        self.root.bind("1", lambda e: self.set_brush(0))
        self.root.bind("2", lambda e: self.set_brush(1))
        self.root.bind("3", lambda e: self.set_brush(2))
        self.root.bind("4", lambda e: self.set_brush(3))
        self.root.bind("<Control-s>", lambda e: self.save_grid())
        self.root.bind("<Control-o>", lambda e: self.load_grid())
        
        # Bottom toolbar
        bottom_bar = ttk.Frame(main_frame)
        bottom_bar.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(bottom_bar, text="Save Grid", command=self.save_grid).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom_bar, text="Load Grid", command=self.load_grid).pack(side=tk.LEFT, padx=2)
        ttk.Button(bottom_bar, text="Export NumPy", command=self.export_numpy).pack(side=tk.LEFT, padx=2)
        
        # Status
        self.status_var = tk.StringVar(value="Ready | Keys: 1-4 for brush, Ctrl+S save, Ctrl+O load")
        ttk.Label(bottom_bar, textvariable=self.status_var).pack(side=tk.RIGHT)
    
    def draw_grid(self):
        self.canvas.delete("all")
        self.cells = {}
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x1 = col * self.cell_px
                y1 = row * self.cell_px
                x2 = x1 + self.cell_px
                y2 = y1 + self.cell_px
                
                val = self.grid[row, col]
                color = self.CELL_VALUES[val][1]
                
                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline="#555555",
                    width=1
                )
                self.cells[(row, col)] = rect
    
    def get_cell_coords(self, event):
        col = event.x // self.cell_px
        row = event.y // self.cell_px
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            return row, col
        return None, None
    
    def paint_cell(self, row, col):
        if row is None or col is None:
            return
        self.grid[row, col] = self.current_brush
        color = self.CELL_VALUES[self.current_brush][1]
        self.canvas.itemconfig(self.cells[(row, col)], fill=color)
    
    def on_click(self, event):
        self.is_drawing = True
        row, col = self.get_cell_coords(event)
        self.paint_cell(row, col)
    
    def on_drag(self, event):
        if self.is_drawing:
            row, col = self.get_cell_coords(event)
            self.paint_cell(row, col)
    
    def on_release(self, event):
        self.is_drawing = False
    
    def update_brush(self):
        self.current_brush = self.brush_var.get()
        name = self.CELL_VALUES[self.current_brush][0]
        self.status_var.set(f"Brush: {name}")
    
    def set_brush(self, val):
        self.brush_var.set(val)
        self.update_brush()
    
    def clear_grid(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.draw_grid()
        self.status_var.set("Grid cleared")
    
    def fill_unknown(self):
        self.grid = np.full((self.grid_size, self.grid_size), 2, dtype=np.uint8)
        self.draw_grid()
        self.status_var.set("Filled with unknown")
    
    def add_border(self):
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        self.draw_grid()
        self.status_var.set("Border added")
    
    def save_grid(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialfile=f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        if filepath:
            data = {
                "grid_size": self.grid_size,
                "resolution_cm": 10,
                "grid": self.grid.tolist()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            self.status_var.set(f"Saved: {os.path.basename(filepath)}")
    
    def load_grid(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("NumPy", "*.npy"), ("All files", "*.*")]
        )
        if filepath:
            try:
                if filepath.endswith('.npy'):
                    self.grid = np.load(filepath).astype(np.uint8)
                else:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    self.grid = np.array(data["grid"], dtype=np.uint8)
                self.draw_grid()
                self.status_var.set(f"Loaded: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")
    
    def export_numpy(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NumPy", "*.npy"), ("All files", "*.*")],
            initialfile=f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        )
        if filepath:
            np.save(filepath, self.grid)
            self.status_var.set(f"Exported: {os.path.basename(filepath)}")


def main():
    root = tk.Tk()
    app = OccupancyGridEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()