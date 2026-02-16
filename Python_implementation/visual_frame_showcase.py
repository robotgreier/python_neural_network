import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os
import sys

class LIF:
    def __init__(self, beta=0.9, threshold=2.0, reset=0.0, learning_rate=0.1):
        self.beta = beta
        self.threshold = threshold
        self.mem = 0.0
        self.reset = reset
        self.learning_rate = learning_rate
        self.spk = 0
        self.eligibility = 0.0
        self.eligibility_decay = 0.90

    def update(self, synaptic_input):
        self.spk = 0
        self.eligibility = self.eligibility_decay * self.eligibility + synaptic_input
        self.eligibility = np.clip(self.eligibility, -10, 10)
        self.mem = self.beta * self.mem + synaptic_input
        if self.mem > self.threshold:
            self.spk = 1
            self.mem = self.reset
            self.eligibility = self.reset
        return self.spk

class FrameProcessor:
    def __init__(self, threshold_edges, n_bins_x=4, n_bins_y=3, frame_width=640, frame_height=480, fast_threshold=20):
        self.threshold_edges = threshold_edges
        self.n_bins_x = n_bins_x
        self.n_bins_y = n_bins_y
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.kp_counts_old = None
        self.fast = cv.FastFeatureDetector_create(threshold=fast_threshold)
        self.fast.setNonmaxSuppression(True)

    def encoder_keypoints_event_driven(self, new_val, prev_val):
        if new_val == prev_val:
            return 0
        new_idx = np.digitize(new_val, self.threshold_edges)
        prev_idx = np.digitize(prev_val, self.threshold_edges)
        return 1 if new_idx != prev_idx else 0

    def process_and_encode_frame(self, frame_path, response_cutoff=0.0, max_keypoints=500):
        kp_counts = []
        spike_train = []
        frame = cv.imread(frame_path, cv.IMREAD_GRAYSCALE)
        if frame is None:
            return []
        keypoints = self.fast.detect(frame, None)
        kp_list = [kp for kp in keypoints if kp.response >= response_cutoff]
        if len(kp_list) > max_keypoints:
            kp_list = sorted(kp_list, key=lambda x: x.response, reverse=True)[:max_keypoints]
        x_data = [kp.pt[0] for kp in kp_list]
        y_data = [kp.pt[1] for kp in kp_list]
        hist, x_edges, y_edges = np.histogram2d(x=x_data, y=y_data, bins=[self.n_bins_x, self.n_bins_y],
                                                 range=[[0, self.frame_width], [0, self.frame_height]])
        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                kp_counts.append(float(hist[i, j]))
        if self.kp_counts_old is not None:
            for new_val, prev_val in zip(kp_counts, self.kp_counts_old):
                spike_train.append(self.encoder_keypoints_event_driven(new_val, prev_val))
        else:
            spike_train = [0] * (self.n_bins_x * self.n_bins_y)
        self.kp_counts_old = kp_counts
        return spike_train

class SimpleNavigationSNN:
    '''Simple SNN for navigation: 12 input neurons -> 3 output neurons (Left, Forward, Right)'''
    def __init__(self, n_inputs=12, n_outputs=3, beta=0.999, threshold=2.0, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # Initialize weights randomly
        self.weights = np.random.uniform(0.3, 0.7, (n_inputs, n_outputs))
        # Create output neurons
        self.output_neurons = [LIF(beta=beta, threshold=threshold, learning_rate=learning_rate) 
                              for _ in range(n_outputs)]
        # Input neuron eligibility traces
        self.input_eligibility = np.zeros(n_inputs)
        self.input_eligibility_decay = 0.90
        
    def forward(self, input_spikes):
        '''Process input spikes through the network'''
        input_spikes = np.array(input_spikes)
        
        # Update input eligibility traces
        self.input_eligibility = self.input_eligibility_decay * self.input_eligibility + input_spikes
        self.input_eligibility = np.clip(self.input_eligibility, -10, 10)
        
        # Compute synaptic inputs for each output neuron
        synaptic_inputs = np.dot(input_spikes, self.weights)
        
        # Update output neurons
        output_spikes = []
        for i, neuron in enumerate(self.output_neurons):
            spike = neuron.update(synaptic_inputs[i])
            output_spikes.append(spike)
        
        return output_spikes, synaptic_inputs
    
    def get_membrane_potentials(self):
        return [n.mem for n in self.output_neurons]
    
    def get_eligibilities(self):
        return [n.eligibility for n in self.output_neurons]

class VisualFrameShowcaseWithSNN:
    def __init__(self, threshold_edges, n_bins_x=4, n_bins_y=3, frame_width=640, frame_height=480, fast_threshold=20):
        self.processor = FrameProcessor(threshold_edges, n_bins_x, n_bins_y, frame_width, frame_height, fast_threshold)
        self.n_bins_x = n_bins_x
        self.n_bins_y = n_bins_y
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fast_threshold = fast_threshold
        # Initialize SNN
        self.snn = SimpleNavigationSNN(n_inputs=n_bins_x*n_bins_y, n_outputs=3)
        
    def visualize_frame_with_network(self, frame_path, frame_number, output_path=None, max_keypoints=500):
        '''Visualize frame processing AND network internals'''
        frame = cv.imread(frame_path, cv.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"ERROR: Failed to load {frame_path}")
            return None
        
        # Process frame
        fast = cv.FastFeatureDetector_create(threshold=self.fast_threshold)
        fast.setNonmaxSuppression(True)
        keypoints = fast.detect(frame, None)
        kp_list = [kp for kp in keypoints if kp.response >= 0.0]
        if len(kp_list) > max_keypoints:
            kp_list = sorted(kp_list, key=lambda x: x.response, reverse=True)[:max_keypoints]
        
        x_data = [kp.pt[0] for kp in kp_list]
        y_data = [kp.pt[1] for kp in kp_list]
        hist, _, _ = np.histogram2d(x=x_data, y=y_data, bins=[self.n_bins_x, self.n_bins_y],
                                   range=[[0, self.frame_width], [0, self.frame_height]])
        
        spike_train = self.processor.process_and_encode_frame(frame_path, max_keypoints=max_keypoints)
        
        # Process through SNN
        output_spikes, synaptic_inputs = self.snn.forward(spike_train)
        membrane_potentials = self.snn.get_membrane_potentials()
        eligibilities = self.snn.get_eligibilities()
        
        # Create visualization with more subplots
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)
        fig.suptitle(f'Frame {frame_number} - Complete SNN Processing Pipeline', fontsize=18, fontweight='bold')
        
        # === ROW 1: Visual Processing ===
        # 1. Frame with keypoints
        ax1 = fig.add_subplot(gs[0, :2])
        frame_rgb = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
        frame_with_kp = cv.drawKeypoints(frame_rgb, kp_list, None, color=(0, 255, 0), flags=0)
        bin_width = self.frame_width // self.n_bins_x
        bin_height = self.frame_height // self.n_bins_y
        for i in range(self.n_bins_x + 1):
            cv.line(frame_with_kp, (i * bin_width, 0), (i * bin_width, self.frame_height), (255, 0, 0), 2)
        for j in range(self.n_bins_y + 1):
            cv.line(frame_with_kp, (0, j * bin_height), (self.frame_width, j * bin_height), (255, 0, 0), 2)
        ax1.imshow(frame_with_kp)
        ax1.set_title(f'Keypoints: {len(kp_list)}', fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # 2. Heatmap
        ax2 = fig.add_subplot(gs[0, 2])
        im = ax2.imshow(hist.T, cmap='YlOrRd', aspect='auto', origin='lower')
        ax2.set_title('Keypoint Density', fontsize=11, fontweight='bold')
        ax2.set_xlabel('X Bins')
        ax2.set_ylabel('Y Bins')
        ax2.set_xticks(range(self.n_bins_x))
        ax2.set_yticks(range(self.n_bins_y))
        for i in range(self.n_bins_x):
            for j in range(self.n_bins_y):
                ax2.text(i, j, f'{int(hist[i, j])}', ha="center", va="center", color="black", fontsize=9)
        plt.colorbar(im, ax=ax2, label='Count', fraction=0.046)
        
        # 3. Input spike train
        ax3 = fig.add_subplot(gs[0, 3])
        if spike_train:
            bin_labels = [f'{i}' for i in range(len(spike_train))]
            spike_colors = ['#FF4444' if s == 1 else '#CCCCCC' for s in spike_train]
            bars = ax3.bar(bin_labels, spike_train, color=spike_colors, edgecolor='black', linewidth=1, alpha=0.9)
            ax3.set_title(f'Input Spikes: {sum(spike_train)}/{len(spike_train)}', fontsize=11, fontweight='bold')
            ax3.set_xlabel('Input Neuron')
            ax3.set_ylabel('Spike')
            ax3.set_ylim(-0.1, 1.3)
            ax3.set_yticks([0, 1])
            ax3.grid(axis='y', alpha=0.3)
        
        # === ROW 2: Synaptic Weights ===
        ax4 = fig.add_subplot(gs[1, :])
        weights = self.snn.weights
        im_weights = ax4.imshow(weights.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax4.set_title('Synaptic Weight Matrix (Input → Output)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Input Neurons (Spatial Bins)', fontsize=10)
        ax4.set_ylabel('Output Neurons', fontsize=10)
        ax4.set_yticks([0, 1, 2])
        ax4.set_yticklabels(['Left', 'Forward', 'Right'])
        ax4.set_xticks(range(0, self.n_inputs, 2))
        
        # Add weight values as text
        for i in range(self.snn.n_outputs):
            for j in range(self.snn.n_inputs):
                text_color = 'white' if weights[j, i] < 0.5 else 'black'
                ax4.text(j, i, f'{weights[j, i]:.2f}', ha="center", va="center", 
                        color=text_color, fontsize=7)
        
        cbar = plt.colorbar(im_weights, ax=ax4, label='Weight Strength', fraction=0.046, pad=0.02)
        
        # === ROW 3: Network Activity ===
        # 4. Synaptic inputs
        ax5 = fig.add_subplot(gs[2, 0])
        output_labels = ['Left', 'Forward', 'Right']
        colors_syn = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        bars_syn = ax5.bar(output_labels, synaptic_inputs, color=colors_syn, edgecolor='black', linewidth=2)
        ax5.set_title('Synaptic Inputs', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Current (Σ w·s)')
        ax5.axhline(y=self.snn.output_neurons[0].threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax5.grid(axis='y', alpha=0.3)
        ax5.legend(fontsize=8)
        for bar, val in zip(bars_syn, synaptic_inputs):
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 5. Membrane potentials
        ax6 = fig.add_subplot(gs[2, 1])
        bars_mem = ax6.bar(output_labels, membrane_potentials, color=colors_syn, edgecolor='black', linewidth=2)
        ax6.set_title('Membrane Potentials', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Voltage (V)')
        ax6.axhline(y=self.snn.output_neurons[0].threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax6.grid(axis='y', alpha=0.3)
        ax6.legend(fontsize=8)
        for bar, val in zip(bars_mem, membrane_potentials):
            ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 6. Eligibility traces
        ax7 = fig.add_subplot(gs[2, 2])
        bars_elig = ax7.bar(output_labels, eligibilities, color=colors_syn, edgecolor='black', linewidth=2)
        ax7.set_title('Eligibility Traces', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Eligibility')
        ax7.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars_elig, eligibilities):
            ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 7. Output spikes
        ax8 = fig.add_subplot(gs[2, 3])
        spike_colors_out = ['#00FF00' if s == 1 else '#FF4444' for s in output_spikes]
        bars_out = ax8.bar(output_labels, output_spikes, color=spike_colors_out, edgecolor='black', linewidth=2, alpha=0.9)
        ax8.set_title('Output Spikes', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Spike')
        ax8.set_ylim(-0.1, 1.3)
        ax8.set_yticks([0, 1])
        ax8.grid(axis='y', alpha=0.3)
        for bar, spike in zip(bars_out, output_spikes):
            label = 'FIRE!' if spike == 1 else 'Silent'
            ax8.text(bar.get_x() + bar.get_width()/2., 0.5, label,
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # === ROW 4: Network Diagram ===
        ax9 = fig.add_subplot(gs[3, :])
        ax9.set_xlim(0, 10)
        ax9.set_ylim(0, 10)
        ax9.axis('off')
        ax9.set_title('Network Architecture & Information Flow', fontsize=12, fontweight='bold')
        
        # Draw input layer
        input_y_positions = np.linspace(2, 8, self.snn.n_inputs)
        for i, (y_pos, spike) in enumerate(zip(input_y_positions, spike_train)):
            color = '#FF4444' if spike == 1 else '#CCCCCC'
            circle = plt.Circle((2, y_pos), 0.2, color=color, ec='black', linewidth=2)
            ax9.add_patch(circle)
            if i % 2 == 0:
                ax9.text(1.4, y_pos, f'I{i}', fontsize=7, ha='right', va='center')
        
        # Draw output layer
        output_y_positions = [2.5, 5, 7.5]
        output_names = ['LEFT', 'FWD', 'RIGHT']
        for i, (y_pos, spike, name) in enumerate(zip(output_y_positions, output_spikes, output_names)):
            color = '#00FF00' if spike == 1 else colors_syn[i]
            circle = plt.Circle((8, y_pos), 0.3, color=color, ec='black', linewidth=3)
            ax9.add_patch(circle)
            ax9.text(8.8, y_pos, name, fontsize=10, ha='left', va='center', fontweight='bold')
        
        # Draw connections (sample a subset for clarity)
        connection_indices = [0, 3, 6, 9]  # Show only some connections
        for in_idx in connection_indices:
            for out_idx in range(self.snn.n_outputs):
                weight = self.snn.weights[in_idx, out_idx]
                alpha = weight  # Opacity based on weight strength
                color = 'green' if weight > 0.5 else 'orange'
                ax9.plot([2.2, 7.7], [input_y_positions[in_idx], output_y_positions[out_idx]],
                        color=color, alpha=alpha, linewidth=weight*3)
        
        # Add labels
        ax9.text(2, 9.2, 'Input Layer\n(12 neurons)', fontsize=10, ha='center', fontweight='bold')
        ax9.text(8, 9.2, 'Output Layer\n(3 neurons)', fontsize=10, ha='center', fontweight='bold')
        ax9.text(5, 0.5, 'Connection strength: Weight × Input Spike', fontsize=9, ha='center', style='italic')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        return fig

    def process_sequence(self, image_folder, output_folder, n_frames=50, max_keypoints=500, save_individual=True):
        if not os.path.exists(image_folder):
            print(f"\n❌ ERROR: Image folder not found: {image_folder}")
            return
        
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
        if not image_files:
            print(f"\n❌ ERROR: No PNG images found")
            return
        
        print(f"\n✓ Found {len(image_files)} images")
        os.makedirs(output_folder, exist_ok=True)
        print(f"📁 Processing up to {n_frames} frames...\n")
        
        processed_count = 0
        for frame_idx in range(1, n_frames + 1):
            frame_path = os.path.join(image_folder, f"img{frame_idx:04d}.png")
            if not os.path.exists(frame_path):
                continue
            
            output_path = os.path.join(output_folder, f"frame_{frame_idx:04d}_network.png") if save_individual else None
            fig = self.visualize_frame_with_network(frame_path, frame_idx, output_path, max_keypoints)
            if fig:
                plt.close(fig)
                processed_count += 1
        
        print(f"\n✅ Done! Processed {processed_count} frames")
        print(f"   Output folder: {output_folder}/")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  SNN Frame Processing + Network Internals Visualization")
    print("="*70)
    
    if len(sys.argv) > 1:
        image_folder = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) > 2 else "visualizations_network"
    else:
        image_folder = fr"C:\Users\eirik\Desktop\Bachelor\fpga_neural_network\Python_implementation\Images"
        output_folder = "visualizations_network"
    
    print(f"\nConfiguration:")
    print(f"  Image folder: {image_folder}")
    print(f"  Output folder: {output_folder}")
    
    thresholds = np.linspace(0, 249, 50)
    visualizer = VisualFrameShowcaseWithSNN(
        threshold_edges=thresholds,
        n_bins_x=4,
        n_bins_y=3,
        frame_width=640,
        frame_height=480,
        fast_threshold=20
    )
    
    visualizer.process_sequence(
        image_folder=image_folder,
        output_folder=output_folder,
        n_frames=50,
        max_keypoints=500,
        save_individual=True
    )


