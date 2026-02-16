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
    def __init__(self, threshold_edges, n_bins_x=6, n_bins_y=4, frame_width=640, frame_height=480, fast_threshold=20):
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
    '''Simple SNN: 12 inputs -> 3 outputs (Left, Forward, Right). Winner-take-all: only highest voltage fires.'''
    def __init__(self, n_inputs=24, n_outputs=3, beta=0.70, threshold=5.0, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = np.random.uniform(0.3, 0.7, (n_inputs, n_outputs))
        self.output_neurons = [LIF(beta=beta, threshold=threshold, learning_rate=learning_rate) 
                              for _ in range(n_outputs)]
        self.input_eligibility = np.zeros(n_inputs)
        self.input_eligibility_decay = 0.90
        
    def forward(self, input_spikes):
        '''Process input spikes. Winner-take-all: only neuron with highest membrane potential can fire.'''
        input_spikes = np.array(input_spikes)
        
        # Update input eligibility
        self.input_eligibility = self.input_eligibility_decay * self.input_eligibility + input_spikes
        self.input_eligibility = np.clip(self.input_eligibility, -10, 10)
        
        # Compute synaptic inputs
        synaptic_inputs = np.dot(input_spikes, self.weights)
        
        # Update neurons (get membrane potentials before spike)
        membrane_potentials_before = []
        for i, neuron in enumerate(self.output_neurons):
            # Update membrane but don't spike yet
            neuron.eligibility = neuron.eligibility_decay * neuron.eligibility + synaptic_inputs[i]
            neuron.eligibility = np.clip(neuron.eligibility, -10, 10)
            neuron.mem = neuron.beta * neuron.mem + synaptic_inputs[i]
            membrane_potentials_before.append(neuron.mem)
        
        # Winner-take-all: only highest membrane potential can spike
        winner_idx = np.argmax(membrane_potentials_before)
        output_spikes = [0] * self.n_outputs
        
        # Check if winner exceeds threshold
        if membrane_potentials_before[winner_idx] > self.output_neurons[winner_idx].threshold:
            output_spikes[winner_idx] = 1
            self.output_neurons[winner_idx].spk = 1
            self.output_neurons[winner_idx].mem = self.output_neurons[winner_idx].reset
            self.output_neurons[winner_idx].eligibility = self.output_neurons[winner_idx].reset
        
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
        
        # Create visualization - reorganized layout
        fig = plt.figure(figsize=(22, 12))
        gs = GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.4, 
                     height_ratios=[1.2, 1, 1.5], width_ratios=[1, 1, 1, 1, 1])
        fig.suptitle(f'Frame {frame_number} - SNN Processing Pipeline', fontsize=18, fontweight='bold')
        
        # === ROW 1: INPUT PROCESSING (left to right flow) ===
        # 1. Frame with keypoints
        ax1 = fig.add_subplot(gs[0, 0:2])
        frame_rgb = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
        frame_with_kp = cv.drawKeypoints(frame_rgb, kp_list, None, color=(0, 255, 0), flags=0)
        bin_width = self.frame_width // self.n_bins_x
        bin_height = self.frame_height // self.n_bins_y
        for i in range(self.n_bins_x + 1):
            cv.line(frame_with_kp, (i * bin_width, 0), (i * bin_width, self.frame_height), (255, 0, 0), 2)
        for j in range(self.n_bins_y + 1):
            cv.line(frame_with_kp, (0, j * bin_height), (self.frame_width, j * bin_height), (255, 0, 0), 2)
        ax1.imshow(frame_with_kp)
        ax1.set_title(f'① Camera Input: {len(kp_list)} Keypoints', fontsize=12, fontweight='bold', loc='left')
        ax1.axis('off')
        
        # 2. Keypoint density heatmap - FLIPPED with origin='upper'
        ax2 = fig.add_subplot(gs[0, 2])
        im = ax2.imshow(hist.T, cmap='YlOrRd', aspect='auto', origin='upper')  # Changed to 'upper'
        ax2.set_title('② Spatial Binning', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X →')
        ax2.set_ylabel('Y ↓')
        ax2.set_xticks(range(self.n_bins_x))
        ax2.set_yticks(range(self.n_bins_y))
        for i in range(self.n_bins_x):
            for j in range(self.n_bins_y):
                ax2.text(i, j, f'{int(hist[i, j])}', ha="center", va="center", 
                        color="black", fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Keypoints', fraction=0.046)
        
        # 3. Input spike train
        ax3 = fig.add_subplot(gs[0, 3:])
        if spike_train:
            bin_labels = [f'{i}' for i in range(len(spike_train))]
            spike_colors = ['#FF4444' if s == 1 else '#DDDDDD' for s in spike_train]
            bars = ax3.bar(bin_labels, spike_train, color=spike_colors, edgecolor='black', linewidth=1.5, alpha=0.9)
            ax3.set_title(f'③ Event-Driven Encoding: {sum(spike_train)}/{len(spike_train)} spikes', 
                         fontsize=12, fontweight='bold', loc='left')
            ax3.set_xlabel('Input Neuron')
            ax3.set_ylabel('Spike')
            ax3.set_ylim(-0.1, 1.3)
            ax3.set_yticks([0, 1])
            ax3.grid(axis='y', alpha=0.3)
        
        # === ROW 2: SYNAPTIC WEIGHTS ===
        ax4 = fig.add_subplot(gs[1, :])
        weights = self.snn.weights
        im_weights = ax4.imshow(weights.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax4.set_title('④ Synaptic Weight Matrix (Input Neurons → Output Neurons)', 
                     fontsize=12, fontweight='bold', loc='left')
        ax4.set_xlabel('Input Neurons (Spatial Bins)')
        ax4.set_ylabel('Output Neurons')
        ax4.set_yticks([0, 1, 2])
        ax4.set_yticklabels(['Left Turn', 'Forward', 'Right Turn'], fontsize=11)
        ax4.set_xticks(range(0, self.snn.n_inputs))
        ax4.set_xticklabels(range(self.snn.n_inputs), fontsize=9)
        
        # Add weight values
        for i in range(self.snn.n_outputs):
            for j in range(self.snn.n_inputs):
                text_color = 'white' if weights[j, i] < 0.5 else 'black'
                ax4.text(j, i, f'{weights[j, i]:.2f}', ha="center", va="center", 
                        color=text_color, fontsize=8)
        
        plt.colorbar(im_weights, ax=ax4, label='Weight Strength', fraction=0.046, pad=0.02)
        
        # === ROW 3: NETWORK DYNAMICS (left to right: computation flow) ===
        output_labels = ['Left\nTurn', 'Forward', 'Right\nTurn']
        colors_syn = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        # Synaptic inputs
        ax5 = fig.add_subplot(gs[2, 0])
        bars_syn = ax5.bar(range(3), synaptic_inputs, color=colors_syn, edgecolor='black', linewidth=2)
        ax5.set_title('⑤ Synaptic\nInputs', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Current\n(Σ w·s)', fontsize=9)
        ax5.set_xticks(range(3))
        ax5.set_xticklabels(output_labels, fontsize=9)
        ax5.axhline(y=self.snn.output_neurons[0].threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax5.grid(axis='y', alpha=0.3)
        for i, val in enumerate(synaptic_inputs):
            ax5.text(i, val + 0.1, f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Membrane potentials
        ax6 = fig.add_subplot(gs[2, 1])
        bars_mem = ax6.bar(range(3), membrane_potentials, color=colors_syn, edgecolor='black', linewidth=2)
        ax6.set_title('⑥ Membrane\nPotentials', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Voltage (V)', fontsize=9)
        ax6.set_xticks(range(3))
        ax6.set_xticklabels(output_labels, fontsize=9)
        ax6.axhline(y=self.snn.output_neurons[0].threshold, color='red', linestyle='--', 
                   linewidth=2, label='Threshold', alpha=0.7)
        ax6.grid(axis='y', alpha=0.3)
        ax6.legend(fontsize=8, loc='upper right')
        for i, val in enumerate(membrane_potentials):
            ax6.text(i, val + 0.1, f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Eligibility traces
        ax7 = fig.add_subplot(gs[2, 2])
        bars_elig = ax7.bar(range(3), eligibilities, color=colors_syn, edgecolor='black', linewidth=2)
        ax7.set_title('⑦ Eligibility\nTraces', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Eligibility', fontsize=9)
        ax7.set_xticks(range(3))
        ax7.set_xticklabels(output_labels, fontsize=9)
        ax7.grid(axis='y', alpha=0.3)
        for i, val in enumerate(eligibilities):
            ax7.text(i, max(val, 0) + 0.05, f'{val:.2f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        # Output spikes (winner-take-all highlighted)
        ax8 = fig.add_subplot(gs[2, 3])
        spike_colors_out = ['#00FF00' if s == 1 else '#FF4444' for s in output_spikes]
        bars_out = ax8.bar(range(3), output_spikes, color=spike_colors_out, 
                          edgecolor='black', linewidth=3, alpha=0.95)
        ax8.set_title('⑧ Output Spikes\n(Winner-Take-All)', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Spike', fontsize=9)
        ax8.set_xticks(range(3))
        ax8.set_xticklabels(output_labels, fontsize=9)
        ax8.set_ylim(-0.1, 1.3)
        ax8.set_yticks([0, 1])
        ax8.grid(axis='y', alpha=0.3)
        for i, spike in enumerate(output_spikes):
            label = '🔥 FIRE!' if spike == 1 else 'Silent'
            color = 'white' if spike == 1 else 'black'
            ax8.text(i, 0.5, label, ha='center', va='center', 
                    fontsize=9, fontweight='bold', color=color)
        
        # Motor command visualization
        ax9 = fig.add_subplot(gs[2, 4])
        ax9.set_xlim(-1, 1)
        ax9.set_ylim(-1, 1)
        ax9.axis('off')
        ax9.set_title('⑨ Motor\nCommand', fontsize=11, fontweight='bold')
        
        # Draw robot representation
        robot = plt.Circle((0, 0), 0.3, color='gray', ec='black', linewidth=2)
        ax9.add_patch(robot)
        
        # Draw direction arrow based on output
        arrow_props = dict(arrowstyle='->', lw=4, color='red')
        if output_spikes[0] == 1:  # Left
            ax9.annotate('', xy=(-0.7, 0), xytext=(0, 0), arrowprops=arrow_props)
            ax9.text(0, -0.7, 'TURN LEFT', ha='center', fontsize=10, fontweight='bold')
        elif output_spikes[1] == 1:  # Forward
            ax9.annotate('', xy=(0, 0.7), xytext=(0, 0), arrowprops=arrow_props)
            ax9.text(0, -0.7, 'GO FORWARD', ha='center', fontsize=10, fontweight='bold')
        elif output_spikes[2] == 1:  # Right
            ax9.annotate('', xy=(0.7, 0), xytext=(0, 0), arrowprops=arrow_props)
            ax9.text(0, -0.7, 'TURN RIGHT', ha='center', fontsize=10, fontweight='bold')
        else:
            ax9.text(0, -0.7, 'NO ACTION', ha='center', fontsize=10, fontweight='bold', color='gray')
        
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



import numpy as np

visualizer = VisualFrameShowcaseWithSNN(
    threshold_edges=np.linspace(0, 249, 50),
    n_bins_x=4,
    n_bins_y=3,
    frame_width=640,
    frame_height=480,
    fast_threshold=20
)

visualizer.process_sequence(
    image_folder=r"C:\Users\eirik\Desktop\Bachelor\fpga_neural_network\Python_implementation\Images",
    output_folder="visualizations_network",
    n_frames=50,
    max_keypoints=500
)