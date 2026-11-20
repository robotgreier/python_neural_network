# SPIKE – Python SNN Software Emulation

Software emulation of the spiking neural network (SNN) used in **SPIKE**, an autonomous taskbot that compares the same reward-modulated SNN running as Python software on a Raspberry Pi 4 and as SystemVerilog hardware on an Arty A7-100T FPGA.

This repository contains:

- the NumPy implementation of the network (leaky integrate-and-fire neurons, reward-modulated STDP, winner-take-all output), and
- the notebooks used for hyperparameter and preprocessing tuning.

The same network file is used inside the ROS 2 node [`python_snn_node`](https://github.com/robotgreier/ros2/tree/main/python_snn_node) in the [ros2](https://github.com/robotgreier/ros2) repository, and is mirrored bit-for-bit by the FPGA implementation in [fpga_project](https://github.com/robotgreier/fpga_project).

## Relation to the paper

| Paper section | Content in this repository |
|---|---|
| §2.2 From Sensing to Action (Eq. 1–2) | `Python_implementation/LIF_SNN_network.py` – LIF neuron, R-STDP synapse, layer with WTA and NOR feedback input |
| §2.3 One Network, Two Hardware Substrates (Table 2) | `Python_implementation/LIF_SNN_*_tuning*.ipynb` – two-stage randomized grid search |
| Table 2 values | `Python_implementation/CSV_results/` – tuning results |

## Repository structure

```
Python_implementation/
├── LIF_SNN_network.py                       # SNN implementation (LIF, Synapse, Layer)
├── LIF_SNN_preprocessing_tuning.ipynb       # Stage 1: preprocessing + SNN parameters
├── LIF_SNN_hyperparameter_tuning.ipynb      # Stage 2: SNN hyperparameters
├── LIF_SNN_hyperparameter_tuning_hidden.ipynb  # Exploratory: hidden-layer variant
├── weights.mem                              # Example weight file (hex, one value per line)
└── CSV_results/                             # Tuning results
test/                                        # Learning-rule checks and parameter sweeps
requirements.txt
```

## Installation

Tested with Python 3.12.

```bash
git clone https://github.com/robotgreier/python_neural_network.git
cd python_neural_network
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Minimal example with the parameters used in the paper (26 sensory inputs + 1 feedback input, 4 output neurons):

```python
import numpy as np
from Python_implementation.LIF_SNN_network import Layer

np.random.seed(42)

layer = Layer(
    n_inputs=26,          # 12 keypoint bins + 5 distance bits + 9 ArUco bits
    n_outputs=4,          # left, reverse, right, forward
    feedback=True,        # adds the NOR feedback input (27 synapses per neuron)
    neuron_params=dict(decay=400, threshold=750, reset=100, refractory=1),
    synapse_params=dict(mode="rstdp", lr_shift=5, t_pre=3, t_post=1,
                        tau_e_shift=2, dw_pos=10, dw_neg=8,
                        w_min=16, w_max=254, w_init=-1),
)

spikes_in = np.random.randint(0, 2, size=26)
spikes_out = layer.forward(spikes_in)
winner = layer.winner_takes_all(spikes_out)   # -1 if no neuron fired
if winner >= 0:
    layer.apply_reward(dopamine=5, winner_idx=winner)
```

To reproduce the tuning, open the notebooks in order: `LIF_SNN_preprocessing_tuning.ipynb`, then `LIF_SNN_hyperparameter_tuning.ipynb`. The recorded spike trains used for tuning are not included in this repository because of their size.

## Related repositories

- [ros2](https://github.com/robotgreier/ros2) – ROS 2 workspace for the taskbot (sensing, encoding, reward, motor control, power logging)
- [fpga_project](https://github.com/robotgreier/fpga_project) – SystemVerilog implementation for the Arty A7-100T

## Citation

E. Holm, H. Torkildsen, G. J. Arason, J. K. Dall, B. Nossum, and A. Muhtaroglu, "R-STDP-Based Biomimetic Robot Navigation: A System-Level Energy and Latency Analysis," submitted to *Biomimetics* (MDPI), 2026.

## License

Apache License 2.0 – see [LICENSE](LICENSE).
