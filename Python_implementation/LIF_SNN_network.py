import numpy as np


class LIF:
    """
    Leaky Integrate-and-Fire neuron with R-STDP learning.
    """

    def __init__(self, beta=0.9, threshold=2.0, reset=0.0):
        # --- Neuron dynamics ---
        self.beta = beta              # Membrane decay (leak factor)
        self.threshold = threshold
        self.reset = reset
        self.mem = 0.0
        self.spk = 0

        self.post_trace = 0.0         # Tracks recent post-synaptic spikes

    def update(self, synaptic_input):
        """Standard LIF membrane update. Returns spike (0 or 1)."""
        self.spk = 0
        self.mem = self.beta * self.mem + synaptic_input

        if self.mem >= self.threshold:
            self.spk = 1
            self.mem = self.reset

        return self.spk


class RSTDPSynapse:
    """
    Reward-modulated STDP synapse.

    Args:
            learning_rate: Scales the weight update
            w_init: Initial weight (random if None)
            tau_pre/tau_post: Time constants for pre/post synaptic traces (frames)
                              Controls the STDP window width
            tau_e: Eligibility trace decay time constant (frames)
            A_plus/A_minus: STDP potentiation/depression amplitudes
                            A_minus > A_plus gives slight depression bias
                            which helps prevent runaway excitation
            w_min/w_max: Weight bounds
    """

    def __init__(self, learning_rate=0.01, w_init=None, tau_pre=5, tau_post=5, tau_e=15, A_plus=0.01, A_minus=0.015, w_min=0.01, w_max=1.0):

        self.learning_rate = learning_rate
        self.weight = w_init if w_init is not None else np.random.uniform(0.1, 0.5)

        # STDP window parameters
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.tau_e = tau_e
        self.A_plus = A_plus
        self.A_minus = A_minus

        # Pre-computed decay constants
        self.pre_decay = np.exp(-1.0 / self.tau_pre)
        self.post_decay = np.exp(-1.0 / self.tau_post)
        self.e_decay = np.exp(-1.0 / self.tau_e)

        # Trace state variables
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self.eligibility = 0.0

        # Weight bounds
        self.w_min = w_min
        self.w_max = w_max

    def update_traces_and_eligibility(self, pre_spike, post_spike):
        """
        Update synaptic traces and eligibility each timestep.
        
        Args:
            pre_spike: 1 if pre-synaptic neuron fired, 0 otherwise
            post_spike: 1 if post-synaptic neuron fired, 0 otherwise  
            dt: timestep duration (frames)
        """
        # Decay traces exponentially
        self.pre_trace *= self.pre_decay
        self.post_trace *= self.post_decay

        # STDP event accumulation into eligibility
        stdp_update = 0.0

        if pre_spike:
            # Pre before post (LTD): depress based on recent post activity
            stdp_update -= self.A_minus * self.post_trace
            self.pre_trace += 1.0

        if post_spike:
            # Post after pre (LTP): potentiate based on recent pre activity
            stdp_update += self.A_plus * self.pre_trace
            self.post_trace += 1.0

        self.eligibility = self.e_decay * self.eligibility + stdp_update

    def apply_reward(self, dopamine):
        """
        Apply reward-modulated weight update.
        
        Args:
            dopamine: Reward signal. Positive reinforces correlated activity, negative punishes it.
        """
        delta_w = self.learning_rate * dopamine * self.eligibility
        self.weight = np.clip(self.weight + delta_w, self.w_min, self.w_max)


class SNNLayer:
    """
    A single fully-connected SNN layer with R-STDP learning.
    Manages neurons and their incoming synapses.

    Args:
            n_input: Number of pre-synaptic input neurons
            n_output: Number of post-synaptic output neurons
            neuron_params: Dict passed to LIF constructor
            synapse_params: Dict passed to RSTDPSynapse constructor
    """

    def __init__(self, n_inputs, n_outputs, neuron_params=None, synapse_params=None):
        neuron_params = neuron_params or {}
        synapse_params = synapse_params or {}

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Create output neurons
        self.neurons = [LIF(**neuron_params) for _ in range(n_outputs)]

        # Create synapse matrix: synapses[post]x[pre]
        self.synapses = [
            [RSTDPSynapse(**synapse_params) for _ in range(n_inputs)]
            for _ in range(n_outputs)
        ]

    def forward(self, input_spikes):
        """
        Process one timestep.
        
        Args:
            input_spikes: List/array of length n_input (0s and 1s)
            
        Returns:
            List of output spikes (length n_output)
        """
        output_spikes = []

        for j, neuron in enumerate(self.neurons):
            # Compute weighted synaptic input
            synaptic_input = sum(
                self.synapses[j][i].weight * input_spikes[i]
                for i in range(self.n_inputs)
            )

            # Update neuron
            spike = neuron.update(synaptic_input)
            output_spikes.append(spike)

            # Update all incoming synapse traces and eligibility
            for i in range(self.n_inputs):
                self.synapses[j][i].update_traces_and_eligibility(
                    pre_spike=input_spikes[i],
                    post_spike=spike,
                )

        return output_spikes
    

    def winner_takes_all(self, output_spikes):
        """
        Returns index of winning neuron.
        """
        spiking = [i for i, s in enumerate(output_spikes) if s == 1]
        
        if len(spiking) == 1:
            return spiking[0]
        elif len(spiking) > 1:
            return spiking[0]
        else:
            # No spikes: highest membrane potential
            return int(np.argmax([n.mem for n in self.neurons]))
        

    def apply_reward(self, dopamine, winner_idx):
        """Apply reward only to the winning neuron's synapses."""
        for i in range(self.n_inputs):
            # Reinforce/punish winner
            self.synapses[winner_idx][i].apply_reward(dopamine)
            

    def get_weights(self):
        """Return weight matrix as numpy array [n_output x n_input]."""
        return np.array([
            [self.synapses[j][i].weight for i in range(self.n_inputs)]
            for j in range(self.n_outputs)
        ])
    
    def reset_state(self):
        """Reset the state of all neurons in the network"""
        for n in self.neurons:
            n.mem = 0.0
            n.spk = 0