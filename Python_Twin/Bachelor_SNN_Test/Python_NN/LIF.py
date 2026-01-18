import numpy as np

class LIF():
    """
        Simple Leaky Integrate-and-Fire (LIF) Neuron with STDP
    """
    def __init__(self, beta=0.9, threshold=2.0, reset=0.5, learning_rate=0.01):
        self.beta = beta                    # Decay factor
        self.threshold = threshold          # Spike threshold value - value that triggers a spike
        self.mem = 0                        # Membrane potential - current value
        self.reset = reset                  # Membrane potential reset value - value after spike
        self.learning_rate = learning_rate  # Learning rate for STDP
        self.spk = 0                        # Spike
        self.eligibility = 0                # Eligibility trace - tracks how recently a spike occured, decays over time
        self.refractory_timer = 0           
        
        # Decay variables
        self.eligibility_decay = 0.99

    def update(self, synaptic_input):
            """
                Update membrane potential and check if a spike should be generated
            """
            # Reset spike from previous timestep
            self.spk = 0

            # Decrese refractory timer if it is above zero
            if self.refractory_timer > 0:
                self.refractory_timer -= 1

            # Track input activity for learning
            self.eligibility = synaptic_input

            # Compute membrane potential
            self.mem = self.beta * self.mem + synaptic_input

            
            # If membrane potential exceeds threshold -> spike
            if self.mem > self.threshold and self.refractory_timer == 0:
                self.spk = 1            # Spike
                self.mem = self.reset   # Reset membrane potential to the reset value
                self.refractory_timer = 3  
            
            # Return spike state
            return self.spk
    

    def update_threshold(self, baseline=1.5, step=0.1, threshold_decay=0.1):
        """ 
            Increase threshold on spike, slowly decay to baseline threshold otherwise
        """
        # Increase threshold if spike
        if self.spk:
            self.threshold += step
        else:
            # Gradually move back down to the baseline
            if self.threshold > baseline:
                self.threshold -= threshold_decay

        self.threshold = np.clip(self.threshold, 1, 10)


    def STDP(self, weight, pre_eligibility, is_winner):
        """
            Basic Spike-Timing-Dependent Plasticity (STDP)
        """
        # Winner strengthens weights based on recent pre-synaptic activity
        if is_winner:
            weight += self.learning_rate * pre_eligibility
                
        return np.clip(weight, 0.1, 0.9)  # Clip the weight to avoid exploding/vanishing weights
    

    def rSTDP(self, weight, pre_eligibility, is_winner, dopamine):
        """
            Reward-modulated STDP using traces
        """
        # Three-factor learning: pre-activity × post-activity × reward
        timing_factor = pre_eligibility * self.eligibility
        
        if is_winner:
            # Strengthen weights proportional to timing correlation and reward
            weight += self.learning_rate * timing_factor * dopamine
        else:
            # Competitive weakening when winner gets positive reward
            if dopamine > 0:
                weight -= self.learning_rate * timing_factor * 0.5
                
        return np.clip(weight, 0.01, 0.9)