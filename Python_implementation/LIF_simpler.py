import numpy as np

class LIF_simpler():
    def __init__(self, beta=0.9, threshold=2.0, reset=0.5, learning_rate=0.1):
        self.beta = beta
        self.threshold = threshold
        self.mem = 0.0
        self.reset = reset
        self.learning_rate = learning_rate
        self.spk = 0
        self.eligibility = 0.0
        self.eligibility_decay = 0.90

    def update(self, synaptic_input):
        # Reset spike
        self.spk = 0

        # Update eligibility trace
        self.eligibility = self.eligibility_decay * self.eligibility + synaptic_input
        self.eligibility = np.clip(self.eligibility, -200, 200)

        # Update membrane potential
        self.mem = self.beta * self.mem + synaptic_input

        # Check for spike
        if self.mem > self.threshold:
            self.spk = 1
            self.mem = self.reset
        
        return self.spk

    def STDP(self, weight, pre_eligibility, is_winner):
        # Standard STDP for offline/unsupervised learning
        timing_factor = np.clip(pre_eligibility * self.eligibility, -300, 300)
        
        if is_winner:
            weight += self.learning_rate * timing_factor
        else:
            weight -= self.learning_rate * timing_factor * 0.5
                
        return np.clip(weight, 0.01, 1.0)

    def rSTDP(self, weight, pre_eligibility, is_winner, dopamine):
        # Reward-modulated STDP for online learning with reward signal
        timing_factor = np.clip(pre_eligibility * self.eligibility, -300, 300)
        effective_lr = self.learning_rate / (1 + abs(timing_factor) / 200)
        
        if is_winner:
            weight += effective_lr * timing_factor * dopamine
        else:
            if dopamine > 0:
                weight -= effective_lr * timing_factor * 4.0
            else:
                weight -= effective_lr * timing_factor * 0.1
                
        return np.clip(weight, 0.01, 1.0)