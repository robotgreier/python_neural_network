import numpy as np


class LIF:
    """
    Leaky Integrate-and-Fire neuron.

    Args:
            decay: Membrane decay (subtraction based)
            threshold: Magnitude of membrane potential needed to produce a spike
            reset: Value the membrane potential is reset to after spike
    """

    def __init__(self, decay=192, threshold=1024, reset=0):
        # --- Neuron dynamics ---
        self.decay = decay              # Membrane decay (uint8, scaled x256)
        self.threshold = threshold      # uint16, scaled x256
        self.reset = reset
        self.mem = 0                    # uint16, never negative
        self.pre_reset_mem = 0         # Membrane potential before reset, used for WTA
        self.spk = 0

    def update(self, synaptic_input):
        """Membrane update. Returns spike (0 or 1)."""
        self.spk = 0
        self.mem = max(0, self.mem - self.decay) + synaptic_input  # saturating sub → uint16
        self.pre_reset_mem = self.mem  # Cache membrane potential before reset for WTA

        if self.mem >= self.threshold:
            self.spk = 1
            self.mem = self.reset

        return self.spk


class RSTDPSynapse:
    """
    Reward-modulated STDP synapse with rectangular window.

    Supports two modes:
        'rstdp': Weight updates only when apply_reward() is called externally
                 with a dopamine signal. Eligibility trace accumulates STDP
                 events and decays over time, acting as a credit assignment window.
        'stdp':  Weight updates immediately on each spike pairing (dopamine=1.0).
                 Eligibility still decays each step, so the effective update is
                 lr * eligibility at the moment of each spike — not a pure
                 instantaneous step, but a short-window trace-based update.

    Args:
            mode: 'rstdp' (default) or 'stdp'
            lr_shift: Learning rate as right-shift (lr = 1 / 2^lr_shift)
            w_init: Initial weight (random if None)
            t_pre/t_post: Rectangular STDP window widths (timesteps)
                          Pre-before-post within t_pre → LTP (causal)
                          Post-before-pre within t_post → LTD (acausal)
            tau_e_shift: Eligibility decay as right-shift (divide by 2^N each step)
                         Higher = slower decay, longer credit assignment window
            dw_pos/dw_neg: Fixed weight increment/decrement on spike pairing
                           Equivalent to A_plus/A_minus in exponential STDP
            w_min/w_max: Weight clamps
    """

    # Eligibility traces start as disabled/inactive
    DISABLED = -1

    # Decode table: dopamine_code → (s1, s2, s2_enable)
    # Two barrel shifts are summed to produce finer-grained magnitudes without DSP slices.
    DOPAMINE_DECODE = {
        0: (0, 0, False),  # disabled (dopamine_enable handles this, but consistent)
        1: (0, 0, False),  # ×1
        2: (1, 0, False),  # ×2
        3: (1, 0, True),   # ×3
        4: (2, 0, False),  # ×4
        5: (2, 0, True),   # ×5
        6: (2, 1, True),   # ×6
        7: (3, 0, False),  # ×8
    }

    def __init__(self, lr_shift=3, w_init=64,
                 t_pre=2, t_post=3, tau_e_shift=4,
                 dw_pos=64, dw_neg=8,
                 w_min=8, w_max=255,
                 mode='rstdp'):

        self.mode = mode
        self.lr_shift = lr_shift        # learning rate as right-shift: 1/8 = >> 3
        self.weight = w_init if w_init is not None else np.random.randint(77, 205)  # uint8

        # STDP window parameters
        self.t_pre = t_pre
        self.t_post = t_post
        self.tau_e_shift = tau_e_shift
        self.dw_pos = dw_pos
        self.dw_neg = dw_neg

        # Counter-based trace state (pre/post_timer = -1 -> inactive, 0+ = counting)
        self.pre_timer = self.DISABLED
        self.post_timer = self.DISABLED
        self.eligibility = 0   # int16, range [−256, 256]

        # Weight bounds
        self.w_min = w_min
        self.w_max = w_max

    def update_eligibility(self, pre_spike, post_spike):
        """
        Update spike timing counters and eligibility each timestep.

        Uses rectangular STDP windows: if a spike pair occurs within
        the window, eligibility is incremented/decremented by dw_pos/dw_neg.
        Timers run independently so multiple pairings within a window
        are detected — equivalent to parallel trace registers in HDL.

        In 'stdp' mode, weight is updated immediately after eligibility update
        (dopamine=1.0), so no external apply_reward() call is needed.

        Args:
            pre_spike: 1 if pre-synaptic neuron fired, 0 otherwise
            post_spike: 1 if post-synaptic neuron fired, 0 otherwise
        """
        # STDP causality timers
        if self.pre_timer >= 0:
            self.pre_timer += 1
            if self.pre_timer > self.t_pre:
                self.pre_timer = self.DISABLED

        if self.post_timer >= 0:
            self.post_timer += 1
            if self.post_timer > self.t_post:
                self.post_timer = self.DISABLED

        # STDP on pre-spike: start pre timer, check for acausality (LTD)
        if pre_spike:
            if self.post_timer >= 0 and self.post_timer <= self.t_post:
                self.eligibility -= self.dw_neg
            self.pre_timer = 0

        # STDP on post-spike: start post timer, check for causality (LTP)
        if post_spike:
            if self.pre_timer >= 0 and self.pre_timer <= self.t_pre:
                self.eligibility += self.dw_pos
            self.post_timer = 0

        # Decay eligibility via arithmetic right-shift (integer)
        self.eligibility = self.eligibility - (self.eligibility >> self.tau_e_shift)

        # Clamp eligibility to int16 range
        self.eligibility = max(-256, min(256, self.eligibility))

        # In stdp mode, apply weight update immediately (code=1 → ×1 multiplier)
        if self.mode == 'stdp':
            self.apply_reward(dopamine_code=1, dopamine_sign=1, dopamine_enable=1)

    def apply_reward(self, dopamine_code, dopamine_sign, dopamine_enable):
        """
        Apply reward-modulated weight update.

        HDL mapping:
            - dopamine_enable: 1-bit gate — skips update entirely when 0
            - Decode LUT: dopamine_code → (s1, s2, s2_enable)
            - Dual barrel shifters: shifted_1 = |elig| << s1, shifted_2 = |elig| << s2 (gated)
            - Adder: magnitude = shifted_1 + shifted_2
            - >> lr_shift, then add/sub mux, then saturating clamp

        Args:
            dopamine_code:  3-bit index into DOPAMINE_DECODE LUT (0–7).
                            Maps to two shift amounts for finer magnitude resolution.
                            E.g. with lr_shift=3: code=1 → 1/8, code=2 → 1/4,
                            code=3 → 3/8, code=4 → 1/2, code=7 → 1.
            dopamine_sign:  1 = reward (weight + delta), 0 = punishment (weight - delta).
            dopamine_enable: 1 = apply update, 0 = no-op.

        Note: eligibility is treated as magnitude only (abs). Dopamine sign alone controls
              the update direction. This simplifies HDL to a single add/sub mux with no
              sign interaction between dopamine and eligibility.
        """
        if not dopamine_enable:
            return

        s1, s2, s2_enable = self.DOPAMINE_DECODE[dopamine_code]

        magnitude = abs(self.eligibility) << s1
        if s2_enable:
            magnitude += abs(self.eligibility) << s2

        delta_w = magnitude >> self.lr_shift

        if dopamine_sign:
            new_weight = self.weight + delta_w
        else:
            new_weight = self.weight - delta_w

        self.weight = max(self.w_min, min(self.w_max, new_weight))


class SNNLayer:
    """
    Fully-connected SNN with vectorised NumPy state arrays.

    All synapse weights, eligibility traces, and STDP timers are stored as
    (n_outputs, n_inputs) arrays so that one forward pass replaces the original
    n_outputs × n_inputs Python-object loops with a single matrix–vector product
    and element-wise array operations.  The public API is identical to the
    original per-object implementation.

    Args:
            n_inputs: Number of pre-synaptic input neurons
            n_outputs: Number of post-synaptic output neurons
            neuron_params: Dict passed to LIF constructor
            synapse_params: Dict passed to RSTDPSynapse constructor
                            Include 'mode': 'stdp' or 'rstdp' (default)
    """

    def __init__(self, n_inputs, n_outputs, neuron_params=None, synapse_params=None, feedback=False):
        neuron_params  = neuron_params  or {}
        synapse_params = synapse_params or {}

        self.n_inputs  = n_inputs + 1 if feedback else n_inputs   # +1 feedback neuron if enabled
        self.n_outputs = n_outputs
        self.mode      = synapse_params.get('mode', 'rstdp')

        # One extra input neuron whose spike = 1 when all outputs were zero last cycle.
        # Maps to a single D flip-flop in HDL driving one extra synapse column.
        self.feedback      = feedback
        self._feedback_reg = 0   # registered bit: NOR-reduce of previous output

        # --- Neuron parameters & state  (n_outputs,) ---
        self.decay     = neuron_params.get('decay',     192)
        self.threshold = neuron_params.get('threshold', 1024)
        self.reset_val = neuron_params.get('reset',     0)

        self.mem           = np.zeros(n_outputs, dtype=np.int32)
        self.pre_reset_mem = np.zeros(n_outputs, dtype=np.int32)
        self.spk           = np.zeros(n_outputs, dtype=np.int32)

        # --- Synapse parameters & state  (n_outputs, n_inputs) ---
        self.lr_shift    = synapse_params.get('lr_shift',    3)
        self.t_pre       = synapse_params.get('t_pre',       2)
        self.t_post      = synapse_params.get('t_post',      3)
        self.tau_e_shift = synapse_params.get('tau_e_shift', 4)
        self.dw_pos      = synapse_params.get('dw_pos',      64)
        self.dw_neg      = synapse_params.get('dw_neg',      8)
        self.w_min       = synapse_params.get('w_min',       8)
        self.w_max       = synapse_params.get('w_max',       255)

        w_init = synapse_params.get('w_init', None)
        if w_init is None:
            self.weights = np.random.randint(77, 205, size=(n_outputs, self.n_inputs), dtype=np.int32)
        else:
            self.weights = np.full((n_outputs, self.n_inputs), w_init, dtype=np.int32)

        self.eligibility = np.zeros((n_outputs, self.n_inputs), dtype=np.int32)
        self.pre_timer   = np.full((n_outputs, self.n_inputs), -1, dtype=np.int32)
        self.post_timer  = np.full((n_outputs, self.n_inputs), -1, dtype=np.int32)

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(self, input_spikes):
        """
        Process one timestep/frame.

        Args:
            input_spikes: List/array of length n_inputs (0s and 1s)

        Returns:
            List of output spikes (length n_outputs)
        """
        input_arr = np.asarray(input_spikes, dtype=np.int32)
        if self.feedback:
            input_arr = np.append(input_arr, self._feedback_reg)

        # Weighted synaptic input for every output neuron in one matmul
        synaptic_inputs = self.weights @ input_arr          # (n_outputs,)

        # LIF membrane update
        self.mem = np.maximum(0, self.mem - self.decay) + synaptic_inputs
        self.pre_reset_mem = self.mem.copy()                # snapshot before reset
        output_arr = (self.mem >= self.threshold).astype(np.int32)
        self.mem = np.where(output_arr, self.reset_val, self.mem)

        if self.mode == 'stdp':
            winner = self._winner_from_arr(output_arr)
            # Lateral inhibition: suppress losers' membrane
            losers = np.arange(self.n_outputs) != winner
            self.mem[losers] = self.reset_val

            # Only winner sees real pre/post spikes; losers get zeros (traces decay only)
            pre_mat  = np.zeros((self.n_outputs, self.n_inputs), dtype=np.int32)
            post_mat = np.zeros((self.n_outputs, self.n_inputs), dtype=np.int32)
            pre_mat[winner]  = input_arr
            post_mat[winner] = output_arr[winner]
            self._update_eligibility(pre_mat, post_mat)

            # Immediate weight update for all synapses (dopamine_sign=1, dopamine_code=1 → ×1)
            delta_w = np.abs(self.eligibility) >> self.lr_shift
            self.weights = np.clip(self.weights + delta_w, self.w_min, self.w_max)
        else:
            # R-STDP: broadcast input/output spikes across synapse matrix
            pre_mat  = np.broadcast_to(input_arr[np.newaxis, :],     (self.n_outputs, self.n_inputs))
            post_mat = np.broadcast_to(output_arr[:, np.newaxis], (self.n_outputs, self.n_inputs))
            self._update_eligibility(pre_mat, post_mat)

        if self.feedback:
            self._feedback_reg = 1 if not np.any(output_arr) else 0

        return output_arr.tolist()

    # ------------------------------------------------------------------
    # Vectorised eligibility trace update
    # ------------------------------------------------------------------

    def _update_eligibility(self, pre_mat, post_mat):
        """
        Update all (n_outputs × n_inputs) eligibility traces in one pass.

        Preserves the exact same causal ordering as the scalar per-synapse
        version: timers advance → expire → LTD on pre-spike (reset pre_timer)
        → LTP on post-spike using the just-reset pre_timer → decay → clamp.

        Uses in-place numpy ops throughout to avoid temporary array allocations.
        """
        # 1. Advance active timers in-place (no temporaries)
        np.add(self.pre_timer,  1, out=self.pre_timer,  where=self.pre_timer  >= 0)
        np.add(self.post_timer, 1, out=self.post_timer, where=self.post_timer >= 0)

        # 2. Expire timers via boolean fancy-index assignment (in-place, no temporaries)
        self.pre_timer[self.pre_timer   > self.t_pre]  = -1
        self.post_timer[self.post_timer > self.t_post] = -1

        # 3. LTD: pre fires while post_timer is still active (acausal pairing)
        pre_fired = pre_mat.astype(bool)          # one allocation, reused twice below
        np.subtract(self.eligibility, self.dw_neg,
                    out=self.eligibility,
                    where=pre_fired & (self.post_timer >= 0))
        # Reset pre_timer AFTER LTD check — same ordering as scalar version
        self.pre_timer[pre_fired] = 0

        # 4. LTP: post fires while pre_timer is active (causal pairing)
        #    Uses the updated pre_timer from step 3, so simultaneous pre+post → LTP
        post_fired = post_mat.astype(bool)        # one allocation, reused twice below
        np.add(self.eligibility, self.dw_pos,
               out=self.eligibility,
               where=post_fired & (self.pre_timer >= 0))
        self.post_timer[post_fired] = 0

        # 5. Decay eligibility (arithmetic right-shift, same as Python int behaviour)
        self.eligibility -= self.eligibility >> self.tau_e_shift

        # 6. Clamp in-place
        np.clip(self.eligibility, -256, 256, out=self.eligibility)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def winner_takes_all(self, output_spikes):
        """
        Returns index of winning neuron.
        Spiking neurons always beat non-spiking ones.
        Pre-reset membrane potential breaks ties, ensuring index-independent
        results equivalent to a parallel hardware comparator on FPGA.
        """
        return self._winner_from_arr(np.asarray(output_spikes, dtype=np.int32))

    def _winner_from_arr(self, output_arr):
        spiking = np.where(output_arr == 1)[0]
        if len(spiking) == 1:
            return int(spiking[0])
        elif len(spiking) > 1:
            return int(spiking[np.argmax(self.pre_reset_mem[spiking])])
        else:
            return int(np.argmax(self.pre_reset_mem))

    # Decode table: dopamine_code → (s1, s2, s2_enable)
    DOPAMINE_DECODE = [
        (0, 0, False),  # code 0: disabled
        (0, 0, False),  # code 1: ×1
        (1, 0, False),  # code 2: ×2
        (1, 0, True),   # code 3: ×3
        (2, 0, False),  # code 4: ×4
        (2, 0, True),   # code 5: ×5
        (2, 1, True),   # code 6: ×6
        (3, 0, False),  # code 7: ×8
    ]

    def apply_reward(self, dopamine_code, dopamine_sign, winner_idx):
        """
        Apply reward only to the winning neuron's synapses.
        No-op in 'stdp' mode (weights already updated in forward()).

        Args:
            dopamine_code:  3-bit index into DOPAMINE_DECODE LUT (0–7).
            dopamine_sign:  1 = reward, 0 = punishment.
            winner_idx:     Row index of the winning neuron.
        """
        if self.mode == 'stdp':
            return

        s1, s2, s2_enable = self.DOPAMINE_DECODE[dopamine_code]

        mag = np.abs(self.eligibility[winner_idx]) << s1
        if s2_enable:
            mag = mag + (np.abs(self.eligibility[winner_idx]) << s2)

        delta_w = mag >> self.lr_shift

        if dopamine_sign:
            new_row = self.weights[winner_idx] + delta_w
        else:
            new_row = self.weights[winner_idx] - delta_w
        np.clip(new_row, self.w_min, self.w_max, out=self.weights[winner_idx])

    def get_weights(self):
        """Return weight matrix as numpy array [n_outputs x n_inputs]."""
        return self.weights.copy()

    def load_weights(self, weight_file="weights.mem"):
        """Loads and sets the weights of the current model from file"""
        with open(weight_file, "r") as f:
            lines = f.readlines()
        idx = 0
        for i in range(self.n_outputs):
            for j in range(self.n_inputs):
                self.weights[i, j] = int(lines[idx].strip(), 16)
                idx += 1

    def reset_state(self):
        """Reset the state of all neurons and synaptic traces in the network."""
        self.mem[:]           = 0
        self.spk[:]           = 0
        self.pre_reset_mem[:] = 0
        self.eligibility[:]   = 0
        self.pre_timer[:]     = -1
        self.post_timer[:]    = -1
        self._feedback_reg    = 0
