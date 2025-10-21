"""
Basal Ganglia Actor Network
Implements the CBGT (Cortico-Basal Ganglia-Thalamic) circuit from Ao et al. 2024
"""

import torch
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import neuron, functional


class BasalGangliaActor(nn.Module):
    """
    Spiking Basal Ganglia Actor Network with direct and indirect pathways.

    Network structure (4 parallel action channels):
        PFC [16] → D1-MSNs [4] → GPi [4] ─────────┐
                                                   ├→ Thalamus [4] → PMC [4] → Action
                   D2-MSNs [4] → GPe [4] → STN [4]┘

    Args:
        n_states: Number of states (default: 16 for FrozenLake)
        n_actions: Number of actions (default: 4)
        tau: LIF time constant (default: 1.0)
        v_reset: Reset potential (default: 0.0)
        v_th: Firing threshold (default: 1.0)
        dt: Timestep (default: 1.0)
        T: Simulation timesteps per decision (default: 2000)
    """

    def __init__(self, n_states=16, n_actions=4,
                 tau=1.0, v_reset=0.0, v_th=1.0, dt=1.0, T=2000):
        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.tau = tau
        self.v_reset = v_reset
        self.v_th = v_th
        self.dt = dt
        self.T = T  # Simulation timesteps per decision

        # ==================== Neuron Populations ====================
        # Using SpikingJelly LIFNode for optimized CUDA acceleration
        # Note: SpikingJelly requires tau > 1.0, so we use tau=1.00000000001 (minimal change from paper's tau=1.0)
        tau_sj = 1.00000000001  # SpikingJelly tau parameter (just above 1.0 constraint)

        # PFC: 16 neurons (state encoding)
        self.pfc = neuron.LIFNode(tau=tau_sj, v_threshold=v_th, v_reset=v_reset,
                                   detach_reset=True, step_mode='s')

        # Striatum: 4 D1-MSNs (direct pathway) + 4 D2-MSNs (indirect pathway)
        self.d1_msn = neuron.LIFNode(tau=tau_sj, v_threshold=v_th, v_reset=v_reset,
                                      detach_reset=True, step_mode='s')
        self.d2_msn = neuron.LIFNode(tau=tau_sj, v_threshold=v_th, v_reset=v_reset,
                                      detach_reset=True, step_mode='s')

        # GPe: 4 neurons (indirect pathway) - has tonic activity
        self.gpe = neuron.LIFNode(tau=tau_sj, v_threshold=v_th, v_reset=v_reset,
                                   detach_reset=True, step_mode='s')

        # STN: 4 neurons (indirect pathway) - has tonic activity
        self.stn = neuron.LIFNode(tau=tau_sj, v_threshold=v_th, v_reset=v_reset,
                                   detach_reset=True, step_mode='s')

        # GPi: 4 neurons (output nucleus, receives from both pathways) - has tonic activity
        self.gpi = neuron.LIFNode(tau=tau_sj, v_threshold=v_th, v_reset=v_reset,
                                   detach_reset=True, step_mode='s')

        # Thalamus: 4 neurons - has tonic activity
        self.thalamus = neuron.LIFNode(tau=tau_sj, v_threshold=v_th, v_reset=v_reset,
                                        detach_reset=True, step_mode='s')

        # PMC (Premotor Cortex): 4 neurons (action output)
        self.pmc = neuron.LIFNode(tau=tau_sj, v_threshold=v_th, v_reset=v_reset,
                                   detach_reset=True, step_mode='s')

        # ==================== Baseline/Tonic Currents ====================
        # These nuclei have intrinsic pacemaker activity that can be modulated
        # Baseline current keeps them active even without input
        self.baseline_gpe = 1.5  # GPe is tonically active
        self.baseline_stn = 1.8  # STN has high tonic firing
        self.baseline_gpi = 1.5  # GPi is tonically active (disinhibits thalamus)
        self.baseline_thal = 1.2  # Thalamus baseline
        self.baseline_pmc = 0.5  # PMC baseline (lower, as it's cortical)

        # ==================== Learnable Weights (Cortico-Striatal) ====================
        # These are the ONLY learnable weights, updated by plasticity rule
        self.w_pfc_d1 = nn.Parameter(torch.zeros(n_states, n_actions))
        self.w_pfc_d2 = nn.Parameter(torch.zeros(n_states, n_actions))

        # Initialize cortico-striatal weights (Section 4.1, Eq. 5)
        self._init_corticostriatal_weights()

        # ==================== Fixed Weights (BG Pathways) ====================
        # Apply W1/W2 templates (Eq. 5) to create structured action channel preferences
        # This ensures stable competitive dynamics and PMC firing

        # Templates from Equation 5:
        # W1 = (w0, w1, w2, w3) where w0 > w1 > w2 > w3 (descending)
        # W2 = (w3, w2, w1, w0) (ascending, reversed)
        w1_template = torch.tensor([0.20, 0.12, 0.07, 0.03])  # descending preference
        w2_template = torch.tensor([0.03, 0.07, 0.12, 0.20])  # ascending preference

        # Direct pathway: D1-MSN → GPi (inhibitory)
        # Use W1 pattern: stronger inhibition for higher-priority actions
        w_d1_gpi = torch.diag(-1.0 * w1_template * 5.0)  # Scale for stronger effect
        self.register_buffer('w_d1_gpi', w_d1_gpi)

        # Indirect pathway connections:
        # D2-MSN → GPe (inhibitory)
        # Use W1 pattern: structured inhibition
        w_d2_gpe = torch.diag(-1.0 * w1_template * 5.0)
        self.register_buffer('w_d2_gpe', w_d2_gpe)

        # GPe → STN (inhibitory)
        # Use W1 pattern scaled down
        w_gpe_stn = torch.diag(-0.8 * w1_template * 5.0)
        self.register_buffer('w_gpe_stn', w_gpe_stn)

        # STN → GPi (excitatory)
        # Use W1 pattern but excitatory
        w_stn_gpi = torch.diag(1.2 * w1_template * 5.0)
        self.register_buffer('w_stn_gpi', w_stn_gpi)

        # GPi → Thalamus (inhibitory)
        # Use W2 pattern (reversed) to counterbalance
        w_gpi_thal = torch.diag(-1.0 * w2_template * 5.0)
        self.register_buffer('w_gpi_thal', w_gpi_thal)

        # Thalamus → PMC (excitatory)
        # Use W1 pattern for final action preference
        w_thal_pmc = torch.diag(1.0 * w1_template * 5.0)
        self.register_buffer('w_thal_pmc', w_thal_pmc)

    def _init_corticostriatal_weights(self):
        """
        Initialize PFC→Striatum weights according to Section 4.1

        From paper: "One is randomly initialized, representing the model's
        random selection of actions at the beginning of training"

        Both W_PFC_D1 and W_PFC_D2 are randomly initialized with positive
        values. These will be learned via the plasticity rule (Eq. 3).

        Note: Weights must be large enough to generate spikes in downstream neurons.
        With one-hot PFC encoding (single neuron firing per state) and V_th=1.0,
        we need weights > 1.0 to ensure MSNs can spike.
        """
        # Random initialization centered around 1.5 with std 0.3
        # This ensures most weights are in [1.0, 2.0] range
        # Using absolute value of Gaussian to ensure all positive
        self.w_pfc_d1.data = torch.abs(torch.randn(self.n_states, self.n_actions) * 0.3 + 1.5)
        self.w_pfc_d2.data = torch.abs(torch.randn(self.n_states, self.n_actions) * 0.3 + 1.5)

        # No normalization - let plasticity rule handle weight evolution

    def _normalize_weights(self, target_norm=0.42):
        """L1-normalize each row of cortico-striatal weights"""
        with torch.no_grad():
            # Normalize D1 weights
            row_sums_d1 = self.w_pfc_d1.sum(dim=1, keepdim=True)
            self.w_pfc_d1.data = self.w_pfc_d1 * (target_norm / row_sums_d1)

            # Normalize D2 weights
            row_sums_d2 = self.w_pfc_d2.sum(dim=1, keepdim=True)
            self.w_pfc_d2.data = self.w_pfc_d2 * (target_norm / row_sums_d2)

    def reset_states(self, batch_size=1):
        """Reset all neuron membrane potentials using SpikingJelly"""
        functional.reset_net(self)

    def encode_state(self, state, T=None):
        """
        Encode discrete state as PFC spike pattern (one-hot, constant spiking)

        Args:
            state: int, state index (0-15 for FrozenLake)
            T: number of timesteps (default: self.T)

        Returns:
            pfc_input: [T, n_states] - constant spike for selected neuron
        """
        if T is None:
            T = self.T

        pfc_input = torch.zeros(T, self.n_states, device=self.w_pfc_d1.device)
        # Set input current above threshold to ensure spiking
        # With V_th=1.0 and LIF dynamics V_new = I, need I > 1.0
        pfc_input[:, state] = 1.5  # Constant current above threshold

        return pfc_input

    def forward(self, state, return_all_activities=False):
        """
        Forward pass through CBGT circuit for T timesteps

        Args:
            state: int or Tensor, environment state (0-15)
            return_all_activities: if True, return spike counts for all layers

        Returns:
            action: int, selected action (argmax of PMC spikes)
            pmc_spike_counts: [n_actions], PMC spike counts
            all_activities: dict (optional), spike counts for all layers
        """
        # Reset neuron states
        self.reset_states()

        # Encode state as PFC input
        if isinstance(state, int):
            pfc_input = self.encode_state(state, self.T)
        else:
            # Assuming state is already a tensor
            pfc_input = state

        # Storage for spike counts
        pfc_spikes_total = torch.zeros(self.n_states, device=self.w_pfc_d1.device)
        d1_spikes_total = torch.zeros(self.n_actions, device=self.w_pfc_d1.device)
        d2_spikes_total = torch.zeros(self.n_actions, device=self.w_pfc_d1.device)
        gpe_spikes_total = torch.zeros(self.n_actions, device=self.w_pfc_d1.device)
        stn_spikes_total = torch.zeros(self.n_actions, device=self.w_pfc_d1.device)
        gpi_spikes_total = torch.zeros(self.n_actions, device=self.w_pfc_d1.device)
        thal_spikes_total = torch.zeros(self.n_actions, device=self.w_pfc_d1.device)
        pmc_spikes_total = torch.zeros(self.n_actions, device=self.w_pfc_d1.device)

        # Run dynamics for T timesteps
        for t in range(self.T):
            # PFC activity (input layer)
            pfc_spikes = self.pfc(pfc_input[t])
            pfc_spikes_total += pfc_spikes

            # Cortico-striatal projections (learnable weights)
            d1_current = torch.matmul(pfc_spikes, self.w_pfc_d1)
            d2_current = torch.matmul(pfc_spikes, self.w_pfc_d2)

            # Striatum
            d1_spikes = self.d1_msn(d1_current)
            d2_spikes = self.d2_msn(d2_current)
            d1_spikes_total += d1_spikes
            d2_spikes_total += d2_spikes

            # Direct pathway: D1 → GPi
            gpi_current_direct = torch.matmul(d1_spikes, self.w_d1_gpi)

            # Indirect pathway: D2 → GPe → STN → GPi
            gpe_current = torch.matmul(d2_spikes, self.w_d2_gpe) + self.baseline_gpe
            gpe_spikes = self.gpe(gpe_current)
            gpe_spikes_total += gpe_spikes

            stn_current = torch.matmul(gpe_spikes, self.w_gpe_stn) + self.baseline_stn
            stn_spikes = self.stn(stn_current)
            stn_spikes_total += stn_spikes

            gpi_current_indirect = torch.matmul(stn_spikes, self.w_stn_gpi)

            # GPi receives from both pathways + baseline
            gpi_current = gpi_current_direct + gpi_current_indirect + self.baseline_gpi
            gpi_spikes = self.gpi(gpi_current)
            gpi_spikes_total += gpi_spikes

            # GPi → Thalamus
            thal_current = torch.matmul(gpi_spikes, self.w_gpi_thal) + self.baseline_thal
            thal_spikes = self.thalamus(thal_current)
            thal_spikes_total += thal_spikes

            # Thalamus → PMC
            pmc_current = torch.matmul(thal_spikes, self.w_thal_pmc) + self.baseline_pmc
            pmc_spikes = self.pmc(pmc_current)
            pmc_spikes_total += pmc_spikes

        # Action selection: argmax of PMC spike counts
        action = torch.argmax(pmc_spikes_total).item()

        if return_all_activities:
            all_activities = {
                'pfc': pfc_spikes_total,
                'd1': d1_spikes_total,
                'd2': d2_spikes_total,
                'gpe': gpe_spikes_total,
                'stn': stn_spikes_total,
                'gpi': gpi_spikes_total,
                'thalamus': thal_spikes_total,
                'pmc': pmc_spikes_total
            }
            return action, pmc_spikes_total, all_activities

        return action, pmc_spikes_total

    def get_corticostriatal_weights(self):
        """Return current cortico-striatal weights for plasticity updates"""
        return self.w_pfc_d1, self.w_pfc_d2

    def clamp_weights(self, w_max=0.4):
        """Clamp cortico-striatal weights to [0, w_max]"""
        with torch.no_grad():
            self.w_pfc_d1.data = torch.clamp(self.w_pfc_d1.data, min=0.0, max=w_max)
            self.w_pfc_d2.data = torch.clamp(self.w_pfc_d2.data, min=0.0, max=w_max)
