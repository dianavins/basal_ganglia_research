# Basal Ganglia Model - PyTorch Implementation

Implementation of **"A Spiking Neural Network Action Decision Method Inspired by Basal Ganglia"** by Ao et al. (2024).

## Overview

This is a faithful PyTorch implementation of the basal ganglia-inspired spiking neural network from the Ao et al. 2024 paper. The model uses an Actor-Critic architecture with biologically plausible spiking neurons and a novel cortico-striatal plasticity learning rule.

## Project Structure

```
Ao_model_pytorch/
├── models/
│   ├── __init__.py
│   ├── lif_neuron.py          # Leaky Integrate-and-Fire neurons (Eq. 1)
│   ├── basal_ganglia.py       # Actor network (CBGT circuit)
│   ├── critic.py              # Critic network for state values
│   └── plasticity.py          # Cortico-striatal plasticity (Eq. 3)
├── config.py                   # All hyperparameters from paper
├── test_phase1_phase2.py      # Component tests
└── README.md                   # This file
```

## Implementation Details

### Phase 1: Core SNN Components ✓

1. **LIF Neuron Layer** (`models/lif_neuron.py`)
   - Implements Equation 1 from the paper
   - Discrete-time Euler integration
   - Parameters: τ=1.0, V_reset=0.0, V_th=1.0, dt=1.0

2. **Basal Ganglia Actor** (`models/basal_ganglia.py`)
   - **Network structure** (Figure 2b):
     - PFC: 16 neurons (state encoding)
     - Striatum: 4 D1-MSNs (direct pathway) + 4 D2-MSNs (indirect pathway)
     - GPe, STN, GPi, Thalamus, PMC: 4 neurons each (action channels)

   - **Learnable weights** (plastic, updated by Eq. 3):
     - W_PFC_D1: [16, 4]
     - W_PFC_D2: [16, 4]
     - Initialized per Section 4.1, Equation 5

   - **Fixed weights** (diagonal, not learned):
     - D1→GPi: -1.0 (inhibitory)
     - D2→GPe: -1.0 (inhibitory)
     - GPe→STN: -0.8 (inhibitory)
     - STN→GPi: +1.2 (excitatory)
     - GPi→Thal: -1.0 (inhibitory)
     - Thal→PMC: +1.0 (excitatory)

### Phase 2: Learning Components ✓

3. **Critic Network** (`models/critic.py`)
   - Standard MLP: Linear(16→64)→ReLU→Linear(64→64)→ReLU→Linear(64→1)
   - Computes state values V(s)
   - Trained with TD error: δ(t) = r + γV(s') - V(s)

4. **Cortico-Striatal Plasticity** (`models/plasticity.py`)
   - Implements Equation 3 with three terms:
     1. **Dopamine-modulated Hebbian**: λ * δ(t) * PFC * MSN
     2. **Direct reward modulation**: α * r * MSN
     3. **Exploration bonus**: β * Φ * MSN

   - Exploration: Φ = sqrt(2 * ln(N) / n) (Equation 4)
   - Learning rates decay over episodes

## Key Parameters (from paper)

```python
# LIF neurons
τ = 1.0          # Time constant
V_reset = 0.0    # Reset potential
V_th = 1.0       # Firing threshold
T = 2000         # Simulation timesteps per decision (updated from 20)

# Plasticity
α = 0.19         # Reward modulation rate
λ_D1 = 0.01      # D1 pathway learning rate (initial)
λ_D2 = 0.01      # D2 pathway learning rate (initial)
β = 0.05         # Exploration bonus (initial)
γ = 0.99         # TD discount factor

# Baseline currents (added for stable spiking dynamics)
baseline_gpe = 1.5   # GPe tonic activity
baseline_stn = 1.8   # STN tonic activity
baseline_gpi = 1.5   # GPi tonic activity
baseline_thal = 1.2  # Thalamus baseline
baseline_pmc = 0.5   # PMC baseline
```

## Testing

All Phase 1 and 2 components have been tested:

```bash
python test_phase1_phase2.py
```

**Test Results:**
- ✓ LIF Neuron Layer
- ✓ Basal Ganglia Actor Network
- ✓ Critic Network
- ✓ Cortico-Striatal Plasticity
- ✓ Component Integration

## Architecture Summary

### Direct Pathway (facilitates movement)
```
PFC [16] → D1-MSNs [4] → GPi [4] → Thalamus [4] → PMC [4]
        (excite)    (inhibit)   (inhibit)    (excite)
```

### Indirect Pathway (inhibits movement)
```
PFC [16] → D2-MSNs [4] → GPe [4] → STN [4] → GPi [4]
        (excite)    (inhibit) (inhibit) (excite)
```

### Decision Flow
1. State → PFC (one-hot encoding, constant spiking for T=20 timesteps)
2. PFC → D1/D2-MSNs (learnable weights)
3. Pathways compete at GPi
4. GPi → Thalamus → PMC
5. Action = argmax(PMC spike counts)

### Learning Flow
1. Critic computes TD error: δ(t) = r + γV(s') - V(s)
2. Plasticity rule updates PFC→Striatum weights (Eq. 3)
3. Weights clamped to [0, 0.4] and renormalized

## Reference

Ao, T., Liu, Q., Fu, L., & Zhou, Y. (2024). A Spiking Neural Network Action Decision Method Inspired by Basal Ganglia. *Procedia Computer Science*, 250, 115-121.

## Implementation Notes

- **No hyperdirect pathway**: Paper states it doesn't function in FrozenLake-v1
- **No PMC lateral inhibition**: Not mentioned in paper
- **Fixed thresholds**: V_th=1.0 (adaptive thresholds not implemented initially)
- **Spike counts**: Use total spike counts over T timesteps (not firing rates)
- **T parameter**: Updated from 20 to 2000 timesteps per decision (from paper Figure 3)
- **Baseline currents**: Added tonic activity to GPe, STN, GPi, Thalamus, and PMC to enable stable spiking dynamics with inhibitory connections
- **Weight initialization**: Cortico-striatal weights initialized with mean ~1.5 to ensure downstream neurons receive sufficient current
- **PFC encoding**: Input current set to 1.5 (above threshold) for constant spiking of active state neuron

## Next Steps (Phase 3)

- Implement training loop with FrozenLake-v1 environment
- Add visualization for firing rates (Figure 3 from paper)
- Compare performance with DQN and PPO
