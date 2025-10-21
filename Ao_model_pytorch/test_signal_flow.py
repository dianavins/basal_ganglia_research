"""
Detailed signal flow diagnostic to understand current propagation
"""

import torch
import numpy as np
from models import BasalGangliaActor
from config import Config

print("=" * 60)
print("Signal Flow Diagnostic")
print("=" * 60)

# Create actor
actor = BasalGangliaActor(
    n_states=16,
    n_actions=4,
    T=20  # Use shorter T for debugging
)

state = 0
print(f"\nTesting with state={state}, T=20 timesteps")

# Reset and get PFC input
actor.reset_states()
pfc_input = actor.encode_state(state, T=20)

print(f"\n1. PFC Input:")
print(f"   PFC input for neuron {state}: {pfc_input[0, state]}")
print(f"   PFC input shape: {pfc_input.shape}")

# Manually run first few timesteps to see currents
print(f"\n2. First 3 timesteps signal propagation:")

for t in range(3):
    print(f"\n   Timestep {t}:")

    # PFC
    pfc_spikes = actor.pfc(pfc_input[t])
    print(f"   PFC spikes: {pfc_spikes}")

    # Cortico-striatal currents
    d1_current = torch.matmul(pfc_spikes, actor.w_pfc_d1)
    d2_current = torch.matmul(pfc_spikes, actor.w_pfc_d2)
    print(f"   D1 current: {d1_current}")
    print(f"   D2 current: {d2_current}")

    # Striatal spikes
    d1_spikes = actor.d1_msn(d1_current)
    d2_spikes = actor.d2_msn(d2_current)
    print(f"   D1 spikes: {d1_spikes}")
    print(f"   D2 spikes: {d2_spikes}")

    # GPi currents from both pathways
    gpi_current_direct = torch.matmul(d1_spikes, actor.w_d1_gpi)
    print(f"   GPi current (direct): {gpi_current_direct}")

    # Indirect pathway
    gpe_current = torch.matmul(d2_spikes, actor.w_d2_gpe)
    gpe_spikes = actor.gpe(gpe_current)

    stn_current = torch.matmul(gpe_spikes, actor.w_gpe_stn)
    stn_spikes = actor.stn(stn_current)

    gpi_current_indirect = torch.matmul(stn_spikes, actor.w_stn_gpi)
    print(f"   GPi current (indirect): {gpi_current_indirect}")

    gpi_current_total = gpi_current_direct + gpi_current_indirect
    print(f"   GPi current (total): {gpi_current_total}")

    gpi_spikes = actor.gpi(gpi_current_total)
    print(f"   GPi spikes: {gpi_spikes}")

    # Thalamus
    thal_current = torch.matmul(gpi_spikes, actor.w_gpi_thal)
    print(f"   Thalamus current: {thal_current}")
    thal_spikes = actor.thalamus(thal_current)
    print(f"   Thalamus spikes: {thal_spikes}")

    # PMC
    pmc_current = torch.matmul(thal_spikes, actor.w_thal_pmc)
    print(f"   PMC current: {pmc_current}")
    pmc_spikes = actor.pmc(pmc_current)
    print(f"   PMC spikes: {pmc_spikes}")

print(f"\n3. Weight analysis:")
print(f"   W_PFC_D1[{state}] (cortico-striatal weights for state {state}):")
print(f"   {actor.w_pfc_d1[state]}")
print(f"   Mean cortico-striatal weight: {actor.w_pfc_d1.mean():.4f}")
print(f"   Max cortico-striatal weight: {actor.w_pfc_d1.max():.4f}")

print(f"\n4. Issue identified:")
print(f"   PFC spike output: 1.0")
print(f"   Typical W_PFC_D1 value: ~{actor.w_pfc_d1.mean():.4f}")
print(f"   Resulting D1 current: ~{actor.w_pfc_d1.mean():.4f}")
print(f"   LIF threshold: 1.0")
print(f"   Problem: Current ({actor.w_pfc_d1.mean():.4f}) << Threshold (1.0)")
print(f"   Solution: Need to either:")
print(f"     a) Increase cortico-striatal weights significantly")
print(f"     b) Use multiple PFC neurons projecting to each MSN")
print(f"     c) Add baseline/tonic current to downstream neurons")

print("\n" + "=" * 60)
