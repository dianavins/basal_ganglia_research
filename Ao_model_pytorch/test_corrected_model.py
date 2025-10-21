"""
Quick diagnostic test for corrected model with T=2000
"""

import torch
import numpy as np
from models import BasalGangliaActor
from config import Config

print("=" * 60)
print("Testing Corrected Basal Ganglia Model")
print("=" * 60)

# Create actor with T=2000
actor = BasalGangliaActor(
    n_states=16,
    n_actions=4,
    T=Config.T_SIMULATION  # Should be 2000
)

print(f"\n1. Configuration:")
print(f"   T_SIMULATION = {Config.T_SIMULATION}")
print(f"   Actor T = {actor.T}")

print(f"\n2. Weight initialization:")
print(f"   W_PFC_D1 sample: {actor.w_pfc_d1[0]}")
print(f"   W_PFC_D2 sample: {actor.w_pfc_d2[0]}")
print(f"   W_PFC_D1 stats: min={actor.w_pfc_d1.min():.4f}, max={actor.w_pfc_d1.max():.4f}, mean={actor.w_pfc_d1.mean():.4f}")

print(f"\n3. Fixed pathway weights (diagonal values):")
print(f"   W_D1_GPi:  {torch.diag(actor.w_d1_gpi)}")
print(f"   W_D2_GPe:  {torch.diag(actor.w_d2_gpe)}")
print(f"   W_GPe_STN: {torch.diag(actor.w_gpe_stn)}")
print(f"   W_STN_GPi: {torch.diag(actor.w_stn_gpi)}")
print(f"   W_GPi_Thal: {torch.diag(actor.w_gpi_thal)}")
print(f"   W_Thal_PMC: {torch.diag(actor.w_thal_pmc)}")

print(f"\n4. Testing forward pass with state=0:")
state = 0
action, pmc_spikes, activities = actor.forward(state, return_all_activities=True)

print(f"   Selected action: {action}")
print(f"   PMC spike counts: {pmc_spikes}")
print(f"   PFC total spikes: {activities['pfc'].sum():.0f}")
print(f"   D1-MSN total spikes: {activities['d1'].sum():.0f}")
print(f"   D2-MSN total spikes: {activities['d2'].sum():.0f}")
print(f"   GPi total spikes: {activities['gpi'].sum():.0f}")
print(f"   Thalamus total spikes: {activities['thalamus'].sum():.0f}")

print(f"\n5. Neuron activity breakdown:")
print(f"   PFC neurons firing: {(activities['pfc'] > 0).sum()}/16")
print(f"   D1-MSN neurons firing: {(activities['d1'] > 0).sum()}/4")
print(f"   D2-MSN neurons firing: {(activities['d2'] > 0).sum()}/4")
print(f"   PMC neurons firing: {(pmc_spikes > 0).sum()}/4")

print(f"\n6. Testing with increased input strength:")
# Manually increase PFC weights to test spiking
with torch.no_grad():
    actor.w_pfc_d1.data *= 10.0  # Temporarily increase
    actor.w_pfc_d2.data *= 10.0

action2, pmc_spikes2, activities2 = actor.forward(state, return_all_activities=True)
print(f"   With 10x weights:")
print(f"   PMC spike counts: {pmc_spikes2}")
print(f"   D1-MSN total spikes: {activities2['d1'].sum():.0f}")
print(f"   PMC total spikes: {pmc_spikes2.sum():.0f}")

print("\n" + "=" * 60)
print("Diagnosis Complete")
print("=" * 60)

# Check if model can generate spikes
if pmc_spikes.sum() == 0:
    print("\nWARNING: No PMC spikes detected!")
    print("Possible causes:")
    print("  1. Initial weights too small (need to accumulate current)")
    print("  2. Threshold too high for weak input currents")
    print("  3. Inhibitory pathways suppressing all activity")
    print("\nSuggestions:")
    print("  - Increase initial weight magnitude")
    print("  - Add baseline current/bias to neurons")
    print("  - Adjust threshold or add noise to membrane potential")
else:
    print(f"\n✓ Model is generating spikes ({pmc_spikes.sum():.0f} total PMC spikes)")
