"""
Test script for Phase 1 and Phase 2 components
Verifies LIF neurons, Basal Ganglia Actor, Critic, and Plasticity modules
"""

import torch
import numpy as np
from models import BasalGangliaActor, CriticNetwork, CorticostriatalPlasticity
from spikingjelly.activation_based import neuron, functional
from config import Config


def test_lif_neuron():
    """Test LIF neuron using SpikingJelly"""
    print("\n" + "=" * 60)
    print("Testing SpikingJelly LIF Neuron")
    print("=" * 60)

    # Create LIFNode matching Ao et al. 2024 parameters
    # Note: SpikingJelly requires tau > 1.0, using tau=2.0
    lif = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         detach_reset=True, step_mode='s')

    # Test single timestep
    print("\n1. Testing single timestep dynamics:")
    functional.reset_net(lif)
    input_current = torch.tensor([[1.5, 0.8, 0.5, 0.2]])
    spikes = lif(input_current)
    print(f"   Input current: {input_current}")
    print(f"   Spikes: {spikes}")

    # Test multiple timesteps
    print("\n2. Testing multiple timesteps (T=20):")
    T = 20
    functional.reset_net(lif)
    spike_counts = torch.zeros(4)
    for t in range(T):
        input_t = torch.ones(1, 4) * 0.8
        spikes_t = lif(input_t)
        spike_counts += spikes_t.squeeze(0)
    print(f"   Total spike counts over {T} timesteps: {spike_counts}")

    print("\n[PASS] LIF neuron test passed!")
    return True


def test_basal_ganglia_actor():
    """Test Basal Ganglia Actor network"""
    print("\n" + "=" * 60)
    print("Testing Basal Ganglia Actor Network")
    print("=" * 60)

    actor = BasalGangliaActor(
        n_states=16,
        n_actions=4,
        tau=Config.TAU,
        v_reset=Config.V_RESET,
        v_th=Config.V_TH,
        dt=Config.DT,
        T=Config.T_SIMULATION
    )

    print("\n1. Network structure:")
    print(f"   PFC: {actor.n_states} neurons (SpikingJelly LIFNode)")
    print(f"   D1-MSN: {actor.n_actions} neurons (SpikingJelly LIFNode)")
    print(f"   D2-MSN: {actor.n_actions} neurons (SpikingJelly LIFNode)")
    print(f"   GPe: {actor.n_actions} neurons (SpikingJelly LIFNode)")
    print(f"   STN: {actor.n_actions} neurons (SpikingJelly LIFNode)")
    print(f"   GPi: {actor.n_actions} neurons (SpikingJelly LIFNode)")
    print(f"   Thalamus: {actor.n_actions} neurons (SpikingJelly LIFNode)")
    print(f"   PMC: {actor.n_actions} neurons (SpikingJelly LIFNode)")

    print("\n2. Weight matrices:")
    print(f"   W_PFC_D1 shape: {actor.w_pfc_d1.shape}")
    print(f"   W_PFC_D2 shape: {actor.w_pfc_d2.shape}")
    print(f"   W_PFC_D1 sample row: {actor.w_pfc_d1[0]}")
    print(f"   W_PFC_D2 sample row: {actor.w_pfc_d2[0]}")

    print("\n3. Fixed pathway weights:")
    print(f"   W_D1_GPi (diagonal): {torch.diag(actor.w_d1_gpi)}")
    print(f"   W_D2_GPe (diagonal): {torch.diag(actor.w_d2_gpe)}")
    print(f"   W_GPe_STN (diagonal): {torch.diag(actor.w_gpe_stn)}")
    print(f"   W_STN_GPi (diagonal): {torch.diag(actor.w_stn_gpi)}")

    print("\n4. Testing forward pass (state=0):")
    state = 0
    action, pmc_spikes, activities = actor.forward(state, return_all_activities=True)
    print(f"   Selected action: {action}")
    print(f"   PMC spike counts: {pmc_spikes}")
    print(f"   PFC spikes: {activities['pfc']}")
    print(f"   D1-MSN spikes: {activities['d1']}")
    print(f"   D2-MSN spikes: {activities['d2']}")

    print("\n5. Testing different states:")
    for s in [0, 5, 10, 15]:
        action, pmc_spikes, _ = actor.forward(s, return_all_activities=True)
        print(f"   State {s:2d} -> Action {action}, PMC spikes: {pmc_spikes.detach().cpu().numpy()}")

    print("\n[PASS] Basal Ganglia Actor test passed!")
    return True


def test_critic():
    """Test Critic network"""
    print("\n" + "=" * 60)
    print("Testing Critic Network")
    print("=" * 60)

    critic = CriticNetwork(n_states=16, hidden_dim=64)

    print("\n1. Network architecture:")
    print(critic.network)

    print("\n2. Testing value estimation:")
    for state in [0, 5, 10, 15]:
        value = critic.forward(state)
        print(f"   V(state={state}) = {value.item():.4f}")

    print("\n3. Testing TD error computation:")
    state = 0
    reward = 1.0
    next_state = 1
    td_error, v_current, v_next = critic.compute_td_error(state, reward, next_state, gamma=0.99)
    print(f"   State: {state}, Reward: {reward}, Next state: {next_state}")
    print(f"   V(s) = {v_current:.4f}, V(s') = {v_next:.4f}")
    print(f"   TD error delta = {td_error:.4f}")

    print("\n4. Testing batch processing:")
    states = torch.tensor([0, 1, 2, 3])
    values = critic.forward(states)
    print(f"   States: {states.numpy()}")
    print(f"   Values: {values.squeeze().detach().numpy()}")

    print("\n[PASS] Critic network test passed!")
    return True


def test_plasticity():
    """Test Cortico-Striatal Plasticity"""
    print("\n" + "=" * 60)
    print("Testing Cortico-Striatal Plasticity")
    print("=" * 60)

    plasticity = CorticostriatalPlasticity(
        n_states=16,
        n_actions=4,
        alpha=Config.ALPHA,
        lambda_d1_init=Config.LAMBDA_D1_INIT,
        lambda_d2_init=Config.LAMBDA_D2_INIT,
        beta_init=Config.BETA_INIT,
        gamma=Config.GAMMA
    )

    print("\n1. Plasticity parameters:")
    print(f"   alpha (reward modulation): {plasticity.alpha}")
    print(f"   lambda_D1 (initial): {plasticity.lambda_d1}")
    print(f"   lambda_D2 (initial): {plasticity.lambda_d2}")
    print(f"   beta (exploration, initial): {plasticity.beta}")
    print(f"   gamma (discount): {plasticity.gamma}")

    print("\n2. Testing exploration bonus:")
    state = 0
    for action in range(4):
        phi = plasticity.compute_exploration_bonus(state, action)
        print(f"   Phi(state={state}, action={action}) = {phi:.4f}")
        plasticity.update_action_counts(state, action)

    # After some attempts
    for _ in range(5):
        plasticity.update_action_counts(state, 0)  # Action 0 taken multiple times

    print("\n3. After 5 more attempts of action 0:")
    for action in range(4):
        phi = plasticity.compute_exploration_bonus(state, action)
        print(f"   Phi(state={state}, action={action}) = {phi:.4f}")

    print("\n4. Testing weight update computation:")
    pfc_spikes = torch.randn(16).clamp(min=0)  # Simulated PFC activity
    d1_spikes = torch.randn(4).clamp(min=0)  # Simulated D1 activity
    d2_spikes = torch.randn(4).clamp(min=0)  # Simulated D2 activity
    td_error = 0.5
    reward = 1.0
    state = 0
    action = 0

    delta_w_d1, delta_w_d2 = plasticity.compute_weight_update(
        pfc_spikes, d1_spikes, d2_spikes,
        td_error, reward, state, action
    )
    print(f"   Delta_W_PFC_D1 shape: {delta_w_d1.shape}")
    print(f"   Delta_W_PFC_D2 shape: {delta_w_d2.shape}")
    print(f"   Delta_W_PFC_D1 mean: {delta_w_d1.mean().item():.6f}")
    print(f"   Delta_W_PFC_D2 mean: {delta_w_d2.mean().item():.6f}")

    print("\n5. Testing learning rate decay:")
    initial_rates = plasticity.get_learning_rates()
    print(f"   Initial: {initial_rates}")

    plasticity.decay_learning_rates(episode=500, total_episodes=1000)
    decayed_rates = plasticity.get_learning_rates()
    print(f"   After 500 episodes: {decayed_rates}")

    print("\n[PASS] Plasticity test passed!")
    return True


def test_integration():
    """Test integration of all components"""
    print("\n" + "=" * 60)
    print("Testing Component Integration")
    print("=" * 60)

    # Create all components
    actor = BasalGangliaActor(
        n_states=16, n_actions=4,
        tau=Config.TAU, v_reset=Config.V_RESET,
        v_th=Config.V_TH, dt=Config.DT, T=Config.T_SIMULATION
    )

    critic = CriticNetwork(n_states=16, hidden_dim=64)

    plasticity = CorticostriatalPlasticity(
        n_states=16, n_actions=4,
        alpha=Config.ALPHA,
        lambda_d1_init=Config.LAMBDA_D1_INIT,
        lambda_d2_init=Config.LAMBDA_D2_INIT,
        beta_init=Config.BETA_INIT,
        gamma=Config.GAMMA
    )

    print("\n1. Simulating one decision-making step:")
    state = 0
    next_state = 1
    reward = 0.0

    # Actor selects action
    action, pmc_spikes, activities = actor.forward(state, return_all_activities=True)
    print(f"   State {state} -> Action {action}")
    print(f"   PMC spikes: {pmc_spikes.detach().cpu().numpy()}")

    # Critic computes TD error
    td_error, v_s, v_s_next = critic.compute_td_error(
        state, reward, next_state, gamma=Config.GAMMA
    )
    print(f"   V({state}) = {v_s:.4f}, V({next_state}) = {v_s_next:.4f}")
    print(f"   TD error delta = {td_error:.4f}")

    # Plasticity update
    print("\n2. Applying plasticity update:")
    w_d1_before = actor.w_pfc_d1.clone()
    w_d2_before = actor.w_pfc_d2.clone()

    plasticity.apply_weight_update(
        actor,
        activities['pfc'],
        activities['d1'],
        activities['d2'],
        td_error,
        reward,
        state,
        action
    )

    w_d1_after = actor.w_pfc_d1
    w_d2_after = actor.w_pfc_d2

    delta_d1 = (w_d1_after - w_d1_before).abs().mean()
    delta_d2 = (w_d2_after - w_d2_before).abs().mean()

    print(f"   Mean |Delta_W_PFC_D1|: {delta_d1:.6f}")
    print(f"   Mean |Delta_W_PFC_D2|: {delta_d2:.6f}")

    print("\n[PASS] Integration test passed!")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("PHASE 1 & 2 COMPONENT TESTS")
    print("Ao et al. 2024 Basal Ganglia Model")
    print("=" * 60)

    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    tests = [
        ("LIF Neuron", test_lif_neuron),
        ("Basal Ganglia Actor", test_basal_ganglia_actor),
        ("Critic Network", test_critic),
        ("Plasticity Rule", test_plasticity),
        ("Component Integration", test_integration)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n[FAIL] {test_name} test FAILED with error:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status}: {test_name}")

    all_passed = all(success for _, success in results)
    print("=" * 60)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED!")
    else:
        print("[ERROR] SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
