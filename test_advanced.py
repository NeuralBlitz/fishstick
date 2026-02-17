"""
Comprehensive tests for advanced features.
"""

import sys
import torch
import numpy as np
from torch import nn

print("=" * 70)
print("ADVANCED FEATURES - COMPREHENSIVE TESTS")
print("=" * 70)


def test_neural_ode():
    """Test Neural ODE components."""
    print("\n[1/6] Testing Neural ODE...")

    from fishstick.neural_ode import (
        ODEFunction,
        NeuralODE,
        AugmentedNeuralODE,
        LatentODE,
    )

    # Test ODE Function
    odefunc = ODEFunction(dim=10, hidden_dim=64)
    x = torch.randn(4, 10)
    t = torch.tensor(0.5)
    dxdt = odefunc(t, x)
    assert dxdt.shape == (4, 10), f"ODE function output shape wrong: {dxdt.shape}"

    # Test Neural ODE
    node = NeuralODE(odefunc, t_span=(0.0, 1.0), method="euler")
    z0 = torch.randn(4, 10)
    z1 = node(z0)
    assert z1.shape == (4, 10), f"Neural ODE output shape wrong: {z1.shape}"

    # Test Augmented Neural ODE
    aug_node = AugmentedNeuralODE(dim=10, augment_dim=5)
    x = torch.randn(4, 10)
    out = aug_node(x)
    assert out.shape == (4, 10), f"Augmented NODE output shape wrong: {out.shape}"

    print("  ✓ ODEFunction, NeuralODE, AugmentedNeuralODE")
    return True


def test_graph_networks():
    """Test geometric graph neural networks."""
    print("\n[2/6] Testing Geometric Graph Networks...")

    from fishstick.graph import EquivariantMessagePassing, SheafGraphConv

    # Test Equivariant Message Passing
    n_nodes = 10
    n_edges = 30

    x = torch.randn(n_nodes, 16)
    pos = torch.randn(n_nodes, 3)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))

    layer = EquivariantMessagePassing(node_dim=16, edge_dim=0, hidden_dim=32)
    x_out, pos_out = layer(x, pos, edge_index)

    assert x_out.shape == (n_nodes, 16), f"Output features shape wrong"
    assert pos_out.shape == (n_nodes, 3), f"Output positions shape wrong"

    # Test Sheaf Graph Convolution
    sheaf_conv = SheafGraphConv(in_channels=16, out_channels=32, stalk_dim=8)
    out = sheaf_conv(x, edge_index)
    assert out.shape == (n_nodes, 32), f"Sheaf conv output shape wrong"

    print("  ✓ EquivariantMessagePassing, SheafGraphConv")
    return True


def test_probabilistic():
    """Test probabilistic/Bayesian components."""
    print("\n[3/6] Testing Probabilistic Layers...")

    from fishstick.probabilistic import (
        BayesianLinear,
        MCDropout,
        VariationalLayer,
        DeepEnsemble,
    )

    # Test Bayesian Linear
    bayesian_layer = BayesianLinear(20, 10, prior_sigma=1.0)
    x = torch.randn(4, 20)
    out = bayesian_layer(x, sample=True)
    assert out.shape == (4, 10), f"Bayesian linear output shape wrong"

    kl = bayesian_layer.kl_divergence()
    assert kl.item() >= 0, f"KL divergence should be non-negative"

    # Test MC Dropout
    mc_dropout = MCDropout(p=0.5)
    x = torch.randn(4, 20)
    out_train = mc_dropout(x)
    mc_dropout.eval()
    out_eval = mc_dropout(x)  # Still applies dropout
    assert not torch.allclose(out_train, out_eval), "MC Dropout should be stochastic"

    # Test Variational Layer
    var_layer = VariationalLayer(20, 10)
    x = torch.randn(4, 20)
    mean, log_var = var_layer(x)
    assert mean.shape == (4, 10), f"Variational mean shape wrong"
    assert log_var.shape == (4, 10), f"Variational log_var shape wrong"

    print("  ✓ BayesianLinear, MCDropout, VariationalLayer")
    return True


def test_flows():
    """Test normalizing flows."""
    print("\n[4/6] Testing Normalizing Flows...")

    from fishstick.flows import RealNVP, Glow, MAF

    # Test RealNVP
    realnvp = RealNVP(dim=8, n_coupling=4, hidden_dim=32)
    x = torch.randn(4, 8)

    # Forward (density estimation)
    z, logdet = realnvp.forward(x)
    assert z.shape == (4, 8), f"RealNVP output shape wrong"
    assert logdet.shape == (4,), f"RealNVP logdet shape wrong"

    # Inverse (sampling)
    x_recon, logdet_inv = realnvp.inverse(z)
    assert x_recon.shape == (4, 8), f"RealNVP inverse shape wrong"

    # Test log probability
    log_prob = realnvp.log_prob(x)
    assert log_prob.shape == (4,), f"RealNVP log_prob shape wrong"

    # Test Glow
    glow = Glow(dim=8, n_levels=2, n_steps_per_level=2)
    z, logdet = glow.forward(x)
    assert z.shape == (4, 8), f"Glow output shape wrong"

    # Test MAF
    maf = MAF(dim=8, n_mades=3, hidden_dim=32)
    z, logdet = maf.forward(x)
    assert z.shape == (4, 8), f"MAF output shape wrong"

    print("  ✓ RealNVP, Glow, MAF")
    return True


def test_equivariant():
    """Test equivariant networks."""
    print("\n[5/6] Testing Equivariant Networks...")

    from fishstick.equivariant import (
        SE3EquivariantLayer,
        SE3Transformer,
        E3Conv,
    )

    n_nodes = 10
    n_edges = 30

    features = torch.randn(n_nodes, 32)
    coords = torch.randn(n_nodes, 3)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))

    # Test SE(3) Equivariant Layer
    layer = SE3EquivariantLayer(in_features=32, out_features=32, hidden_dim=32)
    f_out, c_out = layer(features, coords, edge_index)

    assert f_out.shape == (n_nodes, 32), f"SE3 layer features shape wrong"
    assert c_out.shape == (n_nodes, 3), f"SE3 layer coords shape wrong"

    # Test SE(3) Transformer
    transformer = SE3Transformer(feature_dim=32, num_heads=4, hidden_dim=32)
    f_out = transformer(features, coords, edge_index)
    assert f_out.shape == (n_nodes, 32), f"SE3 transformer output shape wrong"

    # Test E(3) Convolution
    conv = E3Conv(in_channels=32, out_channels=64, hidden_channels=32)
    out = conv(features, coords, edge_index)
    assert out.shape == (n_nodes, 64), f"E3 conv output shape wrong"

    print("  ✓ SE3EquivariantLayer, SE3Transformer, E3Conv")
    return True


def test_causal():
    """Test causal inference components."""
    print("\n[6/6] Testing Causal Inference...")

    from fishstick.causal import (
        CausalGraph,
        StructuralCausalModel,
        CausalDiscovery,
    )

    # Test Causal Graph
    adjacency = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    graph = CausalGraph(n_nodes=3, adjacency=adjacency, node_names=["X", "Y", "Z"])

    assert graph.is_dag(), "Graph should be a DAG"
    assert graph.parents(1) == [0], f"Wrong parents: {graph.parents(1)}"
    assert graph.children(0) == [1], f"Wrong children: {graph.children(0)}"

    # Test Structural Causal Model
    scm = StructuralCausalModel(graph, hidden_dim=16)

    # Observational sampling
    sample = scm.forward()
    assert sample.shape == (1, 3), f"SCM sample shape wrong: {sample.shape}"

    # Interventional sampling
    intervention = {0: torch.tensor([[2.0]])}
    sample_do = scm.do_calculus(intervention_node=0, value=torch.tensor([[2.0]]))
    assert sample_do.shape == (1, 3), f"SCM interventional sample shape wrong"

    # Test Causal Discovery
    # Generate synthetic data
    n_samples = 100
    X = np.random.randn(n_samples)
    Y = 2 * X + np.random.randn(n_samples) * 0.1
    Z = 3 * Y + np.random.randn(n_samples) * 0.1
    data = np.column_stack([X, Y, Z])

    # Run PC algorithm
    adj_learned = CausalDiscovery.pc_algorithm(data, alpha=0.5)
    assert adj_learned.shape == (3, 3), f"Learned adjacency shape wrong"

    print("  ✓ CausalGraph, StructuralCausalModel, CausalDiscovery")
    return True


def main():
    """Run all tests."""
    results = []

    tests = [
        ("Neural ODE", test_neural_ode),
        ("Graph Networks", test_graph_networks),
        ("Probabilistic", test_probabilistic),
        ("Flows", test_flows),
        ("Equivariant", test_equivariant),
        ("Causal", test_causal),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
            results.append((name, True, None))
        except Exception as e:
            failed += 1
            results.append((name, False, str(e)))
            print(f"  ✗ FAILED: {e}")

    print("\n" + "=" * 70)
    print("ADVANCED FEATURES - TEST RESULTS SUMMARY")
    print("=" * 70)

    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error[:60]}...")

    print("-" * 70)
    print(f"Total: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
