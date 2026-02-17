"""
Comprehensive Test Suite for Unified Intelligence Framework.
Tests all 6 frameworks and core components.
"""

import sys
from typing import Dict, Any, List, Tuple, Optional
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("UNIFIED INTELLIGENCE FRAMEWORK - COMPREHENSIVE TESTS")
print("=" * 60)


def test_core_types():
    print("\n[1/12] Testing Core Types...")
    from fishstick.core.types import (
        MetricTensor,
        SymplecticForm,
        PhaseSpaceState,
        ConservationLaw,
        VerificationCertificate,
    )

    metric = MetricTensor(torch.eye(3))
    assert metric.inverse().data.shape == (3, 3), "Metric inverse failed"

    symplectic = SymplecticForm(dim=2)
    assert symplectic.matrix.shape == (4, 4), "Symplectic form shape wrong"

    state = PhaseSpaceState(q=torch.randn(2), p=torch.randn(2))
    assert state.stack().shape == (4,), "Phase space stack failed"

    print("  ✓ MetricTensor, SymplecticForm, PhaseSpaceState")
    return True


def test_statistical_manifold():
    print("\n[2/12] Testing Statistical Manifold...")
    from fishstick.core.manifold import StatisticalManifold

    manifold = StatisticalManifold(dim=10)

    def log_prob(params):
        return (params**2).sum()

    params = torch.randn(10, requires_grad=True)
    metric = manifold.fisher_information(params, log_prob, n_samples=10)
    assert metric.data.shape == (10, 10), "Fisher metric shape wrong"

    print("  ✓ StatisticalManifold, Fisher Information")
    return True


def test_categorical_structures():
    print("\n[3/12] Testing Categorical Structures...")
    from fishstick.categorical.category import (
        MonoidalCategory,
        Functor,
        NaturalTransformation,
        TracedMonoidalCategory,
        DaggerCategory,
    )
    from fishstick.categorical.category import Object

    cat = MonoidalCategory("TestCat")
    obj1 = Object(name="A", shape=(10,))
    obj2 = Object(name="B", shape=(20,))

    cat.add_object(obj1)
    cat.add_object(obj2)
    assert len(cat.objects()) == 2, "Objects not added"

    tensor_obj = cat.tensor_product(obj1, obj2)
    assert tensor_obj.name == "A⊗B", "Tensor product naming wrong"

    traced = TracedMonoidalCategory("TracedCat")
    unit = traced.unit_object()
    assert unit.name == "I", "Unit object wrong"

    dagger = DaggerCategory("DaggerCat")
    f = lambda x: x
    adjoint = dagger.dagger(f)
    assert callable(adjoint), "Dagger not callable"

    print("  ✓ MonoidalCategory, TracedMonoidalCategory, DaggerCategory")
    return True


def test_lens():
    print("\n[4/12] Testing Lens-based Learning...")
    from fishstick.categorical.lens import Lens, BidirectionalLearner

    lens = Lens(get=lambda x: x * 2, put=lambda s, a: s + a)

    result = lens(5)
    assert result == 10, f"Lens get failed: {result}"

    composed = lens.compose(Lens(get=lambda x: x + 1, put=lambda s, a: s))
    assert composed(5) == 11, "Lens composition failed"

    print("  ✓ Lens, Lens composition")
    return True


def test_fisher_natural_gradient():
    print("\n[5/12] Testing Fisher Information & Natural Gradient...")
    from fishstick.geometric.fisher import (
        FisherInformationMetric,
        NaturalGradient,
        NaturalGradientOptimizer,
    )

    fisher = FisherInformationMetric(damping=1e-4)

    model = torch.nn.Linear(10, 5)
    data = torch.randn(32, 10)

    G = fisher.monte_carlo_estimate(model, data, n_samples=10)
    assert G.data.shape[0] > 0, "Fisher estimation failed"

    print("  ✓ FisherInformationMetric, Monte Carlo estimation")
    return True


def test_sheaf():
    print("\n[6/12] Testing Sheaf Theory...")
    from fishstick.geometric.sheaf import (
        DataSheaf,
        SheafCohomology,
        SheafLayer,
    )

    open_cover = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    sheaf = DataSheaf(open_cover, stalk_dim=8)

    for i in range(3):
        sheaf.set_section(i, torch.randn(8))

    betti_1, obstruction = sheaf.compute_cohomology()
    print(f"  Sheaf cohomology H^1: betti_1 = {betti_1}")

    loss = sheaf.consistency_loss()
    assert loss.item() >= 0, "Consistency loss should be non-negative"

    layer = SheafLayer(n_patches=3, feature_dim=8)
    features = {i: torch.randn(8) for i in range(3)}
    updated, cohom_loss = layer(features)
    assert len(updated) == 3, "SheafLayer output wrong"

    print("  ✓ DataSheaf, SheafCohomology, SheafLayer")
    return True


def test_hamiltonian_nn():
    print("\n[7/12] Testing Hamiltonian Neural Networks...")
    from fishstick.dynamics.hamiltonian import (
        HamiltonianNeuralNetwork,
        SymplecticIntegrator,
        HamiltonianLayer,
        NoetherConservation,
    )
    from fishstick.core.types import PhaseSpaceState

    hnn = HamiltonianNeuralNetwork(input_dim=4, hidden_dim=64)

    z0 = torch.randn(2, 8)
    dz = hnn(z0)
    assert dz.shape == z0.shape, f"HNN output shape wrong: {dz.shape}"

    traj = hnn.integrate(z0, n_steps=10, dt=0.1, method="leapfrog")
    assert traj.shape[0] == 11, "Trajectory length wrong"

    H_before = hnn.hamiltonian(z0)
    H_after = hnn.hamiltonian(traj[-1])
    energy_change = torch.abs(H_before - H_after).mean().item()
    print(f"  Energy change over 10 steps: {energy_change:.6f}")

    layer = HamiltonianLayer(dim=4, dt=0.1)
    z_out = layer(z0)
    assert z_out.shape == z0.shape, "HamiltonianLayer shape wrong"

    print("  ✓ HNN, Symplectic Integration, HamiltonianLayer")
    return True


def test_thermodynamic():
    print("\n[8/12] Testing Thermodynamic Gradient Flow...")
    from fishstick.dynamics.thermodynamic import (
        ThermodynamicGradientFlow,
        FreeEnergy,
        WassersteinGradientFlow,
        LandauerBound,
    )

    params = [torch.randn(10, requires_grad=True) for _ in range(3)]
    tgf = ThermodynamicGradientFlow(params, lr=0.01, beta=1.0)

    loss_fn = lambda: sum(p.sum() for p in params)
    loss, work = tgf.step(loss_fn)
    assert isinstance(work, float), "Work should be float"

    efficiency = tgf.thermodynamic_efficiency()
    print(f"  Thermodynamic efficiency: {efficiency:.4f}")

    landauer = LandauerBound(temperature=300.0)
    min_energy = landauer.minimum_energy(bits_erased=1.0)
    print(f"  Landauer minimum energy: {min_energy:.2e} J")

    wgf = WassersteinGradientFlow(dim=10, n_particles=50)
    potential = lambda x: (x**2).sum(dim=-1)
    step_result = wgf.gradient_flow_step(potential)

    print("  ✓ ThermodynamicGradientFlow, LandauerBound, WassersteinGradientFlow")
    return True


def test_sheaf_attention():
    print("\n[9/12] Testing Sheaf-Optimized Attention...")
    from fishstick.sheaf.attention import (
        SheafOptimizedAttention,
        SheafTransformerLayer,
        SheafTransformer,
    )

    attn = SheafOptimizedAttention(embed_dim=64, num_heads=4)
    x = torch.randn(2, 10, 64)

    output, weights = attn(x, need_weights=True)
    assert output.shape == x.shape, f"Attention output shape wrong: {output.shape}"

    open_cover = [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
    output2, _ = attn(x, open_cover=open_cover)
    assert output2.shape == x.shape, "Attention with cover failed"

    layer = SheafTransformerLayer(embed_dim=64, num_heads=4)
    out = layer(x)
    assert out.shape == x.shape, "Transformer layer failed"

    transformer = SheafTransformer(embed_dim=64, num_heads=4, num_layers=2)
    out = transformer(x)
    assert out.shape == x.shape, "Full transformer failed"

    print("  ✓ SheafOptimizedAttention, SheafTransformer")
    return True


def test_rg_autoencoder():
    print("\n[10/12] Testing RG-Aware Autoencoder...")
    from fishstick.rg.autoencoder import (
        RGAutoencoder,
        RGFlow,
        RGAugmentedResNet,
    )

    ae = RGAutoencoder(
        input_dim=784, latent_dims=[256, 128, 64], hidden_dim=256, n_scales=3
    )

    x = torch.randn(4, 784)
    outputs = ae(x)
    assert outputs["reconstruction"].shape == x.shape, "AE reconstruction shape wrong"
    assert len(outputs["latents"]) == 3, "Wrong number of latent scales"

    loss, losses = ae.loss(x, outputs)
    assert loss.item() >= 0, "AE loss should be non-negative"
    print(f"  AE loss: {loss.item():.4f}")

    rg_flow = RGFlow(n_scales=4)
    features = torch.randn(4, 16, 8, 8)
    coarse = rg_flow.coarse_grain(features, method="avg_pool")
    assert coarse.shape[2:] == (4, 4), "Coarse-graining failed"

    resnet = RGAugmentedResNet(in_channels=1, base_channels=16, n_scales=2)
    img = torch.randn(2, 1, 16, 16)
    output, scale_reprs = resnet(img)
    assert output.shape[0] == 2, "ResNet output batch wrong"
    assert len(scale_reprs) == 2, "Wrong number of scale representations"

    print("  ✓ RGAutoencoder, RGFlow, RGAugmentedResNet")
    return True


def test_verification():
    print("\n[11/12] Testing Verification Pipeline...")
    from fishstick.verification.types import (
        DependentlyTypedLearner,
        VerificationPipeline,
        LeanInterface,
        SMTVerifier,
    )
    from fishstick.verification.types import TypeSpec, SafetyProperty

    layers = [torch.nn.Linear(784, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10)]

    model = DependentlyTypedLearner(
        layers=layers,
        input_spec=TypeSpec(name="Input", constraints=["dim=784"]),
        output_spec=TypeSpec(name="Output", constraints=["dim=10"]),
    )

    x = torch.randn(1, 784)
    y = model(x)
    assert y.shape == (1, 10), "Model forward failed"

    lip = model.lipschitz_constant
    print(f"  Lipschitz constant: {lip:.4f}")

    cert = model.verify_robustness(x, epsilon=0.1)
    print(f"  Robustness verified: {cert.is_verified}")

    pipeline = VerificationPipeline(model)
    pipeline.add_property(
        SafetyProperty(name="robustness", property_type="robustness", bound=0.1)
    )

    lean = LeanInterface()
    lean_code = lean.generate_robustness_theorem("test_model", 0.1, lip)
    assert "theorem" in lean_code.lower(), "Lean generation failed"

    smt = SMTVerifier()
    verified, counterexample = smt.verify_robustness_smt(model, x, 0.1)
    print(f"  SMT verification: {verified}")

    print(
        "  ✓ DependentlyTypedLearner, VerificationPipeline, LeanInterface, SMTVerifier"
    )
    return True


def test_all_frameworks():
    print("\n[12/12] Testing All 6 Frameworks...")

    from fishstick.frameworks.uniintelli import create_uniintelli
    from fishstick.frameworks.hsca import create_hsca
    from fishstick.frameworks.uia import create_uia
    from fishstick.frameworks.scif import create_scif
    from fishstick.frameworks.uif import create_uif
    from fishstick.frameworks.uis import create_uis

    frameworks = {
        "UniIntelli": create_uniintelli,
        "HSCA": create_hsca,
        "UIA": create_uia,
        "SCIF": create_scif,
        "UIF": create_uif,
        "UIS": create_uis,
    }

    x = torch.randn(4, 784)

    for name, create_fn in frameworks.items():
        model = create_fn(input_dim=784, output_dim=10)

        n_params = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 10), f"{name} output shape wrong: {output.shape}"
        print(f"  ✓ {name}: {n_params:,} parameters, output shape {output.shape}")

    return True


def test_training_step():
    print("\n[BONUS] Testing Training Step...")

    from fishstick.frameworks.uniintelli import create_uniintelli

    model = create_uniintelli(input_dim=784, output_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.randn(32, 784), torch.randint(0, 10, (32,)))
    dataloader = DataLoader(dataset, batch_size=8)

    model.train()
    total_loss = 0.0

    for batch_idx, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx >= 2:
            break

    avg_loss = total_loss / 3
    print(f"  Training loss: {avg_loss:.4f}")
    print("  ✓ Training step completed")

    return True


def main():
    results = []

    tests = [
        ("Core Types", test_core_types),
        ("Statistical Manifold", test_statistical_manifold),
        ("Categorical Structures", test_categorical_structures),
        ("Lens-based Learning", test_lens),
        ("Fisher & Natural Gradient", test_fisher_natural_gradient),
        ("Sheaf Theory", test_sheaf),
        ("Hamiltonian Neural Networks", test_hamiltonian_nn),
        ("Thermodynamic Gradient Flow", test_thermodynamic),
        ("Sheaf-Optimized Attention", test_sheaf_attention),
        ("RG-Aware Autoencoder", test_rg_autoencoder),
        ("Verification Pipeline", test_verification),
        ("All 6 Frameworks", test_all_frameworks),
        ("Training Step", test_training_step),
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

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error[:50]}...")

    print("-" * 60)
    print(f"Total: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
