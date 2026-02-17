#!/usr/bin/env python3
"""
Comprehensive Demo of New Advanced fishstick Tools

This script demonstrates all the new advanced tools including:
- Advanced Data Augmentation
- Profiling & Debugging
- Experiment Tracking
- Distributed Training Utilities
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Import fishstick modules
from fishstick.training import (
    Trainer,
    EarlyStopping,
    ModelCheckpoint,
    GradientAccumulator,
    ExponentialMovingAverage,
    MixedPrecisionTrainer,
)
from fishstick.augmentation import (
    CutOut,
    MixUp,
    CutMix,
    RandAugment,
    get_augmentation_pipeline,
)
from fishstick.profiling import (
    ModelProfiler,
    GradientChecker,
    DeadNeuronDetector,
    WeightAnalyzer,
    TrainingDebugger,
    profile_memory_usage,
)
from fishstick.experiments import Experiment, ExperimentTracker


def create_sample_model(input_dim=784, hidden_dim=128, num_classes=10):
    """Create a simple model for demonstration."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, num_classes),
    )


def create_sample_data(batch_size=32, num_samples=1000):
    """Create sample dataset."""
    X = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, 10, (num_samples,))

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def demo_1_advanced_training():
    """Demo 1: Advanced Training with Distributed Utilities."""
    print("\n" + "=" * 70)
    print("DEMO 1: Advanced Training Utilities")
    print("=" * 70)

    model = create_sample_model()
    train_loader, _ = create_sample_data()

    print("\n1. Gradient Accumulation:")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accumulator = GradientAccumulator(model, optimizer, accumulation_steps=4)

    model.train()
    for i, (data, target) in enumerate(train_loader):
        if i >= 3:
            break
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        accumulator.step(loss)
        print(f"   Batch {i + 1}: Loss = {loss.item():.4f}")

    print("\n2. Exponential Moving Average (EMA):")
    ema = ExponentialMovingAverage(model, decay=0.999)

    # Simulate training and update EMA
    for i in range(3):
        # Fake training step
        for param in model.parameters():
            param.data += torch.randn_like(param.data) * 0.01
        ema.update()

    # Apply EMA for evaluation
    ema.apply_shadow()
    print("   EMA weights applied for evaluation")
    ema.restore()
    print("   Original weights restored")

    print("\n✓ Advanced training utilities demo completed!\n")


def demo_2_augmentation():
    """Demo 2: Advanced Data Augmentation."""
    print("\n" + "=" * 70)
    print("DEMO 2: Advanced Data Augmentation")
    print("=" * 70)

    # Create sample image
    image = torch.rand(3, 32, 32)

    print("\n1. CutOut:")
    cutout = CutOut(n_holes=1, length=8)
    augmented = cutout(image)
    print(f"   Input shape: {image.shape}, Augmented shape: {augmented.shape}")
    print(f"   Cutout removes {((image - augmented).abs() > 1e-6).sum().item()} pixels")

    print("\n2. MixUp:")
    batch = torch.rand(4, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3])
    mixup = MixUp(alpha=0.2)
    mixed, targets_a, targets_b, lam = mixup(batch, targets)
    print(f"   Lambda: {lam:.3f}, Mixed batch shape: {mixed.shape}")
    print(f"   Targets A: {targets_a}, Targets B: {targets_b}")

    print("\n3. CutMix:")
    cutmix = CutMix(alpha=1.0)
    mixed, targets_a, targets_b, lam = cutmix(batch.clone(), targets)
    print(f"   Lambda: {lam:.3f}, Mixed batch shape: {mixed.shape}")

    print("\n4. RandAugment:")
    rand_aug = RandAugment(n=2, m=9)
    augmented = rand_aug(image)
    print(f"   Augmented shape: {augmented.shape}")

    print("\n5. Recommended Pipeline (CIFAR-10, medium severity):")
    pipeline = get_augmentation_pipeline(dataset="cifar10", severity="medium")
    augmented = pipeline(image)
    print(f"   Augmented shape: {augmented.shape}")

    print("\n✓ Augmentation demo completed!\n")


def demo_3_profiling():
    """Demo 3: Profiling and Debugging."""
    print("\n" + "=" * 70)
    print("DEMO 3: Profiling and Debugging")
    print("=" * 70)

    model = create_sample_model()
    train_loader, _ = create_sample_data()

    print("\n1. Model Profiling:")
    profiler = ModelProfiler(model, device="cpu")
    results = profiler.profile(input_shape=(1, 1, 28, 28), num_runs=20, warmup=5)

    print(
        f"   Inference time: {results['avg_inference_time_ms']:.3f} ± {results['std_inference_time_ms']:.3f} ms"
    )
    print(f"   Total parameters: {results['total_params']:,}")
    print(f"   Trainable parameters: {results['trainable_params']:,}")
    print(f"   Model size: {results['model_size_mb']:.2f} MB")

    print("\n2. Weight Analysis:")
    analyzer = WeightAnalyzer(model)
    analyzer.analyze()
    print(f"   Analyzed {len(analyzer.stats)} parameter tensors")

    # Print first few layers
    for i, (name, stats) in enumerate(list(analyzer.stats.items())[:3]):
        print(f"   {name}: mean={stats['mean']:+.4f}, std={stats['std']:.4f}")

    print("\n3. Gradient Checking:")
    checker = GradientChecker(model)
    stats = checker.check_gradients(train_loader, nn.CrossEntropyLoss(), num_batches=3)
    issues = checker.report_issues(vanishing_threshold=1e-7, exploding_threshold=1e3)

    print("\n4. Dead Neuron Detection:")
    detector = DeadNeuronDetector(model, threshold=1e-6)
    detector.analyze(train_loader, num_batches=5)
    dead_neurons = detector.report()

    if not dead_neurons:
        print("   No dead neurons detected!")

    print("\n5. Training Debugging (NaN/Inf detection):")
    debugger = TrainingDebugger(model)
    debugger.enable_nan_detection()

    # Simulate training with debugging
    model.train()
    for i, (data, target) in enumerate(train_loader):
        if i >= 3:
            break
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)

        if debugger.check_loss(loss, step=i):
            loss.backward()

        if debugger.nan_detected or debugger.inf_detected:
            print("   Issues detected!")
            for diag in debugger.get_diagnosis():
                print(f"     {diag}")
            break
    else:
        print("   No NaN/Inf issues detected in training")

    print("\n✓ Profiling demo completed!\n")


def demo_4_experiment_tracking():
    """Demo 4: Experiment Tracking."""
    print("\n" + "=" * 70)
    print("DEMO 4: Experiment Tracking")
    print("=" * 70)

    # Create tracker
    tracker = ExperimentTracker(log_dir="demo_experiments")

    print("\nCreating and running 3 experiments...")

    # Create and run experiments
    for i in range(3):
        exp = tracker.create_experiment(
            name=f"demo_experiment_{i + 1}",
            tags=["demo", f"run_{i + 1}"],
            config={"seed": 42 + i, "model": "simple_mlp"},
        )

        # Log parameters
        exp.log_params(
            {"lr": 0.001 * (i + 1), "batch_size": 32, "epochs": 10, "hidden_dim": 128}
        )

        # Simulate training with metrics
        print(f"  Experiment {i + 1}: Simulating training...")
        for epoch in range(5):
            # Simulate improving metrics
            train_loss = 1.0 / (epoch + 1) + np.random.randn() * 0.05
            val_acc = 0.5 + 0.08 * epoch + np.random.randn() * 0.03
            train_acc = 0.45 + 0.09 * epoch + np.random.randn() * 0.02

            exp.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "learning_rate": 0.001 * (i + 1) * (0.9**epoch),
                },
                step=epoch,
            )

        exp.finish()
        print(f"    Completed with final val_acc: {val_acc:.4f}")

    # List experiments
    print("\nAll experiments:")
    experiments = tracker.list_experiments()
    for exp in experiments:
        tags_str = ", ".join(exp["tags"][:2]) + ("..." if len(exp["tags"]) > 2 else "")
        print(f"  {exp['name']:20s} - {exp['status']:10s} - tags: [{tags_str}]")

    # Get best experiment
    best = tracker.get_best_experiment(metric="val_accuracy", mode="max")
    if best:
        print(f"\nBest experiment: {best['name']}")
        print(f"  {best['metric']} = {best['best_score']:.4f}")

    # Get metrics summary
    print("\nMetrics Summary (Experiment 1):")
    exp = tracker.get_experiment(experiments[0]["run_id"])
    if exp:
        summary = exp.get_metrics_summary()
        for metric, stats in list(summary.items())[:3]:
            print(f"  {metric}: final={stats['final']:.4f}, best={stats['max']:.4f}")

    print("\n✓ Experiment tracking demo completed!\n")


def demo_5_mixed_precision():
    """Demo 5: Mixed Precision Training."""
    print("\n" + "=" * 70)
    print("DEMO 5: Mixed Precision Training")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\nCUDA not available - skipping mixed precision demo")
        print("(Mixed precision training requires CUDA)")
        return

    model = create_sample_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader, _ = create_sample_data()

    print("\nMixed Precision Trainer:")
    mp_trainer = MixedPrecisionTrainer(model, optimizer, enabled=True)

    model.train()
    for i, (data, target) in enumerate(train_loader):
        if i >= 3:
            break

        with mp_trainer.autocast():
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)

        mp_trainer.backward(loss)
        mp_trainer.step()

        print(f"   Batch {i + 1}: Loss = {loss.item():.4f}")

    print("\n✓ Mixed precision demo completed!\n")


def run_all_demos():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("FISHSTICK ADVANCED TOOLS COMPREHENSIVE DEMO")
    print("=" * 70)
    print("\nNew modules created:")
    print("  ✓ fishstick.augmentation - Advanced data augmentation")
    print("  ✓ fishstick.profiling - Model profiling and debugging")
    print("  ✓ fishstick.experiments - Experiment tracking")
    print("  ✓ fishstick.training.distributed - Distributed training utilities")
    print("=" * 70)

    demos = [
        ("Advanced Training Utilities", demo_1_advanced_training),
        ("Advanced Augmentation", demo_2_augmentation),
        ("Profiling & Debugging", demo_3_profiling),
        ("Experiment Tracking", demo_4_experiment_tracking),
        ("Mixed Precision Training", demo_5_mixed_precision),
    ]

    for name, func in demos:
        try:
            func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {str(e)}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETED!")
    print("=" * 70)
    print("\nSummary of new tools:")
    print("  1. Augmentation: CutOut, MixUp, CutMix, RandAugment")
    print("  2. Profiling: ModelProfiler, GradientChecker, DeadNeuronDetector")
    print("  3. Experiment Tracking: Experiment, ExperimentTracker")
    print("  4. Distributed Training: EMA, GradientAccumulator, MixedPrecision")
    print("\nFor more details, see:")
    print("  - fishstick/augmentation/advanced.py")
    print("  - fishstick/profiling/debug.py")
    print("  - fishstick/experiments/tracker.py")
    print("  - fishstick/training/distributed.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_demos()
