"""
Example usage of the Interactive Training Visualization Dashboard

This script demonstrates all the features of the dashboard module.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from fishstick.visualization.dashboard import (
    TrainingDashboard,
    LayerVisualizer,
    AttentionVisualizer,
    PredictionVisualizer,
    RealTimePlot,
    DashboardCallback,
    DashboardServer,
    quick_plot_loss,
    quick_plot_metrics,
    create_interactive_dashboard,
)
from fishstick.training.advanced import Trainer, accuracy


def example_1_basic_dashboard():
    """Example 1: Basic Training Dashboard with Callback Integration"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Training Dashboard")
    print("=" * 70)

    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Create synthetic data
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    X_val = torch.randn(20, 10)
    y_val = torch.randint(0, 2, (20,))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16)

    # Create dashboard and callback
    dashboard = TrainingDashboard(
        save_dir="examples_output/example1",
        real_time=False,  # Disable real-time for batch training
        backend="matplotlib",
    )

    callback = DashboardCallback(dashboard)

    # Create trainer with callback
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        callbacks=[callback],
        metrics={"accuracy": accuracy},
    )

    # Train
    print("\nTraining for 10 epochs...")
    history = trainer.fit(train_loader, val_loader, epochs=10, verbose=True)

    print("\n✓ Training complete! Check examples_output/example1/ for visualizations")


def example_2_layer_visualization():
    """Example 2: Layer Activation and Gradient Visualization"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Layer Visualization")
    print("=" * 70)

    # Create CNN model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 7 * 7, 64)
            self.fc2 = nn.Linear(64, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleCNN()
    visualizer = LayerVisualizer(model, save_dir="examples_output/example2")

    # Register hooks
    print("\nRegistering hooks on layers...")
    visualizer.register_hooks()

    # Forward pass
    input_data = torch.randn(4, 1, 28, 28)
    output = model(input_data)

    # Backward pass
    loss = output.sum()
    loss.backward()

    print("\nGenerating visualizations...")

    # Visualize activations
    visualizer.plot_activations("conv1", max_channels=8)
    print("  ✓ conv1 activations saved")

    visualizer.plot_activations("conv2", max_channels=8)
    print("  ✓ conv2 activations saved")

    # Visualize weights
    visualizer.plot_weight_distribution(max_layers=6)
    print("  ✓ weight distributions saved")

    # Visualize gradients
    visualizer.plot_gradient_flow()
    print("  ✓ gradient flow saved")

    # Clean up
    visualizer.remove_hooks()

    print("\n✓ All layer visualizations saved to examples_output/example2/")


def example_3_prediction_visualization():
    """Example 3: Prediction Analysis with Confusion Matrix and ROC Curves"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Prediction Visualization")
    print("=" * 70)

    # Simulate classification results
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5

    # Generate targets with class imbalance
    targets = np.random.choice(n_classes, n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])

    # Generate predictions (mostly correct with some errors)
    predictions = targets.copy()
    error_mask = np.random.rand(n_samples) < 0.15
    predictions[error_mask] = np.random.randint(0, n_classes, error_mask.sum())

    # Generate probabilities
    probabilities = np.random.rand(n_samples, n_classes)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    # Boost probability for predicted class
    for i in range(n_samples):
        probabilities[i, predictions[i]] += 0.5
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    # Create visualizer
    class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer"]
    visualizer = PredictionVisualizer(
        class_names=class_names, save_dir="examples_output/example3"
    )

    # Update with data
    print("\nProcessing predictions...")
    visualizer.update(
        torch.tensor(predictions), torch.tensor(targets), torch.tensor(probabilities)
    )

    # Generate plots
    print("\nGenerating visualizations...")

    visualizer.plot_confusion_matrix()
    print("  ✓ Confusion matrix saved")

    visualizer.plot_confusion_matrix(normalize=True)
    print("  ✓ Normalized confusion matrix saved")

    visualizer.plot_roc_curve()
    print("  ✓ ROC curves saved")

    visualizer.plot_pr_curve()
    print("  ✓ Precision-Recall curves saved")

    print("\n✓ All prediction visualizations saved to examples_output/example3/")


def example_4_real_time_plotting():
    """Example 4: Real-time Plotting with Matplotlib and Plotly"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Real-time Plotting")
    print("=" * 70)

    # Matplotlib backend (static)
    print("\nCreating matplotlib plot...")
    plot_mpl = RealTimePlot(backend="matplotlib", title="Training Progress")

    for epoch in range(20):
        loss = 1.0 / (epoch + 1)
        acc = min(0.95, 0.5 + epoch * 0.02)

        plot_mpl.append("loss", epoch, loss)
        plot_mpl.append("accuracy", epoch, acc)

    plot_mpl.update()
    plot_mpl.save("examples_output/example4/training_mpl.png")
    print("  ✓ Matplotlib plot saved")

    # Plotly backend (interactive)
    print("\nCreating interactive plotly plot...")
    plot_plotly = RealTimePlot(backend="plotly", title="Interactive Training Progress")

    for epoch in range(20):
        loss = 1.0 / (epoch + 1)
        val_loss = 1.1 / (epoch + 1)

        plot_plotly.append("train_loss", epoch, loss)
        plot_plotly.append("val_loss", epoch, val_loss)

    plot_plotly.save("examples_output/example4/training_interactive.html")
    print("  ✓ Interactive plotly plot saved")

    print("\n✓ Real-time plotting examples saved to examples_output/example4/")


def example_5_quick_functions():
    """Example 5: Quick Convenience Functions"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Quick Convenience Functions")
    print("=" * 70)

    # Generate sample data
    epochs = 50
    train_loss = [1.0 / (i + 1) + 0.05 * np.random.randn() for i in range(epochs)]
    val_loss = [1.2 / (i + 1) + 0.08 * np.random.randn() for i in range(epochs)]

    # Quick loss plot
    print("\nCreating quick loss plot...")
    quick_plot_loss(train_loss, val_loss, "examples_output/example5/quick_loss.png")
    print("  ✓ Loss plot saved")

    # Quick metrics plot
    print("\nCreating quick metrics plot...")
    metrics = {
        "train_accuracy": [min(0.95, 0.5 + i * 0.01) for i in range(epochs)],
        "val_accuracy": [min(0.93, 0.48 + i * 0.009) for i in range(epochs)],
        "learning_rate": [0.001 * (0.95**i) for i in range(epochs)],
    }
    quick_plot_metrics(metrics, "examples_output/example5/quick_metrics.png")
    print("  ✓ Metrics plot saved")

    # Interactive dashboard
    print("\nCreating interactive HTML dashboard...")
    history = {
        "loss": train_loss,
        "val_loss": val_loss,
        "accuracy": metrics["train_accuracy"],
        "val_accuracy": metrics["val_accuracy"],
    }
    create_interactive_dashboard(
        history, "examples_output/example5/interactive_dashboard.html"
    )
    print("  ✓ Interactive dashboard saved")

    print("\n✓ All quick function examples saved to examples_output/example5/")


def example_6_attention_visualization():
    """Example 6: Attention Visualization (requires transformers)"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Attention Visualization")
    print("=" * 70)

    try:
        from transformers import BertModel, BertTokenizer

        print("\nLoading BERT model...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        model.eval()

        visualizer = AttentionVisualizer(model, save_dir="examples_output/example6")

        # Process text
        text = "The quick brown fox jumps over the lazy dog."
        print(f"\nProcessing text: '{text}'")

        inputs = tokenizer(text, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        visualizer.set_tokens(tokens)

        # Get attention weights
        with torch.no_grad():
            outputs = model(**inputs)

        attentions = torch.stack(outputs.attentions)

        print("\nGenerating attention visualizations...")

        # Single head
        visualizer.plot_attention_heatmap(attentions, layer_idx=0, head_idx=0)
        print("  ✓ Single head attention saved")

        # Averaged heads
        visualizer.plot_attention_heatmap(attentions, layer_idx=0, head_idx=None)
        print("  ✓ Averaged attention saved")

        # Multi-head comparison
        visualizer.plot_multi_head_comparison(attentions, layer_idx=0, max_heads=8)
        print("  ✓ Multi-head comparison saved")

        # Attention rollout
        visualizer.plot_attention_rollout(attentions)
        print("  ✓ Attention rollout saved")

        print("\n✓ All attention visualizations saved to examples_output/example6/")

    except ImportError:
        print("\n⚠ transformers library not installed. Skipping this example.")
        print("  Install with: pip install transformers")


def example_7_web_dashboard():
    """Example 7: Web-based Dashboard with Flask"""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Web Dashboard (Flask Server)")
    print("=" * 70)

    try:
        server = DashboardServer(host="localhost", port=5000)

        print("\n⚠ This example requires Flask and Flask-SocketIO.")
        print("  Install with: pip install flask flask-socketio")
        print("\nTo run the web dashboard:")
        print("  1. Start server: server.start()")
        print("  2. Open browser to http://localhost:5000")
        print(
            "  3. During training, call: server.update_metrics(epoch=N, loss=X, accuracy=Y)"
        )
        print("  4. Stop server: server.stop()")

        print("\n✓ Web dashboard setup complete")

    except ImportError as e:
        print(f"\n⚠ {e}")
        print("  Install required packages and try again.")


def main():
    """Run all examples"""
    import os

    # Create output directory
    os.makedirs("examples_output", exist_ok=True)

    print("\n" + "=" * 70)
    print("FISHSTICK INTERACTIVE TRAINING VISUALIZATION DASHBOARD")
    print("=" * 70)
    print("\nThis script demonstrates all features of the dashboard module.")
    print("Outputs will be saved to 'examples_output/' directory.")

    # Run examples
    examples = [
        example_1_basic_dashboard,
        example_2_layer_visualization,
        example_3_prediction_visualization,
        example_4_real_time_plotting,
        example_5_quick_functions,
        example_6_attention_visualization,
        example_7_web_dashboard,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n⚠ Error in {example.__name__}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 70)
    print("\nCheck the 'examples_output/' directory for all visualizations.")
    print("\nFile structure:")
    for root, dirs, files in os.walk("examples_output"):
        level = root.replace("examples_output", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files[:5]:  # Limit output
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")


if __name__ == "__main__":
    main()
