import pandas as pd
import matplotlib.pyplot as plt


def plot_multiple_training_logs(csv_filenames):
    # Initialize a figure with two side-by-side subplots (Loss and Accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Define a color palette to distinguish between the two models
    colors = ["#1f77b4", "#ff7f0e"]  # Blue for the first, Orange for the second

    # Loop through each provided file
    for i, filename in enumerate(csv_filenames):
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            print(f"Error: Could not find {filename}. Skipping...")
            continue

        # Create a clean display name (e.g., "Train Log" and "Qnn Train Log")
        model_name = filename.split(".")[0].replace("_", " ").title()

        # Pick the color for this model
        c = colors[i % len(colors)]

        # Extract data columns
        epochs = df["epoch"]
        train_loss = df["train_loss"]
        val_loss = df["val_loss"]
        train_acc = df["train_acc"]
        val_acc = df["val_acc"]

        # --- Subplot 1: Loss ---
        ax1.plot(
            epochs, train_loss, label=f"{model_name} (Train)", color=c, linestyle="-"
        )
        ax1.plot(epochs, val_loss, label=f"{model_name} (Val)", color=c, linestyle="--")

        # --- Subplot 2: Accuracy ---
        ax2.plot(
            epochs, train_acc, label=f"{model_name} (Train)", color=c, linestyle="-"
        )
        ax2.plot(epochs, val_acc, label=f"{model_name} (Val)", color=c, linestyle="--")

    # Set up labels, titles, grids, and legends for Loss subplot
    ax1.set_title("Loss Comparison")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend()

    # Set up labels, titles, grids, and legends for Accuracy subplot
    ax2.set_title("Accuracy Comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend()

    # Adjust layout to prevent overlap and display the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Specify the exact files in your directory
    logs_to_plot = ["train_log.csv", "qnn_train_log.csv"]
    plot_multiple_training_logs(logs_to_plot)
