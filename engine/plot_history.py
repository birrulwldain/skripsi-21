import json
import os
import argparse
import matplotlib.pyplot as plt

def plot_and_save_history(history, save_path):
    """
    Membuat dan menyimpan plot histori training dan validasi.
    """
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':')

    # Plot F1-Score and Accuracy
    ax2.plot(epochs, history['val_f1_macro'], 'go-', label='Validation F1-Score (Macro)')
    ax2.set_title('Validation F1-Score & Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1-Score')
    ax2.grid(True, linestyle=':')

    ax2b = ax2.twinx()
    ax2b.plot(epochs, history['val_acc'], 'ms--', label='Validation Accuracy (Subset)', alpha=0.6)
    ax2b.set_ylabel('Accuracy (Subset)', color='m')
    ax2b.tick_params(axis='y', labelcolor='m')

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nPlot histori training disimpan di: {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training history from a JSON file.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the training progress JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output plot image.")
    
    args = parser.parse_args()

    try:
        with open(args.json_path, 'r') as f:
            progress_data = json.load(f)
        histories = progress_data['histories']
        plot_and_save_history(histories, args.output_path)
        print(f"Plot berhasil dibuat dan disimpan di {args.output_path}")
    except FileNotFoundError:
        print(f"Error: File {args.json_path} tidak ditemukan. Pastikan path sudah benar.")
    except Exception as e:
        print(f"Terjadi kesalahan saat membuat plot: {e}")
