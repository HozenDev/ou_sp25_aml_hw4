"""
Advanced Machine Learning, 2025, HW4

Author: Enzo B. Durel (enzo.durel@gmail.com)

Plotting script to analyse models results and performances.
"""

from chesapeake_loader4 import create_single_dataset
import tensorflow as tf

# Gpus initialization
gpus = tf.config.experimental.list_physical_devices('GPU')
n_visible_devices = len(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# Set threading parallelism
import os
cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
if cpus_per_task > 1:
    tf.config.threading.set_intra_op_parallelism_threads(cpus_per_task // 2)
    tf.config.threading.set_inter_op_parallelism_threads(cpus_per_task // 2)

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from parser import check_args, create_parser
    
#########################################
#             Load Results              #
#########################################

def load_trained_model(model_dir, substring_name):
    """
    Load a trained models
    """
    model_files = [f for f in os.listdir(model_dir) if substring_name in f and f.endswith(".keras")]

    if not model_files:
        raise ValueError(f"No model found in {model_dir} matching {substring_name}")

    model_path = os.path.join(model_dir, model_files[0])
    model = tf.keras.models.load_model(model_path)

    return model
    

def load_results(results_dir):
    results = []
    files = []
    for r_dir in results_dir:
        files.extend([os.path.join(r_dir, f) for f in os.listdir(r_dir) if f.endswith(".pkl")])

    for filename in files:
        with open(filename, "rb") as fp:
            data = pickle.load(fp)
            results.append(data)

    return results

#########################################
#             Plot Methods              #
#########################################

def plot_test_sample_with_predictions(test_ds, shallow_model, deep_model, num_classes, num_samples):
    """
    Plots models output from the testing dataset.
    """

    # Extract images from test dataset
    images = []
    for img_batch, _ in test_ds.take(num_samples):
        images.append(img_batch.numpy())  # Convert TensorFlow tensor to numpy
    images = np.concatenate(images, axis=0)
    num_samples = min(num_samples, len(images))

    # Predicting testing images
    shallow_predictions = shallow_model.predict(images[:num_samples])
    deep_predictions = deep_model.predict(images[:num_samples])

    # Format images and class names
    images = (images * 255).astype(np.uint8) # Convert images to uint8 for plotting
    class_names = ['Plug Adapter', 'Scissors', 'Light Bulb', 'Cup']

    # Fix issue when num_samples = 1
    _, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure 2D shape

    for i in range(num_samples):
        img = np.squeeze(images[i]) 

        # Shallow model predictions
        axes[i, 0].imshow(img.astype("uint8"))
        axes[i, 0].axis('off')
        shallow_title = "\n".join([f"{class_names[j]}: {shallow_predictions[i][j]:.2f}" for j in range(num_classes)])
        axes[i, 0].set_title(f"Shallow Model\n{shallow_title}", fontsize=10)

        # Deep model predictions
        axes[i, 1].imshow(img.astype("uint8"))
        axes[i, 1].axis('off')
        deep_title = "\n".join([f"{class_names[j]}: {deep_predictions[i][j]:.2f}" for j in range(num_classes)])
        axes[i, 1].set_title(f"Deep Model\n{deep_title}", fontsize=10)

    plt.tight_layout()
    plt.savefig("figure_3.png")


def plot_combined_confusion_matrix(args, models, num_classes, class_names, title="Confusion Matrix", filename="figure_4.png"):
    """
    Computes and plots a combined confusion matrix from multiple model rotations.
    """
    all_y_true = []
    all_y_pred = []

    for i, model in enumerate(models):

        test_ds = create_single_dataset(base_dir=args.dataset,
                                        full_sat=True,
                                        patch_size=None,
                                        partition='valid',
                                        fold=i,
                                        filt='*',
                                        cache_path='',
                                        repeat=False,
                                        shuffle=None,
                                        batch_size=args.batch,
                                        prefetch=args.prefetch,
                                        num_parallel_calls=args.num_parallel_calls)
        
        for x_batch, y_batch in test_ds:
            preds = model.predict(x_batch)
            y_pred = np.argmax(preds, axis=-1)  # shape: (B, 256, 256)
            y_true = y_batch.numpy()            # shape: (B, 256, 256)

            all_y_pred.append(y_pred.flatten())
            all_y_true.append(y_true.flatten())
            
    y_true_flat = np.concatenate(all_y_true)
    y_pred_flat = np.concatenate(all_y_pred)

    # 4. Compute confusion matrix
    labels = list(range(7))
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)

    # 5. Plot
    _, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar(im, ax=ax)
    plt.xticks(np.arange(7), class_names, rotation=45)
    plt.yticks(np.arange(7), class_names)

    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(filename)


def plot_test_accuracy_scatter(shallow_results, deep_results):
    """
    Plots a scatter plot comparing test accuracies of shallow vs. deep models
    """

    shallow_accuracies, deep_accuracies = [], []
    rotations = list(range(min(len(shallow_results), len(deep_results))))
    
    for result in shallow_results:
        shallow_accuracies.append(result["predict_testing_eval"][1]) 

    for result in deep_results:
        deep_accuracies.append(result["predict_testing_eval"][1])

    # Define color map: distinct color for each rotation
    colors = plt.cm.get_cmap("tab10", len(rotations))

    # Scatter plot
    plt.figure(figsize=(7,7))
    for i in range(len(rotations)):
        plt.scatter(shallow_accuracies[i], deep_accuracies[i], color=colors(i), label=f"Rot {rotations[i]}")
        plt.text(shallow_accuracies[i], deep_accuracies[i], f"{rotations[i]}", fontsize=10, ha='right', va='bottom')

    # Diagonal line    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # y = x line

    plt.xlabel("Shallow Model Accuracy")
    plt.ylabel("Deep Model Accuracy")
    plt.title("Test Accuracy: Deep vs. Shallow")
    plt.legend()
    plt.grid(True)
    plt.savefig("figure_5.png")


#########################################
#            Main Function              #
#########################################
    
if __name__ == "__main__":
    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    # Load testing dataset and number of classes
    num_classes = 7
    class_names = ["No Class", "Water", "Forest", "Low Veg", "Barren", "Impervious", "Road"]

    # Load trained models
    deep_model_dir = "./models/deep_1/"
    shallow_model_dir = "./models/shallow_1/"
    shallow_models = []
    deep_models = []
    nb_rotation = 5
    
    for i in range(nb_rotation):
        try:
            shallow_model = load_trained_model(deep_model_dir, f"rot_0{i}")
            shallow_models.append(shallow_model)
        except Exception as e:
            print(f"Error loading shallow model: {e}")
        
        try:
            deep_model = load_trained_model(shallow_model_dir, f"rot_0{i}")
            deep_models.append(deep_model)
        except Exception as e:
            print(f"Error loading deep model: {e}")

    print(len(shallow_models), len(deep_models))
            
    # Figure 3: Test Sample with Predictions
    # plot_test_sample_with_predictions(test_ds, shallow_models[0], deep_models[0], num_samples=5, num_classes=num_classes)

    # Figure 4a: Shallow Model Confusion Matrix
    # plot_combined_confusion_matrix(args=args, models=shallow_models, class_names=class_names, title="Shallow Model Confusion Matrix", filename="figure_4a.png", num_classes=num_classes)

    # Figure 4b: Deep Model Confusion Matrix
    # plot_combined_confusion_matrix(args=args, models=deep_models, class_names=class_names, title="Deep Model Confusion Matrix", filename="figure_4b.png", num_classes=num_classes)

    # Figure 5: Test Accuracy Scatter Plot
    shallow_results = load_results([shallow_model_dir])
    deep_results = load_results([deep_model_dir])
    plot_test_accuracy_scatter(shallow_results, deep_results)

