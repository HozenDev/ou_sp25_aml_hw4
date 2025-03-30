import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from hw3_base import load_precached_folds, check_args
from hw3_parser import create_parser

CORE50_METADATA_PATH = "/home/fagg/datasets/core50/core50_df.pkl"  # Adjust path if needed

#########################################
#             Class Mapping             #
#########################################

def get_class_mappings(core50_pkl_path, num_classes):
    """
    Loads core50_df.pkl and extracts object names for the trained classes.

    :param core50_pkl_path: Path to the core50 metadata pickle file.
    :param num_classes: Number of classes in the model's predictions
    :return: Dictionary mapping class indices (0 to num_classes-1) to object names
    """
    # Load the metadata file
    with open(core50_pkl_path, "rb") as f:
        df = pickle.load(f)

    # Extract unique class-object mappings
    class_mapping = df[['class', 'object']].drop_duplicates().sort_values(by='class')

    # Only keep the first `num_classes` entries (the ones used in training)
    trained_classes = class_mapping.iloc[:num_classes]

    # Create a dictionary mapping class index (0,1,2,3...) to human-readable object names
    class_dict = {i: f"Class {trained_classes.iloc[i]['class']} - Object {trained_classes.iloc[i]['object']}"
                  for i in range(num_classes)}

    return class_dict

#########################################
#             Load Results              #
#########################################

def load_trained_model(model_dir, substring_name):
    """   
    :param model_dir: Directory containing the trained model
    :param regex: Regular expression to match the model file
    :return: Loaded model
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

def plot_test_sample_with_predictions(test_ds, shallow_model, deep_model, num_samples=5, num_classes=4):
    """
    Plots test images with probability distributions from the shallow and deep models.

    :param test_ds: TensorFlow dataset containing test images and labels
    :param shallow_model: Trained shallow model
    :param deep_model: Trained deep model
    :param num_samples: Number of test images to visualize
    :param num_classes: Number of classes in the model's predictions
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
        # Remove extra dimensions if needed
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


def plot_combined_confusion_matrix(models, test_ds, title="Confusion Matrix", filename="figure_4.png", num_classes=4):
    """
    Computes and plots a combined confusion matrix from multiple model rotations.

    :param model_paths: List of trained model file paths (one per rotation)
    :param test_ds: TensorFlow dataset containing test images and labels
    :param core50_pkl_path: Path to core50 metadata file (core50_df.pkl)
    :param title: Title of the confusion matrix plot
    """
    y_true, y_pred = [], []
    class_names = ['Plug Adapter', 'Scissors', 'Light Bulb', 'Cup']

    for model in models:
        for images, labels in test_ds:
            y_true.extend(labels.numpy())  # Ground truth labels
            y_pred.extend(np.argmax(model.predict(images), axis=1))  # Predicted classes

    # Compute the combined confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[class_names[i] for i in range(num_classes)])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.savefig(filename)


def plot_test_accuracy_scatter(shallow_results, deep_results):
    """
    Plots a scatter plot comparing test accuracies of shallow vs. deep models.

    :param shallow_results_files: List of paths to shallow model results .pkl files
    :param deep_results_files: List of paths to deep model results .pkl files
    """

    shallow_accuracies, deep_accuracies = [], []
    rotations = list(range(len(shallow_results)))
    
    for result in shallow_results:
        shallow_accuracies.append(result["predict_testing_eval"][1])  # Extract accuracy

    for result in deep_results:
        deep_accuracies.append(result["predict_testing_eval"][1])  # Extract accuracy

    # Define color map
    colors = plt.cm.get_cmap("tab10", len(rotations))  # Use a distinct color per rotation

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
    _, _, test_ds, num_classes = load_precached_folds(args)

    # Load trained models
    shallow_models = []
    deep_models = []
    nb_rotation = 5
    
    for i in range(nb_rotation):
        try:
            shallow_model = load_trained_model("./models/shallow_2", f"rot_0{i}")
            shallow_models.append(shallow_model)
        except Exception as e:
            print(f"Error loading shallow model: {e}")
        
        try:
            deep_model = load_trained_model("./models/deep_2/", f"rot_0{i}")
            deep_models.append(deep_model)
        except Exception as e:
            print(f"Error loading deep model: {e}")

    # Figure 3: Test Sample with Predictions
    plot_test_sample_with_predictions(test_ds, shallow_models[0], deep_models[0], num_samples=5, num_classes=num_classes)

    # Figure 4a: Shallow Model Confusion Matrix
    plot_combined_confusion_matrix(shallow_models, test_ds, title="Shallow Model Confusion Matrix", filename="figure_4a.png", num_classes=num_classes)

    # Figure 4b: Deep Model Confusion Matrix
    plot_combined_confusion_matrix(deep_models, test_ds, title="Deep Model Confusion Matrix", filename="figure_4b.png", num_classes=num_classes)

    # Figure 5: Test Accuracy Scatter Plot
    shallow_results = load_results(["./models/shallow_2/"])
    deep_results = load_results(["./models/deep_2/"])
    plot_test_accuracy_scatter(shallow_results, deep_results)

