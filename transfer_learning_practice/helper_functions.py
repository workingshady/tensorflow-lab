### We create a bunch of helpful functions throughout the course.
### Storing them here so they're easily accessible.

import tensorflow as tf

# Set matplotlib style for all plots to 'whitegrid' for enhanced visuals
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img

# Enhanced confusion matrix plotting with whitegrid and improved visuals
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).

  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0]

  fig, ax = plt.subplots(figsize=figsize)
  # Use a more visually appealing colormap
  cax = ax.imshow(cm, interpolation='nearest', cmap='Blues')
  fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

  # Set grid for whitegrid style
  ax.set_axisbelow(True)
  ax.grid(False)

  if classes is not None:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  ax.set(
      title="Confusion Matrix",
      xlabel="Predicted label",
      ylabel="True label",
      xticks=np.arange(n_classes),
      yticks=np.arange(n_classes)
  )
  ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=text_size)
  ax.set_yticklabels(labels, fontsize=text_size)

  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Add text annotations with enhanced formatting
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      text = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
    else:
      text = f"{cm[i, j]}"
    ax.text(j, i, text,
            ha="center", va="center",
            color="white" if cm[i, j] > threshold else "black",
            fontsize=text_size, fontweight='bold', bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))

  # Add grid lines for whitegrid style
  ax.set_xticks(np.arange(-.5, n_classes, 1), minor=True)
  ax.set_yticks(np.arange(-.5, n_classes, 1), minor=True)
  ax.grid(which="minor", color="gray", linestyle=':', linewidth=0.5, alpha=0.5)
  ax.tick_params(which="minor", bottom=False, left=False)

  plt.tight_layout()
  if savefig:
    fig.savefig("confusion_matrix.png", bbox_inches='tight', dpi=300)
  plt.show()

# Make a function to predict on images and plot them (works with multi-class)
def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  img = load_and_prep_image(filename)
  pred = model.predict(tf.expand_dims(img, axis=0))

  if len(pred[0]) > 1: # check for multi-class
    pred_class = class_names[pred.argmax()] # if more than one output, take the max
    confidence = np.max(tf.nn.softmax(pred[0])) * 100
  else:
    pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round
    confidence = float(tf.sigmoid(pred[0][0])) * 100

  plt.figure(figsize=(5,5))
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}\nConfidence: {confidence:.2f}%", fontsize=16, fontweight='bold')
  plt.axis('off')
  plt.grid(False)
  plt.tight_layout()
  plt.show()

import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

# Enhanced loss and accuracy curves with whitegrid and improved visuals
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  plt.figure(figsize=(10, 5))
  # Plot loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label='Training Loss', color='royalblue', linewidth=2)
  plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=2, linestyle='--')
  plt.title('Loss', fontsize=16, fontweight='bold')
  plt.xlabel('Epochs', fontsize=12)
  plt.ylabel('Loss', fontsize=12)
  plt.legend()
  plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)

  # Plot accuracy
  plt.subplot(1, 2, 2)
  plt.plot(epochs, accuracy, label='Training Accuracy', color='seagreen', linewidth=2)
  plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='firebrick', linewidth=2, linestyle='--')
  plt.title('Accuracy', fontsize=16, fontweight='bold')
  plt.xlabel('Epochs', fontsize=12)
  plt.ylabel('Accuracy', fontsize=12)
  plt.legend()
  plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)

  plt.tight_layout()
  plt.show()

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.

    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here)
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    plt.figure(figsize=(12, 8))
    # Accuracy subplot
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy', color='seagreen', linewidth=2)
    plt.plot(total_val_acc, label='Validation Accuracy', color='firebrick', linewidth=2, linestyle='--')
    plt.axvline(x=initial_epochs-1, color='gray', linestyle=':', linewidth=2, label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)

    # Loss subplot
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss', color='royalblue', linewidth=2)
    plt.plot(total_val_loss, label='Validation Loss', color='orange', linewidth=2, linestyle='--')
    plt.axvline(x=initial_epochs-1, color='gray', linestyle=':', linewidth=2, label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.7)

    plt.tight_layout()
    plt.show()

# Create function to unzip a zipfile into current working directory
# (since we're going to be downloading and unzipping a few files)
import zipfile
def unzip_data(filename, extract_dir=None):
    """
    Unzips filename into the specified directory or current working directory.

    Args:
        filename (str): a filepath to a target zip folder to be unzipped.
        extract_dir (str, optional): directory to extract files into. Defaults to current directory.
    """
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(path=extract_dir)

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory

  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results

def create_model_callbacks(model_name, checkpoint_dir):
    all_epochs_dir = os.path.join(checkpoint_dir, "all_epochs")
    os.makedirs(all_epochs_dir, exist_ok=True)

    checkpoint_all_epochs = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(all_epochs_dir, f"{model_name}_checkpoint.h5"),
        save_best_only=False,
        save_weights_only=True,
        verbose=1
    )

    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, f"{model_name}_best.h5"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    return [checkpoint_all_epochs, checkpoint_best]
