"""
Helper functions for loading and preprocessing external datasets like Tiny ImageNet.
"""

import os
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import wget
from zipfile import ZipFile
import numpy as np


def download_tiny_imagenet(output_dir='./'):
    """
    Download and extract Tiny ImageNet dataset.

    Args:
        output_dir: Directory to download and extract to

    Returns:
        Path to extracted dataset
    """
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    # Download if not exists
    zip_path = os.path.join(output_dir, "tiny-imagenet-200.zip")
    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet...")
        wget.download(url, out=output_dir)

    # Extract if not already extracted
    extracted_path = os.path.join(output_dir, "tiny-imagenet-200")
    if not os.path.exists(extracted_path):
        print("Extracting Tiny ImageNet...")
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("Extraction completed.")

    return extracted_path


def load_images_from_folder(folder_path, label, image_shape=(160, 160, 3)):
    """
    Load images from a folder and assign a label.

    Args:
        folder_path: Path to folder containing images
        label: Label to assign to all images
        image_shape: Target shape for images

    Returns:
        Tuple of (images, labels) lists
    """
    image_files = glob.glob(os.path.join(folder_path, "*.JPEG"))
    print(f"Found {len(image_files)} images in {folder_path}")

    images = []
    labels = []

    for img_file in image_files:
        img = tf.io.read_file(img_file)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [image_shape[0], image_shape[1]])
        images.append(img)
        labels.append(label)

    return images, labels


def load_tiny_imagenet_classes(dataset_path, class_mappings, image_shape=(160, 160, 3)):
    """
    Load specific classes from Tiny ImageNet.

    Args:
        dataset_path: Path to extracted Tiny ImageNet dataset
        class_mappings: Dictionary mapping folder names to class indices
                       e.g., {'n03706229': 8, 'n04456115': 9}
        image_shape: Target image shape

    Returns:
        Combined images and labels arrays
    """
    all_images = []
    all_labels = []

    for folder_name, class_label in class_mappings.items():
        folder_path = os.path.join(dataset_path, "train", folder_name, "images")
        images, labels = load_images_from_folder(folder_path, class_label, image_shape)
        all_images.extend(images)
        all_labels.extend(labels)

    return np.array(all_images), np.array(all_labels)


def create_external_m_dataset(images, labels, test_ratio=0.2, random_state=42):
    """
    Create train/test split for external M-dataset.

    Args:
        images: Array of images
        labels: Array of labels
        test_ratio: Ratio for test split
        random_state: Random seed

    Returns:
        Tuple of (x_train_m, x_test_m, y_train_m, y_test_m)
    """
    x_train_m, x_test_m, y_train_m, y_test_m = train_test_split(
        images, labels,
        test_size=test_ratio,
        random_state=random_state
    )

    return x_train_m, x_test_m, y_train_m, y_test_m


def setup_imagenet_preprocessing(image_shape=(160, 160, 3)):
    """
    Set up preprocessing layers for ImageNet data.

    Args:
        image_shape: Target image shape

    Returns:
        Tuple of (preprocessing, data_augmentation) layers
    """
    preprocessing = keras.Sequential([
        layers.Rescaling(1./255.0, offset=0),
        layers.Resizing(image_shape[0], image_shape[1], interpolation='bilinear')
    ], name="preprocessing")

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.15),
        layers.RandomContrast(factor=0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ], name="data_augmentation")

    return preprocessing, data_augmentation


def setup_tiny_imagenet_experiment(output_dir='./', class_mappings=None, image_shape=(160, 160, 3)):
    """
    Complete setup for Tiny ImageNet experiment.

    Args:
        output_dir: Directory for dataset download
        class_mappings: Dictionary mapping folder names to class indices
        image_shape: Target image shape

    Returns:
        Dictionary containing processed datasets and preprocessing layers
    """
    if class_mappings is None:
        # Default mapping: magnetic compass -> class 8, torch -> class 9
        class_mappings = {
            'n03706229': 8,  # magnetic compass
            'n04456115': 9   # torch
        }

    # Download and extract dataset
    dataset_path = download_tiny_imagenet(output_dir)

    # Load images
    images, labels = load_tiny_imagenet_classes(dataset_path, class_mappings, image_shape)

    # Create train/test splits
    x_train_m, x_test_m, y_train_m, y_test_m = create_external_m_dataset(images, labels)

    # Convert labels to categorical
    y_train_m = tf.keras.utils.to_categorical(y_train_m, num_classes=10)
    y_test_m = tf.keras.utils.to_categorical(y_test_m, num_classes=10)

    # Setup preprocessing
    preprocessing, data_augmentation = setup_imagenet_preprocessing(image_shape)

    # Create datasets
    batch_size = 128
    auto = tf.data.AUTOTUNE

    ds_train_m = tf.data.Dataset.from_tensor_slices((x_train_m, y_train_m))
    ds_train_m = ds_train_m.shuffle(len(x_train_m)).batch(batch_size)
    ds_train_m = ds_train_m.map(
        lambda x, y: (data_augmentation(preprocessing(x)), y)
    ).prefetch(auto)

    ds_test_m = tf.data.Dataset.from_tensor_slices((x_test_m, y_test_m))
    ds_test_m = ds_test_m.batch(batch_size)
    ds_test_m = ds_test_m.map(
        lambda x, y: (preprocessing(x), y)
    ).prefetch(auto)

    return {
        'ds_train_m': ds_train_m,
        'ds_test_m': ds_test_m,
        'x_train_m': x_train_m,
        'x_test_m': x_test_m,
        'y_train_m': y_train_m,
        'y_test_m': y_test_m,
        'preprocessing': preprocessing,
        'data_augmentation': data_augmentation,
        'dataset_path': dataset_path
    }