"""
Data preprocessing utilities for CIFAR-10 dataset.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


class DataPreprocessor:
    """Handle data preprocessing for CIFAR-10 dataset."""

    def __init__(self, img_size=160, batch_size=128):
        self.img_size = img_size
        self.batch_size = batch_size
        self.image_shape = (img_size, img_size, 3)

        # Define preprocessing layers
        self.preprocessing = keras.Sequential([
            layers.Rescaling(1./255.0, offset=0),
            layers.Resizing(
                self.image_shape[0],
                self.image_shape[1],
                interpolation='bilinear'
            )
        ], name="preprocessing")

        # Define data augmentation layers
        self.data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.15),
            layers.RandomContrast(factor=0.1),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ], name="data_augmentation")

    def load_cifar10(self):
        """Load and return CIFAR-10 dataset."""
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        # Convert labels to categorical
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

        return (x_train, y_train), (x_test, y_test)

    def create_class_splits(self, x, y, included_classes):
        """
        Create dataset splits based on class inclusion.

        Args:
            x: Input data
            y: Labels
            included_classes: List of class indices to include

        Returns:
            Tuple of (x_included, y_included, x_excluded, y_excluded)
        """
        x_included = []
        y_included = []
        x_excluded = []
        y_excluded = []

        for x_sample, y_sample in zip(x, y):
            class_idx = np.argmax(y_sample) if len(y_sample.shape) > 0 else y_sample[0]

            if class_idx in included_classes:
                x_included.append(x_sample)
                y_included.append(y_sample)
            else:
                x_excluded.append(x_sample)
                y_excluded.append(y_sample)

        return (np.array(x_included), np.array(y_included),
                np.array(x_excluded), np.array(y_excluded))

    def create_tensorflow_dataset(self, x, y, augment=True, shuffle=True):
        """
        Create TensorFlow dataset with preprocessing.

        Args:
            x: Input data
            y: Labels
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle the data

        Returns:
            tf.data.Dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        if shuffle:
            dataset = dataset.shuffle(dataset.cardinality())

        dataset = dataset.batch(self.batch_size)

        if augment:
            dataset = dataset.map(
                lambda x, y: (self.data_augmentation(self.preprocessing(x)), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            dataset = dataset.map(
                lambda x, y: (self.preprocessing(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def prepare_datasets(self, n_classes=None):
        """
        Prepare training and test datasets.

        Args:
            n_classes: List of classes to include in n-dataset.
                      If None, uses all classes except class 9.

        Returns:
            Dict containing all dataset splits
        """
        if n_classes is None:
            n_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # All except class 9

        # Load CIFAR-10
        (x_train, y_train), (x_test, y_test) = self.load_cifar10()

        # Create class-based splits
        x_train_n, y_train_n, x_train_m, y_train_m = self.create_class_splits(
            x_train, y_train, n_classes
        )
        x_test_n, y_test_n, x_test_m, y_test_m = self.create_class_splits(
            x_test, y_test, n_classes
        )

        # Create TensorFlow datasets
        datasets = {
            'train_full': self.create_tensorflow_dataset(x_train, y_train, augment=True),
            'test_full': self.create_tensorflow_dataset(x_test, y_test, augment=False, shuffle=False),
            'train_n': self.create_tensorflow_dataset(x_train_n, y_train_n, augment=True),
            'test_n': self.create_tensorflow_dataset(x_test_n, y_test_n, augment=False, shuffle=False),
            'train_m': self.create_tensorflow_dataset(x_train_m, y_train_m, augment=True),
            'test_m': self.create_tensorflow_dataset(x_test_m, y_test_m, augment=False, shuffle=False),
        }

        return datasets