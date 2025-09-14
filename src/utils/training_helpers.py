"""
Helper functions for training experiments and batch processing.
"""

import tensorflow as tf
import numpy as np
import random
import time
from tensorflow import keras


def take_samples(num, dataset, classes=None):
    """
    Take a specified number of samples from each class in the dataset.

    Args:
        num: Number of samples per class
        dataset: TensorFlow dataset
        classes: List of class indices to consider

    Returns:
        List of (image, label) tuples
    """
    if classes is None:
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    labels = tf.keras.utils.to_categorical(classes, num_classes=10)
    images_labels = []
    samples_per_class = {cls: 0 for cls in classes}

    for x, y in dataset:
        class_idx = np.argmax(y)
        if y in labels and samples_per_class[class_idx] < num:
            images_labels.append((x, y))
            samples_per_class[class_idx] += 1
        if all(count == num for count in samples_per_class.values()):
            break

    return images_labels


def get_unique_images(classes, dataset):
    """
    Get one unique image per class from the dataset.

    Args:
        classes: List of class indices
        dataset: TensorFlow dataset

    Returns:
        List of (image, label) tuples with one sample per class
    """
    labels = tf.keras.utils.to_categorical(classes, num_classes=10)
    images_labels = []

    for x, y in dataset:
        if y in labels and not any(np.array_equal(y.numpy(), item[1].numpy()) for item in images_labels):
            images_labels.append((x, y))
        if len([item[1] for item in images_labels]) == len(classes):
            break

    return images_labels


class PrintLossCallback(tf.keras.callbacks.Callback):
    """Custom callback to print loss during training."""

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: loss = {logs['loss']}, accuracy = {logs['accuracy']}")

    def on_batch_end(self, batch, logs=None):
        print(f"Batch {batch+1}: loss = {logs['loss']}")


def reshape_y_for_single_batch(x, y):
    """Reshape labels for single-batch processing."""
    y = tf.reshape(y, (1, 10))
    return x, y


def create_stratified_test_dataset(dataset, test_size=270, random_seed=42, classes=None):
    """
    Create a stratified test dataset from the input dataset.

    Args:
        dataset: Input TensorFlow dataset
        test_size: Total number of samples in stratified dataset
        random_seed: Random seed for reproducibility
        classes: List of classes to include

    Returns:
        Stratified TensorFlow dataset
    """
    from sklearn.model_selection import train_test_split

    if classes is None:
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    x_test = []
    y_test = []

    for x, y in dataset:
        x_test.append(x.numpy())
        y_test.append(np.argmax(y))

    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    _, x_sampled, _, y_sampled = train_test_split(
        x_test, y_test,
        test_size=test_size,
        stratify=y_test,
        random_state=random_seed
    )

    y_categorical = tf.keras.utils.to_categorical(y_sampled, num_classes=10)
    ds_test_sampled = tf.data.Dataset.from_tensor_slices((x_sampled, y_categorical))
    ds_test_stratified = ds_test_sampled.map(reshape_y_for_single_batch)

    return ds_test_stratified


def run_on_device_training_experiment(model_path, ds_train_n, ds_train_m, ds_test_n, ds_test_m,
                                     ds_test_stratified=None, ds_test_sampled=None,
                                     learning_rate=1.4e-5, max_iterations=1000):
    """
    Run the complete on-device training experiment.

    Args:
        model_path: Path to the pre-trained model
        ds_train_n: Training dataset for N classes
        ds_train_m: Training dataset for M classes
        ds_test_n: Test dataset for N classes
        ds_test_m: Test dataset for M classes
        ds_test_stratified: Stratified test dataset
        ds_test_sampled: Subsampled test dataset
        learning_rate: Learning rate for training
        max_iterations: Maximum number of iterations

    Returns:
        Dictionary containing training results
    """
    from tensorflow.keras.models import load_model

    # Load model
    model = load_model(model_path)
    opt = keras.optimizers.Adam(
        learning_rate=learning_rate,
        epsilon=0.002,
        amsgrad=True,
        weight_decay=1e-5
    )
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

    # Initialize result tracking
    results = {
        'acc_n': [],
        'acc_n_stratified': [],
        'acc_n_subsampling': [],
        'acc_m': [],
        'loss_n': [],
        'loss_n_stratified': [],
        'loss_n_subsampling': [],
        'loss_m': [],
        'training_times': []
    }

    # Training loop setup
    dataset_n = take_samples(5, ds_train_n)
    m_dict = {}
    convergence_reached = False
    convergence_counter = float('inf')

    for i in range(max_iterations):
        print(f"Iteration {i+1}")

        # Get next M sample
        m_data = ds_train_m.take(1)

        if m_data not in list(m_dict.keys()):
            # Create training batch
            ds_n_samples = random.sample(dataset_n, 4)
            x_n = [item[0] for item in ds_n_samples]
            y_n = [item[1] for item in ds_n_samples]
            ds_n_batch = tf.data.Dataset.from_tensor_slices((x_n, y_n))
            training_batch = m_data.concatenate(ds_n_batch)

            # Train
            start_time = time.time()
            model.fit(training_batch, epochs=1, verbose=0)
            training_time = time.time() - start_time
            results['training_times'].append(training_time)

            # Schedule next use of this M sample
            k = random.randint(5, 10)
            m_dict[m_data] = i + k

            # Evaluate on all test sets
            loss_n, acc_n = model.evaluate(ds_test_n, verbose=0)
            results['acc_n'].append(acc_n)
            results['loss_n'].append(loss_n)

            if ds_test_stratified is not None:
                loss_n_str, acc_n_str = model.evaluate(ds_test_stratified, verbose=0)
                results['acc_n_stratified'].append(acc_n_str)
                results['loss_n_stratified'].append(loss_n_str)

            if ds_test_sampled is not None:
                loss_n_sub, acc_n_sub = model.evaluate(ds_test_sampled, verbose=0)
                results['acc_n_subsampling'].append(acc_n_sub)
                results['loss_n_subsampling'].append(loss_n_sub)

            loss_m, acc_m = model.evaluate(ds_test_m, verbose=0)
            results['acc_m'].append(acc_m)
            results['loss_m'].append(loss_m)

            print(f"Test accuracy on N: {acc_n:.4f}")
            print(f"Test accuracy on M: {acc_m:.4f}")

            # Check convergence
            if abs(acc_n - acc_m) <= 0.02 and not convergence_reached:
                convergence_counter = i + 5
                convergence_reached = True
                print(f"Convergence reached at iteration {i+1}")

        # Check for reusing stored M samples
        for m_sample, next_iteration in list(m_dict.items()):
            if next_iteration == i:
                # Update schedule
                k = random.randint(5, 10)
                m_dict[m_sample] += k

                # Create and train on new batch
                ds_n_samples = random.sample(dataset_n, 4)
                x_n = [item[0] for item in ds_n_samples]
                y_n = [item[1] for item in ds_n_samples]
                ds_n_batch = tf.data.Dataset.from_tensor_slices((x_n, y_n))
                training_batch = m_sample.concatenate(ds_n_batch)

                start_time = time.time()
                model.fit(training_batch, epochs=1, verbose=0)
                training_time = time.time() - start_time
                results['training_times'].append(training_time)

                # Evaluate (same as above)
                loss_n, acc_n = model.evaluate(ds_test_n, verbose=0)
                results['acc_n'].append(acc_n)
                results['loss_n'].append(loss_n)

                if ds_test_stratified is not None:
                    loss_n_str, acc_n_str = model.evaluate(ds_test_stratified, verbose=0)
                    results['acc_n_stratified'].append(acc_n_str)
                    results['loss_n_stratified'].append(loss_n_str)

                if ds_test_sampled is not None:
                    loss_n_sub, acc_n_sub = model.evaluate(ds_test_sampled, verbose=0)
                    results['acc_n_subsampling'].append(acc_n_sub)
                    results['loss_n_subsampling'].append(loss_n_sub)

                loss_m, acc_m = model.evaluate(ds_test_m, verbose=0)
                results['acc_m'].append(acc_m)
                results['loss_m'].append(loss_m)

                # Check convergence again
                if abs(acc_n - acc_m) <= 0.02 and not convergence_reached:
                    convergence_counter = i + 5
                    convergence_reached = True

        if i >= convergence_counter:
            print(f"Stopping at iteration {i+1}")
            break

        print()

    return results


def save_results_to_files(results, output_dir='./'):
    """
    Save experimental results to text files.

    Args:
        results: Dictionary containing experimental results
        output_dir: Directory to save files
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    for key, values in results.items():
        if values:  # Only save non-empty lists
            filepath = os.path.join(output_dir, f'{key}.txt')
            with open(filepath, 'w') as f:
                for value in values:
                    f.write(f"{value}\n")
            print(f"Saved {key} to {filepath}")