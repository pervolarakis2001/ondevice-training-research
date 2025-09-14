"""
Training utilities and trainer class.
"""

import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np


class OnDeviceTrainer:
    """Trainer class for on-device training experiments."""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or self._default_config()

    def _default_config(self):
        """Default training configuration."""
        return {
            'epochs': 10,
            'validation_split': 0.2,
            'early_stopping': True,
            'patience': 5,
            'monitor': 'val_loss'
        }

    def train(self, train_dataset, validation_dataset=None, callbacks=None):
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            callbacks: List of Keras callbacks

        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = []

        if self.config['early_stopping']:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=self.config['monitor'],
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)

        history = self.model.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.config['epochs'],
            callbacks=callbacks,
            verbose=1
        )

        return history

    def evaluate(self, test_dataset, verbose=1):
        """
        Evaluate the model on test data.

        Args:
            test_dataset: Test dataset
            verbose: Verbosity level

        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions and true labels
        y_pred = []
        y_true = []

        for x_batch, y_batch in test_dataset:
            predictions = self.model.model.predict(x_batch, verbose=0)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(np.argmax(y_batch, axis=1))

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        # Calculate metrics
        accuracy = np.mean(y_pred == y_true)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_true
        }

        if verbose:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")

        return results

    def fine_tune(self, train_dataset, validation_dataset=None,
                  fine_tune_epochs=5, fine_tune_lr=1e-5):
        """
        Fine-tune the model with unfrozen base layers.

        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            fine_tune_epochs: Number of fine-tuning epochs
            fine_tune_lr: Fine-tuning learning rate

        Returns:
            Fine-tuning history
        """
        # Unfreeze the base model
        base_model = self.model.model.layers[0]
        base_model.trainable = True

        # Use lower learning rate for fine-tuning
        self.model.model.compile(
            optimizer=tf.keras.optimizers.Adam(fine_tune_lr),
            loss=self.model.Loss,
            metrics=['accuracy']
        )

        # Fine-tune
        fine_tune_history = self.model.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=fine_tune_epochs,
            verbose=1
        )

        return fine_tune_history

    def save_model(self, filepath, save_format='keras'):
        """
        Save the trained model.

        Args:
            filepath: Path to save the model
            save_format: Format to save ('keras', 'h5', 'tf')
        """
        if save_format == 'keras':
            self.model.model.save(filepath)
        elif save_format == 'h5':
            self.model.model.save_weights(filepath)
        elif save_format == 'tf':
            tf.saved_model.save(self.model.model, filepath)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")

    def load_model(self, filepath, load_format='keras'):
        """
        Load a saved model.

        Args:
            filepath: Path to the saved model
            load_format: Format of the saved model
        """
        if load_format == 'keras':
            self.model.model = tf.keras.models.load_model(filepath)
        elif load_format == 'h5':
            self.model.model.load_weights(filepath)
        elif load_format == 'tf':
            self.model.model = tf.saved_model.load(filepath)
        else:
            raise ValueError(f"Unsupported load format: {load_format}")