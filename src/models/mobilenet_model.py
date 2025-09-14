"""
MobileNetV2-based model for on-device training.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers


class OnDeviceTrainingModel(tf.Module):
    """On-device training model wrapper for MobileNetV2."""

    def __init__(self, num_classes=10, img_size=160, learning_rate=1.4e-5):
        self.num_classes = num_classes
        self.img_size = img_size
        self.image_shape = (img_size, img_size, 3)

        # Build model
        self.model = self._build_model()
        self.model.trainable = True

        # Freeze batch normalization layers
        for layer in self.model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        # Setup optimizer and loss
        self.opt = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            epsilon=0.002,
            amsgrad=True,
            weight_decay=1e-5
        )
        self.Loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        self.model.compile(optimizer=self.opt, loss=self.Loss, metrics=["accuracy"])

    def _build_model(self):
        """Build MobileNetV2-based model."""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.image_shape,
            include_top=False,
            alpha=1.0
        )
        base_model.trainable = False

        # Add custom top layers
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)

        top_dropout_rate = 0.1
        x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout1")(x)

        outputs = tf.keras.layers.Dense(
            self.num_classes,
            activation="softmax",
            name="pred"
        )(x)

        model = tf.keras.Model(base_model.input, outputs)
        return model

    @tf.function(input_signature=[
        tf.TensorSpec([5, None, None, 3], tf.float32),
        tf.TensorSpec([5, None], tf.float32),
    ])
    def train(self, x, y):
        """Training function for on-device training."""
        with tf.GradientTape() as tape:
            prediction = self.model(x, training=True)
            loss = self.model.loss(y, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        return {"loss": loss}

    @tf.function(input_signature=[
        tf.TensorSpec([None, None, None, 3], tf.float32),
    ])
    def infer(self, x):
        """Inference function."""
        probabilities = self.model(x)
        return {"output": probabilities}

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        """Save model weights."""
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path,
            tensor_names=tensor_names,
            data=tensors_to_save,
            name='save'
        )
        return {"checkpoint_path": checkpoint_path}

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        """Restore model weights."""
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path,
                tensor_name=var.name,
                dt=var.dtype,
                name='restore'
            )
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors

    def convert_to_tflite(self, saved_model_dir="saved_model"):
        """Convert model to TensorFlow Lite format."""
        # Save the model with signatures
        tf.saved_model.save(
            self,
            saved_model_dir,
            signatures={
                'train': self.train.get_concrete_function(),
                'infer': self.infer.get_concrete_function(),
                'save': self.save.get_concrete_function(),
                'restore': self.restore.get_concrete_function(),
            }
        )

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.experimental_enable_resource_variables = True
        tflite_model = converter.convert()

        return tflite_model