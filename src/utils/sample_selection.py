"""
Utilities for sample selection based on model confidence.
"""

import numpy as np
import os
import zipfile
from PIL import Image


def calculate_bvsb(probabilities):
    """
    Calculate Best vs Second Best (BvSB) confidence scores.

    Args:
        probabilities: Model prediction probabilities

    Returns:
        Array of BvSB scores (higher means more confident)
    """
    sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
    bvsb = sorted_probs[:, 0] - sorted_probs[:, 1]
    return bvsb


def save_images_to_zip(images, zip_path, image_format='JPEG'):
    """
    Save a list of images to a zip file.

    Args:
        images: List of image tensors/arrays
        zip_path: Output zip file path
        image_format: Image format (e.g., 'JPEG', 'PNG')
    """
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for idx, image_tensor in enumerate(images):
            # Convert TensorFlow tensor to NumPy array
            if hasattr(image_tensor, 'numpy'):
                image_array = image_tensor.numpy().astype('uint8')
            else:
                image_array = np.array(image_tensor).astype('uint8')

            # Convert NumPy array to PIL Image
            image_pil = Image.fromarray(image_array)

            # Save the PIL Image to a temporary file
            temp_image_path = f'image_{idx}.{image_format.lower()}'
            image_pil.save(temp_image_path, format=image_format)

            # Add the image to the zip file
            zipf.write(temp_image_path, os.path.basename(temp_image_path))

            # Remove the temporary file
            os.remove(temp_image_path)


class SampleSelector:
    """Select best and worst samples based on model confidence."""

    def __init__(self, model):
        self.model = model

    def select_samples_by_confidence(self, dataset, target_class, n_samples=50):
        """
        Select best and worst samples for a specific class.

        Args:
            dataset: TensorFlow dataset
            target_class: Target class index
            n_samples: Number of samples to select for each category

        Returns:
            Dict containing best and worst samples with their metadata
        """
        confidence_data = []

        for x_batch, y_batch in dataset:
            predictions = self.model.predict(x_batch, verbose=0)

            for i in range(len(x_batch)):
                true_class = np.argmax(y_batch[i])
                predicted_class = np.argmax(predictions[i])

                # Only consider samples from target class that are correctly classified
                if true_class == target_class and predicted_class == true_class:
                    bvsb_score = calculate_bvsb(predictions[i:i+1])[0]
                    confidence_data.append({
                        'image': x_batch[i].numpy(),
                        'true_label': y_batch[i].numpy(),
                        'confidence': bvsb_score
                    })

        # Sort by confidence
        sorted_data = sorted(confidence_data, key=lambda x: x['confidence'])

        # Get worst (lowest confidence) and best (highest confidence) samples
        worst_samples = sorted_data[:n_samples]
        best_samples = sorted_data[-n_samples:]

        return {
            'worst': worst_samples,
            'best': best_samples,
            'all_data': sorted_data
        }

    def select_all_classes(self, dataset, class_list, n_samples=50, output_dir='./'):
        """
        Select samples for all classes and save to zip files.

        Args:
            dataset: TensorFlow dataset
            class_list: List of class indices to process
            n_samples: Number of samples per category
            output_dir: Directory to save output files

        Returns:
            Dict with results for each class
        """
        results = {}

        for class_idx in class_list:
            print(f"Processing class {class_idx}...")

            samples = self.select_samples_by_confidence(
                dataset, class_idx, n_samples
            )

            # Extract images
            worst_images = [item['image'] for item in samples['worst']]
            best_images = [item['image'] for item in samples['best']]

            # Save to zip files
            worst_zip_path = os.path.join(output_dir, f'worst_class_{class_idx}.zip')
            best_zip_path = os.path.join(output_dir, f'best_class_{class_idx}.zip')

            if worst_images:
                save_images_to_zip(worst_images, worst_zip_path)
            if best_images:
                save_images_to_zip(best_images, best_zip_path)

            results[class_idx] = {
                'worst_samples': len(worst_images),
                'best_samples': len(best_images),
                'worst_zip': worst_zip_path,
                'best_zip': best_zip_path
            }

        return results