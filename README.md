# On-Device Training Research Project

This repository contains research code for on-device training experiments using MobileNetV2 and CIFAR-10 dataset and ImageNer. The project explores adaptive sample selection and efficient training strategies for mobile and edge devices. This project was part of my diploma thesis 

## Project Overview

This research investigates on-device training methodologies with focus on:

- **Model Optimization**: Converting TensorFlow models to TensorFlow Lite for on-device deployment
- **Sample Selection**: Identifying optimal training samples using confidence-based metrics
- **External Dataset Integration**: Incorporating external datasets (Tiny ImageNet) for improved training
- **Performance Analysis**: Comprehensive evaluation of training strategies and convergence patterns

## Key Features

### 1. Modular Architecture
- **Models**: MobileNetV2-based architecture optimized for on-device training
- **Data Processing**: Automated CIFAR-10 preprocessing with class splits
- **Training**: Flexible training pipeline with multiple evaluation strategies
- **Evaluation**: Comprehensive metrics 

### 2. Sample Selection Strategies
- **Best vs Second Best (BvSB)**: Confidence-based sample ranking
- **Stratified Sampling**: Balanced dataset creation for evaluation
- **Subsampling**: Distribution-preserving sample reduction

### 3. External Dataset Integration
- **Tiny ImageNet**: Integration of external classes for M-dataset
- **Automatic Download**: Streamlined dataset acquisition and preprocessing
- **Class Mapping**: Flexible class assignment for experimental design

### 4. TensorFlow Lite Support
- **Model Conversion**: Automatic conversion to TFLite format
- **Signature Preservation**: Maintaining training and inference signatures
- **Mobile Optimization**: Optimized for mobile and edge deployment

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/pervolarakis2001/ondevice-training-research.git
cd ondevice-training-research
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


#### Running Notebooks

Each notebook demonstrates different aspects of the research:

1. **Sample Selection**: `best_worst_sample_selection.ipynb`
   - Analyze model confidence on different samples
   - Generate best/worst sample collections

2. **Model Conversion**: `tflite_model_conversion.ipynb`
   - Convert models to TensorFlow Lite format
   - Test inference on mobile-optimized models

3. **Training Experiments**: `comprehensive_training_experiment.ipynb`
   - Run complete on-device training experiments
   - Compare different sampling strategies

## Research Methodology

### Experimental Design
- **N-Dataset**: 9 CIFAR-10 classes for base training
- **M-Dataset**: 1-2 external classes from Tiny ImageNet
- **Convergence Criterion**: Accuracy difference ≤ 2% between N and M datasets

### Key Metrics
- **Training Accuracy**: Performance on N and M datasets
- **Convergence Speed**: Number of iterations to reach criterion
- **Sample Efficiency**: Number of M-samples required
- **Training Time**: Per-step execution time analysis

### Sample Selection Methods
1. **Random Sampling**: Baseline approach
2. **Stratified Sampling**: Balanced class distribution
3. **Confidence-based**: Using BvSB scores for sample ranking

## Results

The project demonstrates:
- Efficient convergence with adaptive sample selection
- Reduced training time compared to traditional approaches
- Effective integration of external datasets
- Successful model deployment optimization

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{on_device_training_2024,
  title={On-Device Training with Adaptive Sample Selection},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## Contact

For questions or collaborations, please open an issue or contact [pervoalarakis3@gmail.com].
