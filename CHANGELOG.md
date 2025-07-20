# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and organization
- Comprehensive documentation and README
- Professional GitHub repository setup
- Contributing guidelines and code standards

### Changed
- Reorganized project structure for better maintainability
- Moved source code to `src/` directory
- Separated data, models, utils, and experiments into distinct modules

### Fixed
- Updated import paths for reorganized structure
- Improved .gitignore to exclude large files and sensitive data

## [1.0.0] - 2024-11-29

### Added
- **Core Model Implementation**
  - BERT-Base-Uncased for fact-checking classification
  - Custom WeightedLossTrainer for handling class imbalance
  - ClippyAdagrad optimizer with layer-specific learning rates
  - Early stopping callback to prevent overfitting

- **Data Processing**
  - Comprehensive data preprocessing pipeline
  - Tokenization and encoding utilities
  - Dataset preparation for training and evaluation

- **Training Pipeline**
  - Complete training workflow with experiment tracking
  - Weights & Biases integration for experiment management
  - Comprehensive evaluation metrics (accuracy, F1-score, precision, recall)
  - Visualization tools for training curves and confusion matrices

- **Evaluation and Testing**
  - Model evaluation on validation set
  - Prediction generation for test data
  - Performance analysis and reporting

- **Experiment Management**
  - Multiple experiment configurations
  - Baseline model implementation
  - Experiment comparison utilities
  - Results aggregation and analysis

### Technical Features
- **Model Architecture**
  - 3-class classification (SUPPORTS, REFUTES, NOT ENOUGH INFO)
  - Input format: `[CLS] claim [SEP] evidence [SEP]`
  - Max sequence length: 512 tokens
  - Dropout rate: 0.2 for regularization

- **Training Configuration**
  - Learning rate: 5e-5
  - Batch size: 12
  - Number of epochs: 15
  - Warmup steps: 500
  - Weight decay: 0.01

- **Performance Metrics**
  - Baseline BERT: Accuracy ~40%, F1-Score ~0.39
  - Class-Weighted Training: Accuracy ~49%, F1-Score ~0.47
  - Precision: ~0.49-0.50 across experiments
  - Recall: ~0.48-0.50 across experiments

### Dependencies
- PyTorch >= 2.0.0
- Transformers >= 4.46.0
- Weights & Biases >= 0.15.0
- Other supporting libraries (see requirements.txt)

---

## Version History

- **v1.0.0**: Initial release with complete fact-checking pipeline
- **Unreleased**: Repository reorganization and documentation improvements

## Contributors

- **Wei-Han Tu** - Initial development and project setup
- **CSE 256 Course Staff** - Academic guidance and support

## Acknowledgments

- **Hugging Face** - Transformers library and BERT implementation
- **Weights & Biases** - Experiment tracking and visualization
- **UCSD CSE Department** - Computational resources and academic support 