# FiCoVuL: Fine-grained Code Vulnerability Localization with Graph Attention Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Deep Learning: PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-orange.svg)](https://pytorch.org/)

## Abstract

FiCoVuL is a state-of-the-art system for fine-grained code vulnerability detection and localization. Leveraging Graph Attention Networks (GAT) and Program Dependency Graphs (PDG), FiCoVuL accurately identifies security vulnerabilities in C code and precisely locates the affected code positions.

## Overview

Code vulnerability detection is critical for software security, but existing methods often struggle with fine-grained localization. FiCoVuL addresses this challenge by:
- Converting C code into Program Dependency Graphs (PDG) using Joern
- Employing Graph Attention Networks (GAT) to capture complex code semantics
- Providing both graph-level vulnerability classification and node-level vulnerability localization
- Supporting multiple datasets and preprocessing methods

## Key Features

- **Graph-Based Representation**: Converts code to PDG to preserve structural and semantic information
- **Attention Mechanism**: Uses GAT to focus on relevant code components
- **Fine-Grained Localization**: Identifies specific vulnerable nodes within the code
- **Extensible Design**: Supports multiple datasets and preprocessing techniques
- **User-Friendly Interface**: Easy-to-use scripts for training, testing, and custom code analysis

## Prerequisites

- **Python**: 3.8 or higher
- **Conda**: For environment management
- **Java/Scala**: Required for Joern tool

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Netsec-SJTU/FiCoVuL.git
cd FiCoVuL
```

### 2. Create and Activate Conda Environment

```bash
conda create -n ficovul python=3.8
conda activate ficovul
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Joern Tool

FiCoVuL uses Joern to convert C code into Program Dependency Graphs:

```bash
# Navigate to the Doxygen directory (in the project root)
cd Doxygen
# Download Joern CLI
wget https://github.com/joernio/joern/releases/download/v1.1.190/joern-cli.zip
unzip joern-cli.zip

# Make Joern executable
chmod +x joern-cli/joern
chmod +x joern-cli/joern-parse
```

## Data Processing

### Input Format

- Input data should be C language source code files (`.c` extension)
- Each file should ideally contain a single function for optimal processing and localization

### 1. Generate Program Dependency Graphs

```bash
# Run from project root directory
cd Doxygen
python run_joern.py --dataset MY_DATASET
```

### 2. Preprocess Data

Convert Joern-generated graphs to model-compatible format:

```bash
# Run from project root directory
cd FiCoVuL
python dataset/data_preprocess.py
```

### Preprocessed Data Location

```
data/datasets/[DATASET_NAME]_RAW/
```

- `train.json`, `valid.json`, `test.json`: Split datasets
- `node_type_dict.pkl`, `word_lexical_dict.pkl`, `word_value_dict.pkl`: Vocabulary mappings

## Model Training

### Basic Training

```bash
# Run from project root directory
cd FiCoVuL
python train3.py --task_name my_task
```

### Configuration

Edit `configs/config.json5` to customize training parameters:

```json5
{
  "task_name": "my_task",
  "dataset_name": "CroVul",
  "preprocess": {
    "method": "RAW",
    "proportion": [8, 1, 1]  // [train, validation, test] split ratio
  },
  "model": {
    "num_of_gat_layers": 4,
    "num_heads_per_gat_layer": [8, 4, 4, 6],
    "num_features_per_gat_layer": [64, 32, 64, 64, 64]
  },
  "run": {
    "num_of_epochs": 200,
    "batch_size": 128,
    "lr": 0.0005
  }
}
```

### Training Results

Training results are stored in three locations:

1. **Model Binaries**
   - Path: `data/models/binaries/`
   - Format: `gat_{dataset_name}_{task_name}_{number}.pth`
   - Example: `gat_CroVul_my_task_000000.pth`

2. **Model Checkpoints**
   - Path: `data/models/checkpoints/`
   - Best model: `gat_{dataset_name}_{task_name}_best.pth`
   - Epoch checkpoints: `gat_{dataset_name}_ckpt_{task_name}_epoch_{epoch+1}.pth`

3. **TensorBoard Logs**
   - Path: `data/runs/{task_name}/`
   - View: `tensorboard --logdir data/runs/`

## Model Testing

### Batch Testing

1. Configure test settings in `configs/config.json5`:

```json5
{
  "run": {
    "load_model": "data/models/binaries/gat_CroVul_my_task_000000.pth",
    "test_only": true,
    "loader_batch_size": 1
  }
}
```

2. Run batch testing:

```bash
# Run from project root directory
cd FiCoVuL
python test3.py
```

### Test Results

Test results are saved in:

```
data/results/
```

- Filename format: `test_results_{model_name}_{timestamp}.json`
- Contains graph-level predictions and node-level vulnerability localization

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset_name` | Dataset identifier | `CroVul` |
| `preprocess.method` | Preprocessing technique | `RAW` |
| `model.num_of_gat_layers` | Number of GAT layers | `4` |
| `model.num_heads_per_gat_layer` | Attention heads per GAT layer | `[8, 4, 4, 6]` |
| `run.num_of_epochs` | Training epochs | `200` |
| `run.batch_size` | Batch size | `128` |
| `run.lr` | Learning rate | `0.0005` |
| `run.classification_threshold` | Vulnerability classification threshold | `0.5` |

## Project Structure

```
├── configs/              # Configuration files
│   └── config.json5
├── dataset/              # Data processing module
│   ├── data_loader.py    # Data loading utilities
│   ├── data_preprocess.py # Data preprocessing pipeline
│   └── metrics.py        # Evaluation metrics
├── models/               # Model definitions
│   ├── layers/           # Neural network layers
│   │   ├── embedding_layer.py          # Node embedding
│   │   ├── gat_layer.py                # Graph Attention Layer
│   │   └── nodes_to_graph_representation.py # Graph-level representation
│   └── GATClassification.py # GAT model for vulnerability detection
├── utils/                # Utility functions
├── data/                 # Data and results storage
│   ├── datasets/         # Processed datasets
│   ├── models/           # Model files
│   │   ├── binaries/     # Final model binaries
│   │   └── checkpoints/  # Training checkpoints
│   └── results/          # Test results
├── train3.py             # Training script
├── test3.py              # Testing script
└── README.md             # Project documentation
```

## Usage Examples

### Example 1: Train on Custom Dataset

```bash
# Edit config.json5 to set dataset_name to "MY_DATASET"
python train3.py --task_name my_custom_train
```

### Example 2: Test with Pretrained Model

```bash
# Edit config.json5 to set load_model path
python test3.py
```

## Performance Metrics

FiCoVuL reports the following metrics:
- **Graph-level**: Precision, Recall, Accuracy, F1-score
- **Node-level**: Precision, Recall, Accuracy, F1-score
- **Weighted**: Combined metrics with graph-level (70%) and node-level (30%) weights



This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or collaboration:

- **GitHub**: [https://github.com/Netsec-SJTU/FiCoVuL](https://github.com/Netsec-SJTU/FiCoVuL)
- **Issues**: [https://github.com/Netsec-SJTU/FiCoVuL/issues](https://github.com/Netsec-SJTU/FiCoVuL/issues)

## Acknowledgments

- Special thanks to the Joern development team for their powerful code analysis tool

---

**Disclaimer**: This tool is for research and educational purposes only. Always verify results with manual code review before making security decisions.