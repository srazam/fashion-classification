# Fashion Image Classification

CNN models (VGG, AlexNet, and LeNet) for classifying **patterns on dressed clothing**, along with an investigation into reproducibility and variance in deep learning.

Findings from this work, *"Comparing Deep Learning Models Used for Image Classification on Patterns on Dressed Clothing"*, were presented at the [10th International Conference on Big Data Analytics, Data Mining and Computational Intelligence](https://www.iadisportal.org/bigdaci-csc-eh-2025-proceedings).

> Commits shown in 2024 were made in collaboration with [MSDS1203](https://github.com/MSDS1203).

---

## Table of Contents

- [Motivation](#motivation)
- [Key Results](#key-results)
- [Repository Contents](#repository-contents)
- [Models](#models)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Setup and Usage](#setup-and-usage)
- [Training Pipeline](#training-pipeline)
- [Reproducibility](#reproducibility)
- [Customizing Experiments](#customizing-experiments)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Motivation

Earlier work on fashion image classification has mostly focused on how accurately different clothing items can be told apart, and many of those datasets show garments in isolation rather than worn by a person. This project asks a different question: how well can CNNs classify **patterns on clothes that are actually being worn**? Better answers could support e-commerce recommendations or make tasks like categorizing a wardrobe easier, and they help probe the limits of different model architectures.

While running initial experiments, we found that results varied across different hardware. That led to a second goal: exploring **reproducibility and variance in deep learning models**, and modifying the CNNs to be more reproducible and less variable. After those changes and hyperparameter tuning, **VGG** performed best.

## Key Results

| Model | Notes |
| --- | --- |
| **VGG** | Best-performing model, with **68% classification accuracy** after modifications and hyperparameter tuning |
| AlexNet | Evaluated as a comparison model |
| LeNet | Evaluated as a comparison model |

See the published paper for the full comparison across models.

## Repository Contents

```
fashion-classification/
├── VGG.py          # VGG-16-style model with grid search
├── GSVGG.py        # VGG-16-style model (layers defined explicitly) with grid search
├── AlexNet.py      # AlexNet model
├── GSAlexNet.py    # AlexNet variant with grid search
├── LeNet.py        # LeNet model
├── GSLeNet.py      # LeNet variant with grid search
└── README.md
```

Each script is self-contained: it defines the model, loads the data, trains, and evaluates.

The `GS*` files are variants of the base models in which used Grid Search for hyper-tuning the respective models. 

## Models

- **VGG (VGG-16 style):** 13 convolutional layers (3x3 kernels, each followed by batch normalization and ReLU) grouped into five blocks separated by 2x2 max pooling, followed by three fully connected layers (4096, 4096, number of classes) with dropout of 0.5.
- **AlexNet:** the classic five-convolution architecture, adapted for this dataset.
- **LeNet:** the classic small CNN, used as a lightweight baseline.

## Dataset

The dataset consists of images of **dresses worn by people, categorized by pattern** [from this Kaggle dataset](https://www.kaggle.com/datasets/nitinsss/fashion-dataset-with-over-15000-labelled-images). The scripts expect it in a folder named `fashion_images` in the same directory as the scripts, organized in the standard [`torchvision.datasets.ImageFolder`](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) layout, with one subfolder per class:

```
fashion_images/
├── class_a/
│   ├── img001.jpg
│   └── ...
├── class_b/
│   └── ...
└── ...
```

The number of classes is detected automatically from the folder structure (the models default to 15 classes).

## Requirements

- Python 3.8 or newer
- [PyTorch](https://pytorch.org/) and torchvision
- NumPy
- Matplotlib
- **An NVIDIA GPU with CUDA.** The scripts exit immediately if CUDA is not available.

## Setup and Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/srazam/fashion-classification.git
   cd fashion-classification
   ```

2. **Create an environment and install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   pip install torch torchvision numpy matplotlib
   ```

   Install the PyTorch build that matches your CUDA version by following the selector at [pytorch.org](https://pytorch.org/get-started/locally/).

3. **Add the dataset** as a `fashion_images/` folder in the repository root (see [Dataset](#dataset)).

4. **Run a model**

   ```bash
   python VGG.py
   python AlexNet.py
   python LeNet.py
   # or the grid-search variants:
   python GSVGG.py
   python GSAlexNet.py
   python GSLeNet.py
   ```

Each run prints per-epoch training and validation loss and accuracy, shows a plot of the loss curves for the best configuration, and reports the final test loss and accuracy along with the best hyperparameters.

## Training Pipeline

The scripts follow the same pipeline (details below come from the VGG scripts):

1. **Preprocessing:** images are resized to 224x224, converted to tensors, and normalized with mean 0.5 and standard deviation 0.5 per channel.
2. **Data split:** 60% training, 20% validation, 20% test, using `random_split`.
3. **Loss:** cross-entropy.
4. **Grid search:** every combination of the following is trained from scratch:
   - Epochs: `10`, `20`
   - Batch size: `32`, `64` (plus `128` in `GSVGG.py`)
   - Optimizer: `Adam`, `SGD` (learning rate `0.001`)
5. **Model selection:** the configuration with the highest validation accuracy is kept.
6. **Evaluation:** the selected model is evaluated once on the held-out test set.

## Reproducibility

A central finding of this project was that results can differ between hardware setups, even with the same code. To reduce variance, the scripts:

- set the random seed (`42`) for PyTorch (CPU and all GPUs) and NumPy,
- enable `torch.backends.cudnn.deterministic = True`, and
- disable `torch.backends.cudnn.benchmark`.

Even so, exact numbers may still differ across GPUs, drivers, and PyTorch/CUDA versions. For fair comparisons, keep the hardware and software versions fixed and report them alongside your results.

## Customizing Experiments

- **Search space:** edit the `epochs`, `batch_sizes`, and `optimizers` variables near the bottom of each script.
- **Learning rate:** change the `lr=0.001` argument where the optimizer is created.
- **Split ratios:** adjust `train_size` and `val_size` after the dataset is loaded.
- **Dataset location:** change the `data_dir` variable.
- **Seed:** change the argument to `set_seed(...)`.

## Troubleshooting

**"CUDA isn't available; exiting program"**
The scripts require a CUDA-capable GPU. Check that your NVIDIA drivers are installed and that you installed a CUDA-enabled PyTorch build (`python -c "import torch; print(torch.cuda.is_available())"` should print `True`). To run on CPU, you would need to edit the device check in the script.

**Out-of-memory errors (especially VGG)**
VGG-16 at 224x224 is memory-hungry. Reduce the batch sizes in the grid search, or use a GPU with more memory.

**`FileNotFoundError` for `fashion_images`**
Make sure the dataset folder is named `fashion_images` and sits in the directory you run the script from.

**Long run times**
Each script trains a full model for every grid-search combination, so it can take a while. Reduce the search space to test quickly.

## Citation

If you use this code or build on these findings, please cite the paper:

```
Comparing Deep Learning Models Used for Image Classification on Patterns on Dressed Clothing.
10th International Conference on Big Data Analytics, Data Mining and Computational Intelligence, 2025.
```

## Acknowledgments

- [MSDS1203](https://github.com/MSDS1203) and Dr. Nic Herndon from East Carolina University for collaboration on the 2024 work
- The PyTorch and torchvision teams for the frameworks used throughout
