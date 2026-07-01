# Fashion Similarity Search Engine

## Project Overview

The goal of this project is to build an end-to-end **Fashion Similarity Search Engine** that enables users to retrieve visually similar fashion products by uploading an image. Instead of relying on keyword-based search, the system will leverage deep learning to learn meaningful visual representations of fashion items and perform content-based image retrieval.

The project consists of two major stages:

1. **Representation Learning (Offline Stage)**
2. **Similarity Search and Visualization (Online Stage)**

The system will use a pretrained **ResNet50** convolutional neural network as the image encoder. Rather than using the ImageNet-pretrained weights directly, the model should be **fine-tuned on the fashion dataset through a supervised image classification task**. The purpose of this fine-tuning step is to adapt the learned visual features to the fashion domain, allowing the embeddings to better capture clothing-specific characteristics such as category, style, shape, and texture.

After fine-tuning, the classification layer will be removed, and the backbone network will be used as a feature extractor to generate high-dimensional embeddings for all images in the dataset. These embeddings will then be indexed using **FAISS** to enable efficient nearest-neighbor search.

The final application will be implemented as an interactive **Streamlit** web application that allows users to:

- Upload a fashion image.
- Retrieve visually similar products.
- Explore the learned embedding space.
- Visualize how different fashion categories are organized in the learned feature space.
- Inspect product metadata and dataset statistics.

---

# Project Objectives

The project should demonstrate the practical application of:

- Transfer Learning
- Fine-Tuning Pretrained CNNs
- Representation Learning
- Similarity Search
- Approximate Nearest Neighbor Search
- Dimensionality Reduction
- Interactive Visualization
- Web Application Development

---

# Dataset

Use the following dataset:

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

Expected dataset structure:

```text
data/
│
├── images/
│     ├── 15970.jpg
│     ├── 15971.jpg
│     └── ...
│
└── styles.csv
```

The metadata file contains information such as:

- gender
- masterCategory
- subCategory
- articleType
- baseColour
- season
- usage
- productDisplayName

---

# High-Level System Architecture

## Stage 1: Fine-Tuning

```text
Fashion Images
        ↓
Image Augmentation
        ↓
Pretrained ResNet50
        ↓
Classification Head
        ↓
Fashion Category Prediction
```

The pretrained ResNet50 should be fine-tuned on the dataset using a standard supervised image classification task.

Recommended classification targets:

### Option 1 (preferred)

```text
articleType
```

Examples:

- Tshirts
- Casual Shoes
- Jeans
- Dresses
- Shirts

### Option 2

```text
subCategory
```

### Option 3

```text
masterCategory
```

The project should be designed such that changing the classification target requires minimal code modifications.

---

# Fine-Tuning Requirements

## Model Architecture

```text
Input Image
      ↓
Resize(224×224)
      ↓
Normalization
      ↓
ResNet50 Backbone
      ↓
Global Average Pooling
      ↓
Fully Connected Layer
      ↓
Softmax Classification
```

Use:

```python
torchvision.models.resnet50(
    weights="IMAGENET1K_V2"
)
```

---

## Training Strategy

Initially:

- Freeze early layers.
- Train the classifier head.

Then:

- Unfreeze the final residual block (`layer4`).
- Fine-tune the network end-to-end.

Recommended optimization:

```text
Optimizer:
AdamW

Backbone Learning Rate:
1e-5

Classifier Learning Rate:
1e-4

Weight Decay:
1e-4
```

Recommended loss:

```text
CrossEntropyLoss
```

---

# Evaluation Metrics

During classification training, monitor:

- Training Loss
- Validation Loss
- Top-1 Accuracy
- Top-5 Accuracy
- Confusion Matrix

Save:

- best model checkpoint
- training curves
- classification metrics

---

# Stage 2: Feature Extraction

After training is complete:

1. Remove the classification head.
2. Use the fine-tuned ResNet50 backbone as a feature extractor.

Pipeline:

```text
Image
    ↓
Fine-tuned ResNet50
    ↓
2048-dimensional embedding
    ↓
L2 normalization
```

The output embeddings should be generated for every image in the dataset.

---

# Stage 3: Similarity Search

Build an image retrieval system:

```text
Query Image
      ↓
ResNet50 Encoder
      ↓
Embedding
      ↓
FAISS Search
      ↓
Top-K Similar Images
```

Use:

```text
faiss.IndexFlatIP
```

Since all embeddings are normalized:

```text
Inner Product = Cosine Similarity
```

---

# Stage 4: Embedding Visualization

Generate a two-dimensional representation of the learned embedding space using UMAP.

Pipeline:

```text
Embeddings
      ↓
UMAP
      ↓
2D Coordinates
```

The visualization should reveal natural clusters between:

- shoes
- shirts
- dresses
- accessories
- bags
- sportswear

---

# Streamlit Application

The final application should be implemented using **Streamlit**.

The application should contain four major sections.

---

# Page 1 — Similarity Search

Features:

- Upload an image.
- Compute embedding.
- Retrieve Top-K similar products.
- Display results in a grid.

Each result should display:

- image
- similarity score
- product name
- category
- color
- season

---

# Page 2 — Embedding Explorer

Display an interactive UMAP visualization.

Requirements:

- zoom
- pan
- category filtering
- hover information
- clickable points

Each point should display:

- product image
- product name
- category
- metadata

---

# Page 3 — Category Explorer

Allow users to:

- select categories
- filter by gender
- filter by season
- browse products belonging to specific categories

Display:

- category statistics
- nearest-neighbor examples
- embedding distribution

---

# Page 4 — Dataset Analytics

Display:

- total images
- number of categories
- category histogram
- gender distribution
- season distribution
- top article types
- images per class

Use interactive Plotly charts.

---

# Offline Pipeline

```text
Dataset
    ↓
Fine-Tune ResNet50
    ↓
Save Best Model
    ↓
Generate Embeddings
    ↓
Build FAISS Index
    ↓
Generate UMAP Coordinates
    ↓
Save Artifacts
```

---

# Online Pipeline

```text
Upload Image
      ↓
Fine-Tuned ResNet50 Encoder
      ↓
Embedding
      ↓
FAISS Search
      ↓
Retrieve Similar Products
      ↓
Interactive Streamlit Visualization
```

---

# Expected Artifacts

```text
artifacts/
│
├── best_model.pt
├── image_embeddings.npy
├── image_paths.pkl
├── metadata.pkl
├── faiss.index
├── umap_coords.npy
├── label_encoder.pkl
└── training_history.json
```

---

# Repository Structure

```text
fashion-search-engine/
│
├── data/
├── artifacts/
├── notebooks/
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── encoder.py
│   ├── build_embeddings.py
│   ├── build_index.py
│   ├── search_engine.py
│   ├── visualization.py
│   ├── utils.py
│   ├── config.py
│   └── logger.py
│
├── pages/
│   ├── similarity_search.py
│   ├── embedding_explorer.py
│   ├── category_explorer.py
│   └── statistics.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

# Future Improvements

The project should be implemented in a modular way so that future extensions can be easily added:

- Metric learning with Triplet Loss
- Supervised Contrastive Learning
- Text-to-Image Search
- CLIP-based Retrieval
- Personalized Recommendations
- Outfit Recommendation System
- Deployment to Streamlit Cloud
- User Feedback-Based Re-ranking

---

# Deliverables

The final project should include:

1. Fine-tuned ResNet50 model.
2. Feature extraction pipeline.
3. FAISS similarity search engine.
4. Interactive Streamlit application.
5. Embedding visualization.
6. Documentation and README.
7. Reproducible training and inference pipeline.
8. Clean, modular, production-quality code suitable for a portfolio project.
