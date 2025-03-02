# Vision Transformer (ViT) for MNIST Classification

This repository implements a Vision Transformer (ViT) architecture for classifying MNIST handwritten digits.

ViT applies the transformer architecture (originally designed for NLP) to image classification by:
1. Splitting images into patches
2. Flattening patches into tokens
3. Adding positional embeddings
4. Processing through transformer encoder blocks
5. Using the CLS token for final classification

![Vision Transformer Architecture](https://viso.ai/wp-content/uploads/2021/09/vision-transformer-vit.png)

## Implementation Details

- **Dataset**: MNIST (28×28 grayscale images of handwritten digits)
- **Patch Size**: 4×4 (dividing 28×28 images into 49 patches)
- **Model Parameters**:
  - Embedding Dimension: 16
  - Encoder Blocks: 4
  - Attention Heads: 4
  - Dropout: 0.001
  - Activation: GELU
- **Training**:
  - Optimizer: Adam
  - Learning Rate: 1e-4
  - Weight Decay: 0.1
  - Batch Size: 512
  - Epochs: 40

## Paper

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## Getting Started

```
# Install requirements
pip install torch torchvision

# Run training
python train.py
```

## Todo:
 - Implement a DiT model
 - Implement a scheduler
 - Add visualizations