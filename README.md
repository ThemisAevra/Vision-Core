# ThemisAevra Vision Core

<div align="center">
  <img src="../assets/images/11KSGbIZoRSg4pjdnUoif6MKHI.svg" alt="ThemisAevra Logo" width="200"/>
  <br>
  <h3>Neural Perception for Autonomous Physical Agents</h3>
</div>

<br>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/research-beta-orange.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Paper](https://img.shields.io/badge/IROS-2025-blue.svg)]()

</div>

**ThemisAevra Vision Core** is the research repository for our state-of-the-art perception stack. It contains the model definitions, training pipelines, and inference optimization tools for the `ViT-H/14` and `Spatial-Net` architectures used in ThemisAevra agents.

## 👁️ Architecture: Spatial-Net

Spatial-Net treats 3D space not as voxels or point clouds, but as a continuous semantic field.

```mermaid
graph LR
    Input[RGB-D Stream] -->|Backbone (ResNet-50)| Feats[2D Features]
    Feats -->|Projection| BEV[Bird's Eye View]
    BEV -->|Temporal Fusion| Map[Local Map]
    Map -->|Decoder| Semantics[3D Occupancy Grid]
```

## 📊 Model Zoo & Benchmarks

| Model | Parameters | Latency (Orin NX) | mIoU (Industry-V2) | Links |
|-------|------------|-------------------|--------------------|-------|
| `themis-snet-tiny` | 12M | 4.2ms | 78.4% | [Config](./configs/tiny.yaml) |
| `themis-snet-base` | 48M | 12.8ms | 84.1% | [Config](./configs/base.yaml) |
| `themis-snet-large` | 120M | 34.1ms | 89.2% | [Config](./configs/large.yaml) |

> **Note:** Latency measured on Jetson Orin NX (15W Mode) with TensorRT FP16 optimization.

## 🚀 Usage

```python
import torch
from themisaevra_vision import SpatialNet

# Load pre-trained industrial model
model = SpatialNet.from_pretrained("themis-snet-base")

# Input: [Batch, Channel, Depth, Height, Width]
voxel_grid = torch.randn(1, 3, 32, 256, 256).cuda()

# Predict semantic occupancy
with torch.inference_mode():
    occupancy = model(voxel_grid)
```

## 🏗️ Repo Structure

- `themisaevra_vision/models`: PyTorch implementations of Spatial-Net.
- `themisaevra_vision/ops`: Custom CUDA kernels for sparse voxel convolution.
- `training/`: Distributed training recipes using FSDP.
- `benchmarks/`: TensorRT latency measurement scripts.

## 📜 Citation

If you use Spatial-Net in your research, please cite:

```bibtex
@inproceedings{themisaevra2025spatialnet,
  title={Spatial-Net: Real-Time 3D Semantic Occupancy for Industrial Agents},
  author={ThemisAevra Research Team},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025}
}
```

## 📜 License

Apache 2.0 - See [LICENSE](LICENSE) for details.
