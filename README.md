<p align="center">

<h1 align="center">Frame pack on vace-wan implementation</h1>



## Introduction
We implemented FramePack within the VACE-WAN 2.1 video generation pipeline to enable efficient long-form video synthesis. FramePack compresses redundant temporal information by selectively retaining key frames, reducing context length while preserving essential motion and scene details. Our integration adapts WAN’s transformer-based architecture to handle extended temporal spans without retraining, optimizes GPU memory usage, and maintains solver state continuity across overlapping sliding windows. This allows for generating longer, coherent videos on multi-GPU setups while preserving visual quality and temporal consistency.


## 🪄 Models
| Models                   | Download Link                                                                                                                                           | Video Size        | License                                                                                       |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------------------------------------------|
| VACE-Wan2.1-1.3B-Preview | [Huggingface](https://huggingface.co/ali-vilab/VACE-Wan2.1-1.3B-Preview) 🤗  [ModelScope](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview) 🤖 | ~ 81 x 480 x 832  | [Apache-2.0](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/LICENSE.txt)             |
| VACE-LTX-Video-0.9       | [Huggingface](https://huggingface.co/ali-vilab/VACE-LTX-Video-0.9) 🤗     [ModelScope](https://modelscope.cn/models/iic/VACE-LTX-Video-0.9) 🤖          | ~ 97 x 512 x 768  | [RAIL-M](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.license.txt) |
| Wan2.1-VACE-1.3B         | [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) 🤗     [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-VACE-1.3B) 🤖          | ~ 81 x 480 x 832  | [Apache-2.0](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/LICENSE.txt)             |
| Wan2.1-VACE-14B          | [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) 🤗     [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B) 🤖            | ~ 81 x 720 x 1280 | [Apache-2.0](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/blob/main/LICENSE.txt)             |

- The input supports any resolution, but to achieve optimal results, the video size should fall within a specific range.
- All models inherit the license of the original model.


## ⚙️ Installation
The codebase was tested with Python 3.10.13, CUDA version 12.4, and PyTorch >= 2.5.1.

### Setup for Model Inference
You can setup for VACE model inference by running:
```bash
git clone https://github.com/Long-form-AI-video-generation/framepack-vace-wan-experiment.git && cd VACE
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124  # If PyTorch is not installed.
pip install -r requirements.txt
pip install wan@git+https://github.com/Wan-Video/Wan2.1  
```
Please download your preferred base model to `<repo-root>/models/`. 

### Setup for Preprocess Tools
If you need preprocessing tools, please install:
```bash
pip install -r requirements/annotator.txt
```
Please download [VACE-Annotators](https://huggingface.co/ali-vilab/VACE-Annotators) to `<repo-root>/models/`.

### Local Directories Setup
It is recommended to download [VACE-Benchmark](https://huggingface.co/datasets/ali-vilab/VACE-Benchmark) to `<repo-root>/benchmarks/` as examples in `run_vace_xxx.sh`.

We recommend to organize local directories as:
```angular2html
VACE
├── ...
├── benchmarks
│   └── VACE-Benchmark
│       └── assets
│           └── examples
│               ├── animate_anything
│               │   └── ...
│               └── ...
├── models
│   ├── VACE-Annotators
│   │   └── ...
│   ├── VACE-LTX-Video-0.9
│   │   └── ...
│   └── VACE-Wan2.1-1.3B-Preview
│       └── ...
└── ...
```

## 🚀 Usage

#### Model inference
the model inference process can be performed as follows:
```bash
# For Wan2.1 single GPU inference (1.3B-480P)
python vace/framepack_vace_wan_inference.py --ckpt_dir <path-to-model>--src_ref_images <paths-to-src-ref-images> --prompt "xxx"

```
The output video together with intermediate video, mask and images will be saved into `./results/` by default.
