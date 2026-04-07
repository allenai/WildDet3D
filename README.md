<div align="center">
  <img src="assets/wilddet3d_banner.svg" alt="WildDet3D" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>

# WildDet3D:<br> Scaling Promptable 3D Detection in the Wild

<a href="https://huggingface.co/allenai/WildDet3D" target="_blank">
    <img alt="HF Model: WildDet3D" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-WildDet3D-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/datasets/allenai/WildDet3D-Data" target="_blank">
    <img alt="HF Dataset: WildDet3D Data" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Data-WildDet3D--Data-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/spaces/allenai/WildDet3D" target="_blank">
    <img alt="HF Demo: WildDet3D" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Demo-WildDet3D-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/collections/allenai/wilddet3d-69d42220ec4f942f1951263e" target="_blank">
    <img alt="HF Collection" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Collection-WildDet3D-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://apps.apple.com/us/app/wilddet3d/id6760861157" target="_blank">
    <img alt="iPhone App" src="https://img.shields.io/badge/App_Store-WildDet3D-blue?logo=apple&logoColor=white" height="25" />
</a>
<a href="https://allenai.github.io/WildDet3D/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/Website-WildDet3D-blue" height="25" />
</a>
<a href="https://allenai.org/blog/wilddet3d" target="_blank">
    <img alt="Blog" src="https://img.shields.io/badge/Blog-WildDet3D-green" height="25" />
</a>

<p style="width:80%; margin:auto">
    <a href="https://weikaih04.github.io/" target="_blank">Weikai Huang</a><sup style="color:#F26035">&#9829;</sup><sup>1,2</sup>&nbsp;&nbsp;
    <a href="https://jieyuz2.github.io/" target="_blank">Jieyu Zhang</a><sup style="color:#F26035">&#9829;</sup><sup>1,2</sup>
    <br>
    <a href="https://github.com/Silicon23" target="_blank">Sijun Li</a><sup>2</sup>&nbsp;&nbsp;
    <a href="https://github.com/taoyangJ" target="_blank">Taoyang Jia</a><sup>2</sup>&nbsp;&nbsp;
    <a href="https://duanjiafei.com/" target="_blank">Jiafei Duan</a><sup>1,2</sup>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=Li5XdUsAAAAJ" target="_blank">Yunqian Cheng</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://j-min.io/" target="_blank">Jaemin Cho</a><sup>1,2</sup>&nbsp;&nbsp;
    <a href="https://mattwallingford.github.io/" target="_blank">Matthew Wallingford</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://github.com/RustinS" target="_blank">Rustin Soraki</a><sup>1,2</sup>&nbsp;&nbsp;
    <a href="https://cdjkim.github.io/" target="_blank">Chris Dongjoo Kim</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://shuoliu1024.github.io/" target="_blank">Shuo Liu</a><sup>1,2</sup>&nbsp;&nbsp;
    <a href="https://www.linkedin.com/in/donovanclay/" target="_blank">Donovan Clay</a><sup>1,2</sup>&nbsp;&nbsp;
    <a href="https://www.linkedin.com/in/taira-anderson/" target="_blank">Taira Anderson</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://winsonhan.com/" target="_blank">Winson Han</a><sup>1</sup>
    <br>
    <a href="https://homes.cs.washington.edu/~ali/" target="_blank">Ali Farhadi</a><sup>1,2</sup>&nbsp;&nbsp;
    <a href="https://www.cs.cornell.edu/~bharathh/" target="_blank">Bharath Hariharan</a><sup>3</sup>&nbsp;&nbsp;
    <a href="https://jason718.github.io/" target="_blank">Zhongzheng Ren</a><sup style="color:#F26035">&#9829;</sup><sup>1,2,4</sup>&nbsp;&nbsp;
    <a href="https://ranjaykrishna.com/" target="_blank">Ranjay Krishna</a><sup style="color:#F26035">&#9829;</sup><sup>1,2</sup>
</p>

<p>
<span style="color:#F26035">&#9829;</span> core contributors &nbsp;&nbsp;
<sup>1</sup>Allen Institute for AI&nbsp;&nbsp;&nbsp;&nbsp;<sup>2</sup>University of Washington&nbsp;&nbsp;&nbsp;&nbsp;<sup>3</sup>Cornell University&nbsp;&nbsp;&nbsp;&nbsp;<sup>4</sup>UNC-Chapel Hill
</p>

</div>

## Demo & Applications

<table>
<tr>
<td align="center">
<a href="demo/huggingface/README.md"><img src="assets/demo_huggingface.png" height="200"></a>
</td>
<td align="center">
<a href="demo/iphone/README.md"><img src="assets/demo_iphone.png" height="200"></a>
</td>
</tr>
<tr>
<td align="center">
<b>HuggingFace Interactive Demo</b>
<br>Interactive web demo with text, point, and box prompts
<br><a href="https://huggingface.co/spaces/allenai/WildDet3D">Live Demo</a> | <a href="demo/huggingface/README.md">Run Locally</a>
</td>
<td align="center">
<b>iPhone App</b>
<br>Real-time on-device 3D detection
<br><a href="https://apps.apple.com/us/app/wilddet3d/id6760861157">App Store</a> | <a href="demo/iphone/README.md">README</a>
</td>
</tr>
<tr><td colspan="2"></td></tr>
<tr>
<td align="center">
<a href="demo/vlm/README.md"><img src="assets/demo_vlm.png" height="200"></a>
</td>
<td align="center">
<a href="demo/tracking/README.md"><img src="assets/demo_tracking.gif" height="200"></a>
</td>
</tr>
<tr>
<td align="center">
<b>Integrate with VLM</b>
<br>Combine with vision-language models
<br><a href="demo/vlm/README.md">README</a>
</td>
<td align="center">
<b>Zero-Shot Tracking</b>
<br>3D object tracking without training
<br><a href="demo/tracking/README.md">README</a>
</td>
</tr>
<tr><td colspan="2"></td></tr>
<tr>
<td align="center">
<a href="demo/meta_quest/README.md"><img src="assets/demo_meta_quest.png" height="200"></a>
</td>
<td align="center">
<a href="demo/robotics/README.md"><img src="assets/demo_robotics.png" height="200"></a>
</td>
</tr>
<tr>
<td align="center">
<b>Meta Quest</b>
<br>3D detection in AR/VR
<br><a href="demo/meta_quest/README.md">README</a>
</td>
<td align="center">
<b>Robotics</b>
<br>3D perception for robotic manipulation
<br><a href="demo/robotics/README.md">README</a>
</td>
</tr>
</table>

## TODO
- [x] Release inference code
- [x] Release WildDet3D-Bench evaluation
- [ ] Release evaluation on other benchmarks (Omni3D, Argoverse2, ScanNet)
- [ ] Release training code

## Contents
- [Demo & Applications](#demo--applications)
- [Model Weights](#model-weights)
- [Installation](#installation)
- [Inference](#inference)
- [Evaluation](#evaluation)
  - [WildDet3D-Bench](#wilddet3d-bench)
  - [WildDet3D-Stereo4D-Bench](#wilddet3d-stereo4d-bench)
- [Results](#results)
  - [WildDet3D-Bench (In-the-Wild)](#wilddet3d-bench-in-the-wild)
  - [Omni3D](#omni3d)
- [WildDet3D Data](#wilddet3d-data)
- [Citation](#citation)

## Model Weights

| Model | Backbone | Depth Backend | Params | Download |
|---|---|---|---|---|
| WildDet3D | SAM3 ViT | LingBot-Depth (DINOv2 ViT-L/14) | ~1.2B | [allenai/WildDet3D](https://huggingface.co/allenai/WildDet3D) |

```bash
# Download checkpoint
pip install huggingface_hub
huggingface-cli download allenai/WildDet3D wilddet3d_alldata_all_prompt_v1.0.pt --local-dir ckpt/
```

## Installation

```bash
git clone --recurse-submodules https://github.com/allenai/WildDet3D.git
cd WildDet3D
conda create -n wilddet3d python=3.11 -y
conda activate wilddet3d

# Install all dependencies
pip install -r requirements.txt
```

## Inference

```python
from wilddet3d import build_model, preprocess
from wilddet3d.vis.visualize import draw_3d_boxes
import numpy as np
from PIL import Image

# Build model
model = build_model(
    checkpoint="ckpt/wilddet3d_alldata_all_prompt_v1.0.pt",
    score_threshold=0.3,
    skip_pretrained=True,
)

# Load and preprocess image
image = np.array(Image.open("image.jpg")).astype(np.float32)

# With known camera intrinsics
intrinsics = np.load("intrinsics.npy")  # (3, 3)
data = preprocess(image, intrinsics)

# Without intrinsics (uses default: focal=max(H,W), principal point at center)
# data = preprocess(image)

# Text prompt: detect all instances of given categories
results = model(
    images=data["images"].cuda(),
    intrinsics=data["intrinsics"].cuda()[None],
    input_hw=[data["input_hw"]],
    original_hw=[data["original_hw"]],
    padding=[data["padding"]],
    input_texts=["car", "person", "bicycle"],
)
boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = results

# Box prompt (geometric): lift a 2D box to 3D (one-to-one)
results = model(
    images=data["images"].cuda(),
    intrinsics=data["intrinsics"].cuda()[None],
    input_hw=[data["input_hw"]],
    original_hw=[data["original_hw"]],
    padding=[data["padding"]],
    input_boxes=[[100, 200, 300, 400]],  # pixel xyxy
    prompt_text="geometric",
)

# Exemplar prompt: use a 2D box as visual exemplar, find all similar objects (one-to-many)
results = model(
    images=data["images"].cuda(),
    intrinsics=data["intrinsics"].cuda()[None],
    input_hw=[data["input_hw"]],
    original_hw=[data["original_hw"]],
    padding=[data["padding"]],
    input_boxes=[[100, 200, 300, 400]],
    prompt_text="visual",
)

# Point prompt
results = model(
    images=data["images"].cuda(),
    intrinsics=data["intrinsics"].cuda()[None],
    input_hw=[data["input_hw"]],
    original_hw=[data["original_hw"]],
    padding=[data["padding"]],
    input_points=[[(150, 250, 1), (200, 300, 0)]],  # (x, y, label): 1=positive, 0=negative
    prompt_text="geometric",
)

# Visualize results
boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = results
draw_3d_boxes(
    image=image.astype(np.uint8),
    boxes3d=boxes3d[0],
    intrinsics=intrinsics,
    scores_2d=scores_2d[0],
    scores_3d=scores_3d[0],
    class_ids=class_ids[0],
    class_names=["car", "person", "bicycle"],
    save_path="output.png",
)
```

**Notes:**
- If `intrinsics` is not provided, a default intrinsic matrix is used (`focal=max(H,W)`, principal point at image center).
- Optional depth input: pass `depth_gt=depth_tensor` (shape `(B, 1, H, W)`, meters) for improved 3D localization with sparse/dense depth (e.g., LiDAR).

See **[docs/INFERENCE.md](docs/INFERENCE.md)** for the full API reference.

## Evaluation

### WildDet3D-Bench

Download the evaluation data from [allenai/WildDet3D-Data](https://huggingface.co/datasets/allenai/WildDet3D-Data). Evaluate using the [vis4d](https://github.com/SysCV/vis4d) framework:

```bash
# Text prompt
vis4d test --config configs/eval/in_the_wild/text.py --gpus 1 --ckpt ckpt/wilddet3d_alldata_all_prompt_v1.0.pt

# Text prompt + GT depth
vis4d test --config configs/eval/in_the_wild/text_with_depth.py --gpus 1 --ckpt ckpt/wilddet3d_alldata_all_prompt_v1.0.pt

# Box prompt (oracle)
vis4d test --config configs/eval/in_the_wild/box_prompt.py --gpus 1 --ckpt ckpt/wilddet3d_alldata_all_prompt_v1.0.pt

# Box prompt + GT depth
vis4d test --config configs/eval/in_the_wild/box_prompt_with_depth.py --gpus 1 --ckpt ckpt/wilddet3d_alldata_all_prompt_v1.0.pt
```

| Mode | Config |
|---|---|
| Text | `configs/eval/in_the_wild/text.py` |
| Text + Depth | `configs/eval/in_the_wild/text_with_depth.py` |
| Box Prompt | `configs/eval/in_the_wild/box_prompt.py` |
| Box Prompt + Depth | `configs/eval/in_the_wild/box_prompt_with_depth.py` |

### WildDet3D-Stereo4D-Bench

Download the evaluation data from [allenai/WildDet3D-Stereo4D-Bench-Images](https://huggingface.co/datasets/allenai/WildDet3D-Stereo4D-Bench-Images). Evaluate (383 images with real stereo depth):

```bash
# Text prompt
vis4d test --config configs/eval/stereo4d/text.py --gpus 1 --ckpt ckpt/wilddet3d_alldata_all_prompt_v1.0.pt

# Text prompt + GT depth
vis4d test --config configs/eval/stereo4d/text_with_depth.py --gpus 1 --ckpt ckpt/wilddet3d_alldata_all_prompt_v1.0.pt

# Box prompt (oracle)
vis4d test --config configs/eval/stereo4d/box_prompt.py --gpus 1 --ckpt ckpt/wilddet3d_alldata_all_prompt_v1.0.pt

# Box prompt + GT depth
vis4d test --config configs/eval/stereo4d/box_prompt_with_depth.py --gpus 1 --ckpt ckpt/wilddet3d_alldata_all_prompt_v1.0.pt
```

| Mode | Config |
|---|---|
| Text | `configs/eval/stereo4d/text.py` |
| Text + Depth | `configs/eval/stereo4d/text_with_depth.py` |
| Box Prompt | `configs/eval/stereo4d/box_prompt.py` |
| Box Prompt + Depth | `configs/eval/stereo4d/box_prompt_with_depth.py` |

## Results

### WildDet3D-Bench (In-the-Wild)

AP is computed using center-distance matching. AP_r, AP_c, AP_f denote rare (<5), common (5-20), and frequent (>20) category splits.

| Method | Data | AP_r | AP_c | AP_f | AP |
|---|---|---|---|---|---|
| **Text Prompt** | | | | | |
| 3D-MOOD | Omni3D | 2.4 | 2.1 | 2.6 | 2.3 |
| WildDet3D | Omni3D | 9.0 | 6.5 | 5.2 | 6.8 |
| WildDet3D w/ depth | Omni3D | 23.0 | 21.5 | 16.1 | 20.7 |
| WildDet3D | Omni3D, Others, WildDet3D-Data | 28.3 | 21.6 | 18.7 | 22.6 |
| WildDet3D w/ depth | Omni3D, Others, WildDet3D-Data | **47.4** | **40.7** | **37.2** | **41.6** |
| **Box Prompt** | | | | | |
| OVMono3D-LIFT | Omni3D | 7.4 | 8.8 | 5.1 | 7.7 |
| DetAny3D | Omni3D, Others | 9.9 | 7.4 | 6.3 | 7.8 |
| WildDet3D | Omni3D | 12.0 | 7.9 | 5.3 | 8.4 |
| WildDet3D w/ depth | Omni3D | 26.4 | 24.4 | 19.6 | 23.9 |
| WildDet3D | Omni3D, Others, WildDet3D-Data | 30.0 | 24.2 | 20.3 | 24.8 |
| WildDet3D w/ depth | Omni3D, Others, WildDet3D-Data | **53.7** | **46.1** | **42.5** | **47.2** |

### Omni3D

AP is computed at 3D IoU [0.5:0.95].

| Method | KITTI | nuScenes | SUNRGBD | Hypersim | ARKitScenes | Objectron | AP |
|---|---|---|---|---|---|---|---|
| **Text Prompt** | | | | | | | |
| Cube R-CNN | 32.6 | 30.1 | 15.3 | 7.5 | 41.7 | 50.8 | 23.3 |
| 3D-MOOD Swin-T | 32.8 | 31.5 | 21.9 | 10.5 | 51.0 | 64.3 | 28.4 |
| 3D-MOOD Swin-B | 31.4 | 35.8 | 23.8 | 9.1 | 53.9 | 67.9 | 30.0 |
| WildDet3D | **37.0** | 31.7 | 38.9 | 16.5 | 64.6 | 60.5 | 34.2 |
| WildDet3D w/ depth | 36.1 | 32.0 | **51.1** | **26.6** | **73.3** | **68.3** | **41.6** |
| **Box Prompt** | | | | | | | |
| OVMono3D-LIFT | 31.4 | 32.5 | 23.2 | 11.9 | 54.2 | 63.5 | 29.6 |
| DetAny3D | 38.7 | **37.6** | 46.1 | 16.0 | 50.6 | 56.8 | 34.4 |
| WildDet3D | **44.3** | 35.3 | 43.1 | 17.3 | 66.6 | 60.8 | 36.4 |
| WildDet3D w/ depth | 42.8 | 35.9 | **58.7** | **30.4** | **76.6** | **68.5** | **45.8** |

### Qualitative Results

**Box prompt comparison** (WildDet3D vs OVMono3D vs DetAny3D):
<p align="center">
  <img src="assets/qualitative_box_prompt.jpg" width="100%" />
</p>

**Text prompt comparison**:
<p align="center">
  <img src="assets/qualitative_text_prompt.jpg" width="100%" />
</p>

## WildDet3D Data

We introduce **WildDet3D-Data**, a large-scale in-the-wild dataset for monocular 3D detection with human-verified 3D bounding box annotations. The dataset covers images from COCO, LVIS, Objects365, and V3Det.

| Split | Description | Images | Annotations | Categories |
|---|---|---|---|---|
| Val | Validation set (human) | 2,470 | 9,256 | 785 |
| Test | Test set (human) | 2,433 | 5,596 | 633 |
| Train (Human) | Human-reviewed only | 102,979 | 229,934 | 11,879 |
| Train (Essential) | Human + VLM-qualified small objects | 102,979 | 412,711 | 12,064 |
| Train (Synthetic) | VLM auto-selected | 896,004 | 3,483,292 | 11,896 |
| **Total** | | **1,003,886** | **3,910,855** | **13,499** |

The dataset is hosted on HuggingFace: [allenai/WildDet3D-Data](https://huggingface.co/datasets/allenai/WildDet3D-Data). See the [dataset README](https://huggingface.co/datasets/allenai/WildDet3D-Data) for download instructions and data format.

## Citation

If you find this work useful, please cite:

```bibtex
@article{wilddet3d2026,
    title={WildDet3D: Scaling Promptable 3D Detection in the Wild},
    author={Huang, Weikai and Zhang, Jieyu and Li, Sijun and Jia, Taoyang and Duan, Jiafei and Cheng, Yunqian and Cho, Jaemin and Wallingford, Matthew and Soraki, Rustin and Kim, Chris Dongjoo and Liu, Shuo and Clay, Donovan and Anderson, Taira and Han, Winson and Farhadi, Ali and Hariharan, Bharath and Ren, Zhongzheng and Krishna, Ranjay},
    year={2026},
}
```

## Acknowledgements

- [Omni3D](https://github.com/facebookresearch/omni3d) -- 3D detection benchmarks and baselines
- [vis4d](https://github.com/SysCV/vis4d) -- Training and evaluation framework
- [SAM 3](https://github.com/facebookresearch/sam3) -- Segment Anything Model 3
- [LingBot-Depth](https://github.com/Robbyant/lingbot-depth) -- Monocular depth estimation
- [3D-MOOD](https://github.com/cvg/3D-MOOD) -- Open-vocabulary monocular 3D detection
- [DetAny3D](https://github.com/OpenDriveLab/DetAny3D) -- Detect anything in 3D
- [OVMono3D](https://github.com/UVA-Computer-Vision-Lab/ovmono3d) -- Open-vocabulary monocular 3D detection
- [LabelAny3D](https://github.com/UVA-Computer-Vision-Lab/LabelAny3D) -- 3D bounding box annotation tool

## License

**Codebase:** This codebase incorporates code from [SAM 3](https://github.com/facebookresearch/sam3), and is licensed under the [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE). It is intended for research and educational use in accordance with [Ai2's Responsible Use Guidelines](https://allenai.org/responsible-use).

**Model:** This model is based on [SAM 3](https://github.com/facebookresearch/sam3) and [LingBot-Depth](https://github.com/Robbyant/lingbot-depth), and is licensed under the [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE). This model is intended for research and educational use in accordance with [Ai2's Responsible Use Guidelines](https://allenai.org/responsible-use).
