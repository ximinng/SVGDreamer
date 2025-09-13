# SVGDreamer: Text Guided SVG Generation with Diffusion Model

<div align="center" style="line-height: 1;">

[![CVPR 2024](https://img.shields.io/badge/CVPR%202024-Paper-4169E1?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2312.16476) [![arXiv](https://img.shields.io/badge/arXiv-2312.16476-8A2BE2?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2312.16476) [![Project Website](https://img.shields.io/badge/Website-Project%20Page-4682B4?style=for-the-badge&logo=github&logoColor=white)](https://ximinng.github.io/SVGDreamer-project/) [![English Blog](https://img.shields.io/badge/Blog-English-00CED1?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/blog/xingxm/svgdreamer) [![中文博客](https://img.shields.io/badge/博客-中文-1E90FF?style=for-the-badge&logo=zhihu&logoColor=white)](https://zhuanlan.zhihu.com/p/687525994)

</div>

This repository contains the official implementation of our CVPR 2024 paper, "SVGDreamer: Text-Guided SVG Generation
with Diffusion Model." The method leverages a diffusion-based approach to produce high-quality SVGs guided by text
prompts.

![title](./assets/illustrate.png)
![title](./assets/teaser_svg_asset.png)

## :new: Latest Update

- [11/2024] 🔥 **We released the [SVGDreamer++](https://arxiv.org/abs/2411.17832), offering stronger visual
  representation and improved editing capabilities.**
- [03/2024] 🔥 We released the **code** for [SVGDreamer](https://ximinng.github.io/SVGDreamer-project/).
- [02/2024] 🎉 SVGDreamer accepted by CVPR2024. 🎉
- [12/2023] 🔥 We released the **[SVGDreamer Paper](https://arxiv.org/abs/2312.16476)**. SVGDreamer is
  a novel text-guided vector graphics synthesis method. This method considers both the editing of vector graphics and
  the quality of the synthesis.

## 📌 Installation Guide

### 🛠️ Step 1: Set Up the Environment

To quickly get started with **SVGDreamer**, follow the steps below.  
These instructions will help you run **quick inference locally**.

#### 🚀 **Option 1: Standard Installation**

Run the following command in the **top-level directory**:

```shell
chmod +x script/install.sh
bash script/install.sh
```

#### 🐳 Option 2: Using Docker

```shell
chmod +x script/run_svgdreamer_docker.sh
sudo bash script/run_svgdreamer_docker.sh
```

### 🛠️ Step 2: Download Pretrained Stable Diffusion Model

SVGDreamer requires a pretrained Stable Diffusion (SD) model.
You can download it automatically or manually.

#### 🔄 Option 1: Auto-Download (Recommended)

Set `diffuser.download=True` in `/conf/config.yaml` before running SVGDreamer.
Alternatively, append `diffuser.download=True` to the execution script.

#### ⬇️ Option 2: Manual Download

If you prefer manual setup, download the model from Hugging Face:

🔗 Model Link: [Stable Diffusion 2.1 Base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)

The model will be stored at:

📁 Default Path: `~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-base`

## 🔥 Quickstart: synthesize **6** SVGs at once

### SIVE + VPSD

**Prompt:** an image of Batman. full body action pose, complete detailed body, white background, high quality, 4K, ultra
realistic <br/>
**Preview:**

|                  Particle 1                   |                  Particle 2                   |                  Particle 3                   |                  Particle 4                  |                  Particle 5                   |                  Particle 6                   |
|:---------------------------------------------:|:---------------------------------------------:|:---------------------------------------------:|:--------------------------------------------:|:---------------------------------------------:|:---------------------------------------------:|
|                    init p1                    |                    init p2                    |                    init p3                    |                   init p4                    |                    init p5                    |                    init p6                    |
| <img src="./assets/case-batman/init_p0.svg">  | <img src="./assets/case-batman/init_p1.svg">  | <img src="./assets/case-batman/init_p2.svg">  | <img src="./assets/case-batman/init_p3.svg"> | <img src="./assets/case-batman/init_p4.svg">  | <img src="./assets/case-batman/init_p5.svg">  |
|                   final p1                    |                   final p2                    |                   final p3                    |                   final p4                   |                   final p5                    |                   final p6                    |
| <img src="./assets/case-batman/final_p0.svg"> | <img src="./assets/case-batman/final_p1.svg"> | <img src="./assets/case-batman/final_p2.svg"> | <img src="assets/case-batman/final_p3.svg">  | <img src="./assets/case-batman/final_p4.svg"> | <img src="./assets/case-batman/final_p5.svg"> |

**Script:**

```shell
python svgdreamer.py x=iconography skip_sive=False "prompt='an image of Batman. full body action pose, complete detailed body. white background. empty background, high quality, 4K, ultra realistic'" token_ind=4 x.vpsd.t_schedule='randint' result_path='./logs/batman' multirun=True
```

🔹Parameter:

- `x=iconography`(str): style configs
- `skip_sive`(bool): enable the SIVE stage
- `token_ind`(int): the index of text prompt, from 1
- `result_path`(str):  the path to save the result
- `multirun`(bool): run the script multiple times with different random seeds
- `mv`(bool): save the intermediate results of the run and record the video (This increases the run time)

More parameters in `./conf/x/style.yaml`, you can modify these parameters from the command line. For example,
append `x.vpsd.n_particle=4` to the end of the script.

### SIVE

**Prompt:** an astronaut walking across a desert, planet mars in the background, floating beside planets, space
art <br/>
**Preview:**

|                    attn-map                    |                      bg init                      |                      fg init                      |                      bg final                      |                      fg final                      |                      final                       |
|:----------------------------------------------:|:-------------------------------------------------:|:-------------------------------------------------:|:--------------------------------------------------:|:--------------------------------------------------:|:------------------------------------------------:|
| <img src="./assets/SIVE-astronaut-1/attn.png"> | <img src="./assets/SIVE-astronaut-1/init_bg.svg"> | <img src="./assets/SIVE-astronaut-1/init_fg.svg"> | <img src="./assets/SIVE-astronaut-1/bg_final.svg"> | <img src="./assets/SIVE-astronaut-1/fg_final.svg"> | <img src="./assets/SIVE-astronaut-1/result.svg"> |

**Script:**

```shell
python svgdreamer.py x=iconography-s1 skip_sive=False "prompt='an astronaut walking across a desert, planet mars in the background, floating beside planets, space art'" token_ind=5 result_path='./logs/astronaut_sive' seed=116740
```

### VPSD

#### ✍️ Iconography style

**Prompt:** Sydney opera house. oil painting. by Van Gogh <br/>
**Preview:**

|                       Particle 1                       |                       Particle 2                       |                       Particle 3                       |                       Particle 4                       |                       Particle 5                       |                       Particle 6                       |
|:------------------------------------------------------:|:------------------------------------------------------:|:------------------------------------------------------:|:------------------------------------------------------:|:------------------------------------------------------:|:------------------------------------------------------:|
|                        init p1                         |                        init p2                         |                        init p3                         |                        init p4                         |                        init p5                         |                        init p6                         |
| <img src="./assets/Icon-SydneyOperaHouse/init_p0.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p1.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p2.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p3.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p4.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p5.svg"> |
|                        final p1                        |                        final p2                        |                        final p3                        |                        final p4                        |                        final p5                        |                        final p6                        |
|   <img src="./assets/Icon-SydneyOperaHouse/p_0.svg">   |   <img src="./assets/Icon-SydneyOperaHouse/p_1.svg">   |   <img src="./assets/Icon-SydneyOperaHouse/p_2.svg">   |    <img src="assets/Icon-SydneyOperaHouse/p_3.svg">    |   <img src="./assets/Icon-SydneyOperaHouse/p_4.svg">   |   <img src="./assets/Icon-SydneyOperaHouse/p_5.svg">   |

**Script:**

```shell
python svgdreamer.py x=iconography "prompt='Sydney opera house. oil painting. by Van Gogh'" result_path='./logs/SydneyOperaHouse-OilPainting' state.mprec='fp16'
```

#### ✍️ Painting style

**Prompt:** Abstract Vincent van Gogh Oil Painting Elephant, featuring earthy tones of green and brown <br/>
**Preview:**

|                     Particle 1                     |                     Particle 2                     |                     Particle 3                     |                     Particle 4                     |                     Particle 5                     |                     Particle 6                     |
|:--------------------------------------------------:|:--------------------------------------------------:|:--------------------------------------------------:|:--------------------------------------------------:|:--------------------------------------------------:|:--------------------------------------------------:|
|                      init p1                       |                      init p2                       |                      init p3                       |                      init p4                       |                      init p5                       |                      init p6                       |
| <img src="./assets/Painting-Elephant/init_p0.svg"> | <img src="./assets/Painting-Elephant/init_p1.svg"> | <img src="./assets/Painting-Elephant/init_p2.svg"> | <img src="./assets/Painting-Elephant/init_p3.svg"> | <img src="./assets/Painting-Elephant/init_p4.svg"> | <img src="./assets/Painting-Elephant/init_p5.svg"> |
|                      final p1                      |                      final p2                      |                      final p3                      |                      final p4                      |                      final p5                      |                      final p6                      |
|   <img src="./assets/Painting-Elephant/p_0.svg">   |   <img src="./assets/Painting-Elephant/p_1.svg">   |   <img src="./assets/Painting-Elephant/p_2.svg">   |   <img src="./assets/Painting-Elephant/p_3.svg">   |   <img src="./assets/Painting-Elephant/p_4.svg">   |   <img src="./assets/Painting-Elephant/p_5.svg">   |

**Script:**

```shell
python svgdreamer.py x=painting "prompt='Abstract Vincent van Gogh Oil Painting Elephant, featuring earthy tones of green and brown.'" x.num_paths=256 result_path='./logs/Elephant-OilPainting'
```

#### ✍️ Pixel-Art style

**Prompt:** Darth vader with lightsaber <br/>
**Preview:**

|                      Particle 1                      |                      Particle 2                      |                      Particle 3                      |                      Particle 4                      |                      Particle 5                      |                      Particle 6                      |
|:----------------------------------------------------:|:----------------------------------------------------:|:----------------------------------------------------:|:----------------------------------------------------:|:----------------------------------------------------:|:----------------------------------------------------:|
|                       init p1                        |                       init p2                        |                       init p3                        |                       init p4                        |                       init p5                        |                       init p6                        |
| <img src="./assets/Pixelart-DarthVader/init_p0.svg"> | <img src="./assets/Pixelart-DarthVader/init_p1.svg"> | <img src="./assets/Pixelart-DarthVader/init_p2.svg"> | <img src="./assets/Pixelart-DarthVader/init_p3.svg"> | <img src="./assets/Pixelart-DarthVader/init_p4.svg"> | <img src="./assets/Pixelart-DarthVader/init_p5.svg"> |
|                       final p1                       |                       final p2                       |                       final p3                       |                       final p4                       |                       final p5                       |                       final p6                       |
|   <img src="./assets/Pixelart-DarthVader/p0.svg">    |   <img src="./assets/Pixelart-DarthVader/p1.svg">    |   <img src="./assets/Pixelart-DarthVader/p2.svg">    |   <img src="./assets/Pixelart-DarthVader/p3.svg">    |   <img src="./assets/Pixelart-DarthVader/p4.svg">    |   <img src="./assets/Pixelart-DarthVader/p5.svg">    |

**Script:**

```shell
python svgdreamer.py x=pixelart "prompt='Darth vader with lightsaber.'" result_path='./logs/DarthVader'
```

#### ✍️Low-poly style

**Prompt:** A picture of a bald eagle. low-ploy. polygon. minimal flat 2d vector <br/>
**Preview:**

|                     Particle 1                     |                     Particle 2                     |                     Particle 3                     |                     Particle 4                     |                     Particle 5                     |                     Particle 6                     |
|:--------------------------------------------------:|:--------------------------------------------------:|:--------------------------------------------------:|:--------------------------------------------------:|:--------------------------------------------------:|:--------------------------------------------------:|
|                      init p1                       |                      init p2                       |                      init p3                       |                      init p4                       |                      init p5                       |                      init p6                       |
| <img src="./assets/Lowpoly-BaldEagle/init_p0.svg"> | <img src="./assets/Lowpoly-BaldEagle/init_p1.svg"> | <img src="./assets/Lowpoly-BaldEagle/init_p2.svg"> | <img src="./assets/Lowpoly-BaldEagle/init_p3.svg"> | <img src="./assets/Lowpoly-BaldEagle/init_p4.svg"> | <img src="./assets/Lowpoly-BaldEagle/init_p5.svg"> |
|                      final p1                      |                      final p2                      |                      final p3                      |                      final p4                      |                      final p5                      |                      final p6                      |
|   <img src="./assets/Lowpoly-BaldEagle/p0.svg">    |   <img src="./assets/Lowpoly-BaldEagle/p1.svg">    |   <img src="./assets/Lowpoly-BaldEagle/p2.svg">    |   <img src="./assets/Lowpoly-BaldEagle/p3.svg">    |   <img src="./assets/Lowpoly-BaldEagle/p4.svg">    |   <img src="./assets/Lowpoly-BaldEagle/p5.svg">    |

**Script:**

```shell
python svgdreamer.py x=lowpoly "prompt='A picture of a bald eagle. low-ploy. polygon. minimal flat 2d vector'" neg_prompt='' result_path='./logs/BaldEagle'
```

#### ✍️ Sketch style

**Prompt:** A free-hand drawing of A speeding Lamborghini. black and white drawing. <br/>
**Preview:**

|                     Particle 1                      |                     Particle 2                      |                     Particle 3                      |                     Particle 4                      |                     Particle 5                      |                     Particle 6                      |
|:---------------------------------------------------:|:---------------------------------------------------:|:---------------------------------------------------:|:---------------------------------------------------:|:---------------------------------------------------:|:---------------------------------------------------:|
|                       init p1                       |                       init p2                       |                       init p3                       |                       init p4                       |                       init p5                       |                       init p6                       |
| <img src="./assets/Sketch-Lamborghini/init_p0.svg"> | <img src="./assets/Sketch-Lamborghini/init_p1.svg"> | <img src="./assets/Sketch-Lamborghini/init_p2.svg"> | <img src="./assets/Sketch-Lamborghini/init_p3.svg"> | <img src="./assets/Sketch-Lamborghini/init_p4.svg"> | <img src="./assets/Sketch-Lamborghini/init_p5.svg"> |
|                      final p1                       |                      final p2                       |                      final p3                       |                      final p4                       |                      final p5                       |                      final p6                       |
|   <img src="./assets/Sketch-Lamborghini/p0.svg">    |   <img src="./assets/Sketch-Lamborghini/p1.svg">    |   <img src="./assets/Sketch-Lamborghini/p2.svg">    |   <img src="./assets/Sketch-Lamborghini/p3.svg">    |   <img src="./assets/Sketch-Lamborghini/p4.svg">    |   <img src="./assets/Sketch-Lamborghini/p5.svg">    |

**Script:**

```shell
python svgdreamer.py x=sketch "prompt='A free-hand drawing of A speeding Lamborghini. black and white drawing.'" neg_prompt='' result_path='./logs/Lamborghini'
```

#### ✍️ Ink and Wash style

**Prompt:** Big Wild Goose Pagoda. ink style. Minimalist abstract art grayscale watercolor. empty background <br/>
**Preview:**

|                       Particle 1                        |                       Particle 2                        |                       Particle 3                        |                       Particle 4                        |                       Particle 5                        |                       Particle 6                        |
|:-------------------------------------------------------:|:-------------------------------------------------------:|:-------------------------------------------------------:|:-------------------------------------------------------:|:-------------------------------------------------------:|:-------------------------------------------------------:|
|                         init p1                         |                         init p2                         |                         init p3                         |                         init p4                         |                         init p5                         |                         init p6                         |
| <img src="./assets/Ink-BigWildGoosePagoda/init_p0.svg"> | <img src="./assets/Ink-BigWildGoosePagoda/init_p1.svg"> | <img src="./assets/Ink-BigWildGoosePagoda/init_p2.svg"> | <img src="./assets/Ink-BigWildGoosePagoda/init_p3.svg"> | <img src="./assets/Ink-BigWildGoosePagoda/init_p4.svg"> | <img src="./assets/Ink-BigWildGoosePagoda/init_p5.svg"> |
|                        final p1                         |                        final p2                         |                        final p3                         |                        final p4                         |                        final p5                         |                        final p6                         |
|   <img src="./assets/Ink-BigWildGoosePagoda/p0.svg">    |   <img src="./assets/Ink-BigWildGoosePagoda/p1.svg">    |   <img src="./assets/Ink-BigWildGoosePagoda/p2.svg">    |   <img src="./assets/Ink-BigWildGoosePagoda/p3.svg">    |   <img src="./assets/Ink-BigWildGoosePagoda/p4.svg">    |   <img src="./assets/Ink-BigWildGoosePagoda/p5.svg">    |

**Script:**

```shell
python svgdreamer.py x=ink "prompt='Big Wild Goose Pagoda. ink style. Minimalist abstract art grayscale watercolor. empty background'" neg_prompt='' result_path='./logs/BigWildGoosePagoda'
```

#### 🎨 Supported Styles

**For more examples, visit [Examples.md](https://github.com/ximinng/SVGDreamer/blob/main/Examples.md)**.

## 🔑 Tips for Best Results

- I highly recommend turning on xformer `enable_xformers=True` to speed up optimization.
- `x.vpsd.t_schedule` greatly affects the style of the result. Please try more.
- `neg_prompt` negative prompts affect the quality of the results
- By setting `state.mprec='fp16'`, you can significantly reduce GPU memory usage.

## 📋 TODO

- [x] Release the code.
- [x] Add docker image.
- [x] Support fp16 optimization.

## :books: Acknowledgement

The project is built based on the following repository:

- [BachiLi/diffvg](https://github.com/BachiLi/diffvg)
- [huggingface/diffusers](https://github.com/huggingface/diffusers)
- [ximinng/DiffSketcher](https://github.com/ximinng/DiffSketcher)
- [THUDM/ImageReward](https://github.com/THUDM/ImageReward)
- [ximinng/PyTorch-SVGRender](https://github.com/ximinng/PyTorch-SVGRender)

We gratefully thank the authors for their wonderful works.

## :paperclip: Citation

If you use this code for your research, please cite the following work:

```
@InProceedings{xing2024svgdreamer,
    author    = {Xing, Ximing and Zhou, Haitao and Wang, Chuang and Zhang, Jing and Xu, Dong and Yu, Qian},
    title     = {SVGDreamer: Text Guided SVG Generation with Diffusion Model},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {4546-4555}
}
```

## :copyright: Licence

This work is licensed under a MIT License.