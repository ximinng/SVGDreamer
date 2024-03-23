# SVGDreamer: Text Guided SVG Generation with Diffusion Model

[![cvpr24](https://img.shields.io/badge/CVPR-2024-387ADF.svg)](https://arxiv.org/abs/2312.16476)
[![arXiv](https://img.shields.io/badge/arXiv-2312.16476-b31b1b.svg)](https://arxiv.org/abs/2312.16476)
[![website](https://img.shields.io/badge/Website-Gitpage-4CCD99)](https://ximinng.github.io/SVGDreamer-project/)
[![blog](https://img.shields.io/badge/Blog-EN-9195F6)](https://huggingface.co/blog/xingxm/svgdreamer)
[![blog](https://img.shields.io/badge/Blog-CN-9195F6)](https://huggingface.co/blog/xingxm/svgdreamer)

This repository contains our official implementation of the CVPR 2024 paper: SVGDreamer: Text-Guided SVG Generation with
Diffusion Model. It can generate high-quality SVGs based on text prompts.

![title](./assets/illustrate.png)
![title](./assets/teaser_svg_asset.png)

## :new: Update

- [03/2024] 🔥 We have released the **code** for [SVGDreamer](https://ximinng.github.io/SVGDreamer-project/).
- [02/2024] 🎉 **SVGDreamer accepted by CVPR2024.** 🎉
- [12/2023] 🔥 We have released the **[SVGDreamer Paper](https://arxiv.org/abs/2312.16476)**. SVGDreamer is
  a novel text-guided vector graphics synthesis method. This method considers both the editing of vector graphics and
  the quality of the synthesis.

## Installation

You can follow the steps below to quickly get up and running with SVGDreamer.
These steps will let you run quick inference locally.

In the top level directory run,

```bash
sh script/install.sh
```

or using docker images,

```shell
docker run --name svgdreamer --gpus all -it --ipc=host ximingxing/svgrender:v1 /bin/bash
```

## 🔥 Quickstart

Before running the code, download the stable diffusion model. Append `diffuser.download=True` to the end of the script.

### SIVE + VPSD

**Prompt:** An image of Batman. full body action pose, complete detailed body. white background. empty background, high
quality, 4K, ultra realistic <br/>
**Script:**

```shell
python svgdreamer.py x=iconography skip_sive=False "prompt='an image of Batman. full body action pose, complete detailed body. white background. empty background, high quality, 4K, ultra realistic'" token_ind=4 x.vpsd.t_schedule='randint' result_path='./logs/batman' multirun=True
```

- `x=iconography`(str): style configs
- `skip_sive`(bool): enable the SIVE stage
- `token_ind`(int): the index of text prompt, from 1
- `result_path`(str):  the path to save the result
- `multirun`(bool): run the script multiple times with different random seeds
- `mv`(bool): save the intermediate results of the run and record the video (This increases the run time)

More parameters in `./conf/x/style.yaml`, you can modify these parameters from the command line. For example,
append `x.vpsd.n_particle=4` to the end of the script.

### SIVE

**Prompt:** an astronaut walking across a desert, planet mars in the background, floating beside planets, space art <br/>
**Preview:**

| attn-map                                       | bg init                                           | fg init                                           | bg final                                           | fg final                                           | final                                            |
|------------------------------------------------|---------------------------------------------------|---------------------------------------------------|----------------------------------------------------|----------------------------------------------------|--------------------------------------------------|
| <img src="./assets/SIVE-astronaut-1/attn.png"> | <img src="./assets/SIVE-astronaut-1/init_bg.svg"> | <img src="./assets/SIVE-astronaut-1/init_fg.svg"> | <img src="./assets/SIVE-astronaut-1/final_bg.svg"> | <img src="./assets/SIVE-astronaut-1/final_fg.svg"> | <img src="./assets/SIVE-astronaut-1/result.svg"> |


**Script:**

```shell
python svgdreamer.py x=iconography_s1 skip_sive=False "prompt='a man in an astronaut suit walking across a desert, inspired by James Gurney, space art, planet mars in the background, banner, floating beside planets'" token_ind=5 x.vpsd.t_schedule='randint' result_path='./logs/astronaut_sive' multirun=True
```

### VPSD

#### Iconography style

**Prompt:** Sydney opera house. oil painting. by Van Gogh <br/>
**Preview:**

| Particle 1                                             | Particle 2                                             | Particle 3                                             | Particle 4                                             | Particle 5                                             | Particle 6                                             |
|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|
| init p1                                                | init p2                                                | init p3                                                | init p4                                                | init p5                                                | init p6                                                |
| <img src="./assets/Icon-SydneyOperaHouse/init_p0.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p1.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p2.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p3.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p4.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p5.svg"> |
| final p1                                               | final p2                                               | final p3                                               | final p4                                               | final p5                                               | final p6                                               |
| <img src="./assets/Icon-SydneyOperaHouse/p_0.svg">     | <img src="./assets/Icon-SydneyOperaHouse/p_1.svg">     | <img src="./assets/Icon-SydneyOperaHouse/p_2.svg">     | <img src="assets/Icon-SydneyOperaHouse/p_3.svg">       | <img src="./assets/Icon-SydneyOperaHouse/p_4.svg">     | <img src="./assets/Icon-SydneyOperaHouse/p_5.svg">     |

**Script:**

```shell
python svgdreamer.py x=iconography "prompt='Sydney opera house. oil painting. by Van Gogh'" result_path='./logs/SydneyOperaHouse-OilPainting'
```

#### Painting style

**Prompt:** Abstract Vincent van Gogh Oil Painting Elephant, featuring earthy tones of green and brown <br/>
**Preview:**

| Particle 1                                         | Particle 2                                         | Particle 3                                         | Particle 4                                         | Particle 5                                         | Particle 6                                         |
|----------------------------------------------------|----------------------------------------------------|----------------------------------------------------|----------------------------------------------------|----------------------------------------------------|----------------------------------------------------|
| init p1                                            | init p2                                            | init p3                                            | init p4                                            | init p5                                            | init p6                                            |
| <img src="./assets/Painting-Elephant/init_p0.svg"> | <img src="./assets/Painting-Elephant/init_p1.svg"> | <img src="./assets/Painting-Elephant/init_p2.svg"> | <img src="./assets/Painting-Elephant/init_p3.svg"> | <img src="./assets/Painting-Elephant/init_p4.svg"> | <img src="./assets/Painting-Elephant/init_p5.svg"> |
| final p1                                           | final p2                                           | final p3                                           | final p4                                           | final p5                                           | final p6                                           |
| <img src="./assets/Painting-Elephant/p_0.svg">     | <img src="./assets/Painting-Elephant/p_1.svg">     | <img src="./assets/Painting-Elephant/p_2.svg">     | <img src="./assets/Painting-Elephant/p_3.svg">     | <img src="./assets/Painting-Elephant/p_4.svg">     | <img src="./assets/Painting-Elephant/p_5.svg">     |

**Script:**

```shell
python svgdreamer.py x=painting "prompt='Abstract Vincent van Gogh Oil Painting Elephant, featuring earthy tones of green and brown.'" x.num_paths=500 result_path='./logs/Elephant-OilPainting'
```

#### Pixel-Art style

**Prompt:** Darth vader with lightsaber <br/>
**Preview:**

| Particle 1                                           | Particle 2                                           | Particle 3                                           | Particle 4                                           | Particle 5                                           | Particle 6                                           |
|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
| init p1                                              | init p2                                              | init p3                                              | init p4                                              | init p5                                              | init p6                                              |
| <img src="./assets/Pixelart-DarthVader/init_p0.svg"> | <img src="./assets/Pixelart-DarthVader/init_p1.svg"> | <img src="./assets/Pixelart-DarthVader/init_p2.svg"> | <img src="./assets/Pixelart-DarthVader/init_p3.svg"> | <img src="./assets/Pixelart-DarthVader/init_p4.svg"> | <img src="./assets/Pixelart-DarthVader/init_p5.svg"> |
| final p1                                             | final p2                                             | final p3                                             | final p4                                             | final p5                                             | final p6                                             |
| <img src="./assets/Pixelart-DarthVader/p0.svg">      | <img src="./assets/Pixelart-DarthVader/p1.svg">      | <img src="./assets/Pixelart-DarthVader/p2.svg">      | <img src="./assets/Pixelart-DarthVader/p3.svg">      | <img src="./assets/Pixelart-DarthVader/p4.svg">      | <img src="./assets/Pixelart-DarthVader/p5.svg">      |

**Script:**

```shell
python svgdreamer.py x=pixelart "prompt='Darth vader with lightsaber.'" result_path='./logs/DarthVader'
```

#### Other Styles

```shell
# Style: low-ploy
python svgdreamer.py x=lowpoly "prompt='A picture of a bald eagle. low-ploy. polygon'" result_path='./logs/BaldEagle'
# Style: sketch
python svgdreamer.py x=sketch "prompt='A free-hand drawing of A speeding Lamborghini. black and white drawing.'" result_path='./logs/Lamborghini'
# Style: ink and wash
python svgdreamer.py x=ink "prompt='Big Wild Goose Pagoda. ink style. Minimalist abstract art grayscale watercolor.'" result_path='./logs/BigWildGoosePagoda'
# Style: painting
python svgdreamer.py x=painting "prompt='self portrait of Van Gogh. oil painting. cmyk portrait. multi colored. defiant and beautiful. cmyk. expressive eyes.'" result_path='./logs/VanGogh-Portrait'
```

## 🔑 Tips

- `x.vpsd.t_schedule` greatly affects the style of the result. Please try more.
- `neg_prompt` negative prompts affect the quality of the results.

## 📋 TODO

- [x] Release the code
- [x] Add docker image

## :books: Acknowledgement

The project is built based on the following repository:

- [BachiLi/diffvg](https://github.com/BachiLi/diffvg)
- [huggingface/diffusers](https://github.com/huggingface/diffusers)
- [ximinng/DiffSketcher](https://github.com/ximinng/DiffSketcher)
- [THUDM/ImageReward](https://github.com/THUDM/ImageReward)
- [ximinng//PyTorch-SVGRender](https://github.com/ximinng/PyTorch-SVGRender)

We gratefully thank the authors for their wonderful works.

## :paperclip: Citation

If you use this code for your research, please cite the following work:

```
@article{xing2023svgdreamer,
  title={SVGDreamer: Text Guided SVG Generation with Diffusion Model},
  author={Xing, Ximing and Zhou, Haitao and Wang, Chuang and Zhang, Jing and Xu, Dong and Yu, Qian},
  journal={arXiv preprint arXiv:2312.16476},
  year={2023}
}
```

## :copyright: Licence

This work is licensed under a MIT License.