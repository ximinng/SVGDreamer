# SVGDreamer: Text Guided SVG Generation with Diffusion Model

[![cvpr24](https://img.shields.io/badge/CVPR-2024-387ADF.svg)](https://arxiv.org/abs/2312.16476)
[![arXiv](https://img.shields.io/badge/arXiv-2312.16476-b31b1b.svg)](https://arxiv.org/abs/2312.16476)
[![website](https://img.shields.io/badge/Website-Gitpage-4CCD99)](https://ximinng.github.io/SVGDreamer-project/)
[![blog](https://img.shields.io/badge/Blog-ENG-9195F6)](https://huggingface.co/blog/xingxm/svgdreamer)
[![blog](https://img.shields.io/badge/Blog-CN-9195F6)](https://huggingface.co/blog/xingxm/svgdreamer)

This repository contains our official implementation of the CVPR 2024 paper: SVGDreamer: Text-Guided SVG Generation with
Diffusion Model. It can generate high-quality SVGs based on text prompts.

[//]: # (> Project Page: https://ximinng.github.io/SVGDreamer-project/)

![title](./assets/illustrate.png)
![title](./assets/teaser_svg_asset.png)

## :new: Update

- [03/2024] 🔥 We have released the **code** for [SVGDreamer](https://ximinng.github.io/SVGDreamer-project/).
- [02/2024] 🎉 **SVGDreamer accepted by CVPR2024.** 🎉
- [12/2023] 🔥 We have released the **[SVGDreamer Paper](https://arxiv.org/abs/2312.16476)**. SVGDreamer is
  a novel text-guided vector graphics synthesis method. This method considers both the editing of vector graphics and
  the quality of the synthesis.

## 🔥Quickstart

Before running the code, download the stable diffusion model. Append `diffuser.download=True` to the end of the script.

### SIVE + VPSD

**Script:**

```shell
python svgdreamer.py x=iconography skip_sive=False "prompt='an image of Batman. full body action pose, complete detailed body. white background. empty background, high quality, 4K, ultra realistic'" token_ind=4 x.vpsd.t_schedule='randint' result_path='./logs/batman' multirun=True mv=True
```

- `x=iconography`(str): style configs
- `skip_sive`(bool): enable the SIVE stage
- `token_ind`(int): the index of text prompt, from 1
- `result_path`(str):  the path to save the result
- `multirun`(bool): run the script multiple times with different random seeds
- `mv`(bool): save the intermediate results of the run and record the video (This increases the run time)

**More parameters in `./conf/x/style.yaml`, you can modify these parameters from the command line. For
example, append `x.vpsd.n_particle=4` to the end of the script.**

### VPSD

**Prompt:** Sydney opera house. oil painting. by Van Gogh <br/>
**Style:** iconography <br/>
**Preview:**

| Particle 1                                             | Particle 2                                             | Particle 3                                             | Particle 4                                             | Particle 5                                             | Particle 6                                             |
|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|
| init p1                                                | init p2                                                | init p3                                                | init p4                                                | init p5                                                | init p6                                                |
| <img src="./assets/Icon-SydneyOperaHouse/init_p0.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p1.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p2.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p3.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p4.svg"> | <img src="./assets/Icon-SydneyOperaHouse/init_p5.svg"> |
| final p1                                               | final p2                                               | final p3                                               | final p4                                               | final p5                                               | final p6                                               |
| <img src="./assets/Icon-SydneyOperaHouse/p_0.svg">     | <img src="assets/Icon-SydneyOperaHouse/p_1.svg">       | <img src="assets/Icon-SydneyOperaHouse/p_2.svg">       | <img src="assets/Icon-SydneyOperaHouse/p_3.svg">       | <img src="assets/Icon-SydneyOperaHouse/p_4.svg">       | <img src="assets/Icon-SydneyOperaHouse/p_5.svg">       |

**Script:**

```shell
python svgdreamer.py x=iconography "prompt='Sydney opera house. oil painting. by Van Gogh'" result_path='./logs/SydneyOperaHouse-OilPainting' 
```

**Other Styles:**

```shell
# Style: low-ploy
python svgdreamer.py x=lowpoly "prompt='A picture of a bald eagle. low-ploy. polygon'" result_path='./logs/BaldEagle'
# Style: pixel-art
python svgdreamer.py x=pixelart "prompt='Darth vader with lightsaber.'" result_path='./log/DarthVader'
# Style: painting
python svgdreamer.py x=painting "prompt='self portrait of Van Gogh. oil painting. cmyk portrait. multi colored. defiant and beautiful. cmyk. expressive eyes.'" result_path='./logs/VanGogh-Portrait'
# Style: sketch
python svgdreamer.py x=sketch "prompt='A free-hand drawing of A speeding Lamborghini. black and white drawing.'" result_path='./logs/Lamborghini'
# Style: ink and wash
python svgdreamer.py x=ink "prompt='Big Wild Goose Pagoda. ink style. Minimalist abstract art grayscale watercolor.'" result_path='./logs/BigWildGoosePagoda'
```

## 🔑 Tips

- `x.vpsd.t_schedule` greatly affects the style of the result. Please try more.
- `neg_prompt` negative prompts affect the quality of the results.

## 📋 TODO

- [x] Release the code

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