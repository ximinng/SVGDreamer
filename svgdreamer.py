# -*- coding: utf-8 -*-
# Author: ximing xing
# Description: the main func of this project.
# Copyright (c) 2023, XiMing Xing.

import os
import sys
from functools import partial

from accelerate.utils import set_seed
import hydra
import omegaconf

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from svgdreamer.utils import render_batch_wrap, get_seed_range
from svgdreamer.pipelines.SVGDreamer_pipeline import SVGDreamerPipeline


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(cfg: omegaconf.DictConfig):
    """
    The project configuration is stored in './conf/config.yaml'
    And style configurations are stored in './conf/x/iconographic.yaml'
    """

    # set seed
    set_seed(cfg.seed)
    seed_range = get_seed_range(cfg.srange) if cfg.multirun else None

    # render function
    render_batch_fn = partial(render_batch_wrap, cfg=cfg, seed_range=seed_range)

    if not cfg.multirun:  # generate SVG multiple times
        pipe = SVGDreamerPipeline(cfg)
        pipe.painterly_rendering(cfg.prompt)
    else:  # generate many SVG at once
        render_batch_fn(pipeline=SVGDreamerPipeline, text_prompt=cfg.prompt, target_file=None)


if __name__ == '__main__':
    main()
