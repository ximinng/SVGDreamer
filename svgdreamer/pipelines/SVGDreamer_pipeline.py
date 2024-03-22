# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import os
import pathlib
from PIL import Image
from typing import AnyStr, Union, Tuple, List

import cv2
import cairosvg
import omegaconf
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torchvision
from torchvision import transforms
from skimage.color import rgb2gray

from svgdreamer.libs import ModelState, get_optimizer
from svgdreamer.painter import (CompPainter, CompPainterOptimizer, xing_loss_fn, Painter, PainterOptimizer,
                                CosineWithWarmupLRLambda, VectorizedParticleSDSPipeline, DiffusionPipeline)
from svgdreamer.token2attn.attn_control import EmptyControl, AttentionStore
from svgdreamer.token2attn.ptp_utils import view_images
from svgdreamer.utils.plot import plot_img, plot_couple, plot_attn, save_image
from svgdreamer.utils import init_tensor_with_color, AnyPath, mkdir
from svgdreamer.svgtools import merge_svg_files, is_valid_svg
from svgdreamer.diffusers_warp import model2res

import ImageReward as RM


class SVGDreamerPipeline(ModelState):

    def __init__(self, args):
        assert args.x.style in ["iconography", "pixelart", "low-poly", "painting", "sketch", "ink"]
        assert args.x.vpsd.n_particle >= args.x.vpsd.vsd_n_particle
        assert args.x.vpsd.n_particle >= args.x.vpsd.phi_n_particle
        assert args.x.vpsd.n_phi_sample >= 1

        logdir_ = f"sd{args.seed}" \
                  f"-{'vpsd' if args.skip_sive else 'sive'}" \
                  f"-{args.x.style}" \
                  f"-P{args.x.num_paths}" \
                  f"{'-RePath' if args.x.path_reinit.use else ''}"
        super().__init__(args, log_path_suffix=logdir_)

        """SIVE log dirs"""
        self.sive_attn_dir = self.result_path / "SIVE_attn_logs"
        self.mask_dir = self.result_path / "SIVE_mask_logs"
        self.sive_init_dir = self.result_path / "SIVE_init_logs"
        self.sive_final_dir = self.result_path / "SIVE_final_logs"
        # fg dir
        self.fg_png_logs_dir = self.result_path / "SIVE_fg_png_logs"
        self.fg_svg_logs_dir = self.result_path / "SIVE_fg_svg_logs"
        # bg dir
        self.bg_png_logs_dir = self.result_path / "SIVE_bg_png_logs"
        self.bg_svg_logs_dir = self.result_path / "SIVE_bg_svg_logs"
        # refine
        self.refine_dir = self.result_path / "SIVE_refine_logs"
        self.refine_png_dir = self.result_path / "SIVE_refine_png_logs"
        self.refine_svg_dir = self.result_path / "SIVE_refine_svg_logs"
        """VPSD log dirs"""
        self.ft_png_logs_dir = self.result_path / "VPSD_png_logs"
        self.ft_svg_logs_dir = self.result_path / "VPSD_svg_logs"
        self.reinit_dir = self.result_path / "VPSD_reinit_logs"
        self.ft_init_dir = self.result_path / "VPSD_init_logs"
        self.phi_samples_dir = self.result_path / "VPSD_phi_sampling_logs"

        mkdir([self.sive_attn_dir, self.mask_dir, self.fg_png_logs_dir, self.fg_svg_logs_dir,
               self.sive_final_dir, self.refine_dir, self.refine_png_dir, self.refine_svg_dir,
               self.bg_png_logs_dir, self.bg_svg_logs_dir, self.sive_init_dir, self.reinit_dir,
               self.ft_init_dir, self.phi_samples_dir, self.ft_png_logs_dir, self.ft_svg_logs_dir])

        # make video log
        self.make_video = self.args.mv
        if self.make_video:
            self.frame_idx = 0
            self.frame_log_dir = self.result_path / "frame_logs"
            self.frame_log_dir.mkdir(parents=True, exist_ok=True)

        # torch Generator seed
        self.g_device = torch.Generator(device=self.device).manual_seed(args.seed)

        # for convenience
        self.style = self.x_cfg.style
        self.im_size = self.x_cfg.image_size
        self.sive_cfg = self.x_cfg.sive
        self.sive_optim = self.x_cfg.sive_stage_optim
        self.vpsd_cfg = self.x_cfg.vpsd
        self.vpsd_optim = self.x_cfg.vpsd_stage_optim

        if self.style == "pixelart":
            self.x_cfg.sive_stage_optim.lr_schedule = False
            self.x_cfg.vpsd_stage_optim.lr_schedule = False

    def painterly_rendering(self, text_prompt: str, target_file: AnyPath = None):
        # log prompts
        self.print(f"prompt: {text_prompt}")
        self.print(f"neg_prompt: {self.args.neg_prompt}\n")

        if self.args.skip_sive:
            # mode 1: optimization with VPSD from scratch
            self.print("optimization with VPSD from scratch...")
            input_svg_path: AnyPath = None
            input_images = None
        elif target_file is not None:
            # mode 2: load the SVG file and use VPSD finetune it (skip SIVE)
            assert pathlib.Path(target_file).exists() and is_valid_svg(target_file)
            self.print(f"load svg from {target_file} ...")
            self.print(f"SVG fine-tuning via VPSD...")
            input_svg_path: AnyPath = target_file
            img_path = (self.result_path / "init_image.png").as_posix()
            cairosvg.svg2png(url=input_svg_path.as_posix(), write_to=img_path)
            input_images = self.target_file_preprocess(img_path)
            self.x_cfg.coord_init = 'sparse'
        else:
            # mode 3: SIVE + VPSD
            input_svg_path, input_images = self.SIVE_stage(text_prompt)
            self.print("SVG fine-tuning via VPSD...")

        self.VPSD_stage(text_prompt, init_svg_path=input_svg_path, init_image=input_images)
        self.close(msg="painterly rendering complete.")

    def SIVE_stage(self, text_prompt: str):
        # init diffusion model
        pipeline = DiffusionPipeline(self.x_cfg.sive_model_cfg, self.args.diffuser, self.device)

        merged_svg_paths = []
        merged_images = []
        for i in range(self.vpsd_cfg.n_particle):
            select_sample_path = self.result_path / f'select_sample_{i}.png'

            # generate sample and attention map
            fg_attn_map, bg_attn_map, controller = self.extract_ldm_attn(self.x_cfg.sive_model_cfg,
                                                                         pipeline,
                                                                         text_prompt,
                                                                         select_sample_path,
                                                                         self.sive_cfg.attn_cfg,
                                                                         self.im_size,
                                                                         self.args.token_ind)
            # load selected file
            select_img = self.target_file_preprocess(select_sample_path.as_posix())
            self.print(f"load target file from: {select_sample_path.as_posix()}")

            # get objects by attention map
            fg_img, bg_img, fg_mask, bg_mask = self.extract_object(select_img, fg_attn_map, bg_attn_map, iter=i)
            self.print(f"fg_img shape: {fg_img.shape}, bg_img: {bg_img.shape}")

            # background rendering
            self.print(f"-> background rendering: ")
            bg_render_path = self.component_rendering(tag=f'{i}_bg',
                                                      prompt=text_prompt,
                                                      target_img=bg_img,
                                                      mask=bg_mask,
                                                      attention_map=bg_attn_map,
                                                      canvas_size=(self.im_size, self.im_size),
                                                      render_cfg=self.sive_cfg.bg,
                                                      optim_cfg=self.sive_optim,
                                                      log_png_dir=self.bg_png_logs_dir,
                                                      log_svg_dir=self.bg_svg_logs_dir)
            # foreground rendering
            self.print(f"-> foreground rendering: ")
            fg_render_path = self.component_rendering(tag=f'{i}_fg',
                                                      prompt=text_prompt,
                                                      target_img=fg_img,
                                                      mask=fg_mask,
                                                      attention_map=fg_attn_map,
                                                      canvas_size=(self.im_size, self.im_size),
                                                      render_cfg=self.sive_cfg.fg,
                                                      optim_cfg=self.sive_optim,
                                                      log_png_dir=self.fg_png_logs_dir,
                                                      log_svg_dir=self.fg_svg_logs_dir)
            # merge foreground and background
            merged_svg_path = self.result_path / f'SIVE_render_final_{i}.svg'
            merge_svg_files(
                svg_path_1=bg_render_path,
                svg_path_2=fg_render_path,
                merge_type='simple',
                output_svg_path=merged_svg_path.as_posix(),
                out_size=(self.im_size, self.im_size)
            )

            # foreground and background refinement
            # Note: you are not allowed to add further paths here
            if self.sive_cfg.tog.reinit:
                self.print("-> enable vector graphic refinement:")
                merged_svg_path = self.refine_rendering(tag=f'{i}_refine',
                                                        prompt=text_prompt,
                                                        target_img=select_img,
                                                        canvas_size=(self.im_size, self.im_size),
                                                        render_cfg=self.sive_cfg.tog,
                                                        optim_cfg=self.sive_optim,
                                                        init_svg_path=merged_svg_path)

            # svg-to-png, to tensor
            merged_png_path = self.result_path / f'SIVE_render_final_{i}.png'
            cairosvg.svg2png(url=merged_svg_path.as_posix(), write_to=merged_png_path.as_posix())

            # collect paths
            merged_svg_paths.append(merged_svg_path)
            merged_images.append(self.target_file_preprocess(merged_png_path))
            # empty attention record
            controller.reset()

            self.print(f"Vector Particle {i} Rendering End...\n")

        # free the VRAM
        del pipeline
        torch.cuda.empty_cache()
        # update paths
        self.x_cfg.num_paths = self.sive_cfg.bg.num_paths + self.sive_cfg.fg.num_paths

        return merged_svg_paths, merged_images

    def component_rendering(self,
                            tag: str,
                            prompt: AnyPath,
                            target_img: torch.Tensor,
                            mask: Union[np.ndarray, None],
                            attention_map: Union[np.ndarray, None],
                            canvas_size: Tuple[int, int],
                            render_cfg: omegaconf.DictConfig,
                            optim_cfg: omegaconf.DictConfig,
                            log_png_dir: pathlib.Path,
                            log_svg_dir: pathlib.Path):

        # set path_schedule
        path_schedule = self.get_path_schedule(render_cfg.path_schedule,
                                               render_cfg.schedule_each,
                                               render_cfg.num_paths)
        if render_cfg.style == 'pixelart':
            path_schedule = [render_cfg.grid]
        self.print(f"path_schedule: {path_schedule}")

        # for convenience
        n_iter = render_cfg.num_iter
        style = render_cfg.style
        trainable_bg = render_cfg.optim_bg
        total_step = len(path_schedule) * n_iter

        # set renderer
        renderer = CompPainter(style,
                               target_img,
                               canvas_size,
                               render_cfg.num_segments,
                               render_cfg.segment_init,
                               render_cfg.radius,
                               render_cfg.grid,
                               render_cfg.width,
                               device=self.device,
                               attn_init=render_cfg.use_attn_init and attention_map is not None,
                               attention_map=attention_map,
                               attn_prob_tau=render_cfg.softmax_tau)

        if attention_map is not None:
            # init fist control points by attention_map
            attn_thresh, select_inds = renderer.attn_init_points(num_paths=sum(path_schedule), mask=mask)
            # log attention, just once
            plot_attn(attention_map, attn_thresh, target_img, select_inds,
                      (self.sive_attn_dir / f"attention_{tag}_map.jpg").as_posix())
        else:
            # init fist control points by GT
            renderer.component_wise_path_init(pred=None, init_type=render_cfg.coord_init)

        optimizer_list = [
            CompPainterOptimizer(renderer, style, n_iter, optim_cfg, trainable_bg)
            for _ in range(len(path_schedule))
        ]

        pathn_record = []
        loss_weight_keep = 0
        step = 0
        loss_weight = 1
        with tqdm(initial=step, total=total_step, disable=not self.accelerator.is_main_process) as pbar:
            for path_idx, pathn in enumerate(path_schedule):
                # record path
                pathn_record.append(pathn)
                # init graphic
                img = renderer.init_image(num_paths=pathn)
                plot_img(img, self.sive_init_dir, fname=f"{tag}_init_img_{path_idx}")
                # rebuild optimizer
                optimizer_list[path_idx].init_optimizers(pid_delta=int(path_idx * pathn))

                pbar.write(f"=> adding {pathn} paths, n_path: {sum(pathn_record)}, "
                           f"n_point: {len(renderer.get_point_params())}, "
                           f"n_width: {len(renderer.get_width_params())}, "
                           f"n_color: {len(renderer.get_color_params())}")

                for t in range(n_iter):
                    raster_img = renderer.get_image(step=t).to(self.device)

                    if render_cfg.use_distance_weighted_loss and style == "iconography":
                        loss_weight = renderer.calc_distance_weight(loss_weight_keep)

                    # reconstruction loss
                    if style == "pixelart":
                        loss_recon = torch.nn.functional.l1_loss(raster_img, target_img)
                    else:
                        if render_cfg.use_distance_weighted_loss:
                            # UDF loss
                            loss_recon = ((raster_img - target_img) ** 2)
                            loss_recon = (loss_recon.sum(1) * loss_weight).mean()
                        else:
                            loss_recon = F.mse_loss(raster_img, target_img)

                    # Xing Loss for Self-Interaction Problem
                    loss_xing = torch.tensor(0.)
                    if style == "iconography":
                        loss_xing = xing_loss_fn(renderer.get_point_params()) * render_cfg.xing_loss_weight

                    # total loss
                    loss = loss_recon + loss_xing

                    lr_str = ""
                    for k, lr in optimizer_list[path_idx].get_lr().items():
                        lr_str += f"{k}_lr: {lr:.4f}, "

                    pbar.set_description(
                        lr_str +
                        f"L_total: {loss.item():.4f}, "
                        f"L_recon: {loss_recon.item():.4f}, "
                        f"L_xing: {loss_xing.item():.4e}"
                    )

                    # optimization
                    for i in range(path_idx + 1):
                        optimizer_list[i].zero_grad_()

                    loss.backward()

                    for i in range(path_idx + 1):
                        optimizer_list[i].step_()

                    renderer.clip_curve_shape()

                    if render_cfg.lr_schedule:
                        for i in range(path_idx + 1):
                            optimizer_list[i].update_lr()

                    if step % self.args.save_step == 0 and self.accelerator.is_main_process:
                        plot_couple(target_img,
                                    raster_img,
                                    step,
                                    prompt=prompt,
                                    output_dir=log_png_dir.as_posix(),
                                    fname=f"{tag}_iter{step}")
                        renderer.save_svg(log_svg_dir / f"{tag}_svg_iter{step}.svg")

                    step += 1
                    pbar.update(1)

                if render_cfg.use_distance_weighted_loss and style == "iconography":
                    loss_weight_keep = loss_weight.detach().cpu().numpy() * 1
                # calc center
                renderer.component_wise_path_init(raster_img)

        # end LIVE
        final_svg_fpth = self.sive_final_dir / f"{tag}_final_render.svg"
        renderer.save_svg(final_svg_fpth)

        return final_svg_fpth

    def refine_rendering(self,
                         tag: str,
                         prompt: str,
                         target_img: torch.Tensor,
                         canvas_size: Tuple[int, int],
                         render_cfg: omegaconf.DictConfig,
                         optim_cfg: omegaconf.DictConfig,
                         init_svg_path: str):
        # init renderer
        content_renderer = CompPainter(self.style,
                                       target_img,
                                       path_svg=init_svg_path,
                                       canvas_size=canvas_size,
                                       device=self.device)
        # init graphic
        img = content_renderer.init_image()
        plot_img(img, self.refine_dir, fname=f"{tag}_before_refined")

        n_iter = render_cfg.num_iter
        # build painter optimizer
        optimizer = CompPainterOptimizer(content_renderer, self.style, n_iter, optim_cfg)
        # init optimizer
        optimizer.init_optimizers()

        print(f"=> n_point: {len(content_renderer.get_point_params())}, "
              f"n_width: {len(content_renderer.get_width_params())}, "
              f"n_color: {len(content_renderer.get_color_params())}")

        step = 0
        with tqdm(initial=step, total=n_iter, disable=not self.accelerator.is_main_process) as pbar:
            for t in range(n_iter):
                raster_img = content_renderer.get_image(step=t).to(self.device)

                loss_recon = F.mse_loss(raster_img, target_img)

                lr_str = ""
                for k, lr in optimizer.get_lr().items():
                    lr_str += f"{k}_lr: {lr:.4f}, "

                pbar.set_description(lr_str + f"L_refine: {loss_recon.item():.4f}")

                # optimization
                optimizer.zero_grad_()
                loss_recon.backward()
                optimizer.step_()

                content_renderer.clip_curve_shape()

                if step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    plot_couple(target_img,
                                raster_img,
                                step,
                                prompt=prompt,
                                output_dir=self.refine_png_dir.as_posix(),
                                fname=f"{tag}_iter{step}")
                    content_renderer.save_svg(self.refine_svg_dir / f"{tag}_svg_iter{step}.svg")

                step += 1
                pbar.update(1)

        # update current svg
        content_renderer.save_svg(init_svg_path)
        # save
        img = content_renderer.get_image()
        plot_img(img, self.refine_dir, fname=f"{tag}_refined")

        return init_svg_path

    def VPSD_stage(self,
                   text_prompt: AnyStr,
                   init_svg_path: Union[List[AnyPath], AnyPath] = None,
                   init_image: Union[List[torch.Tensor], torch.Tensor] = None):
        if not self.vpsd_cfg.use:
            return

        # for convenience
        guidance_cfg = self.x_cfg.vpsd
        vpsd_model_cfg = self.x_cfg.vpsd_model_cfg
        n_particle = guidance_cfg.n_particle
        total_step = guidance_cfg.num_iter
        path_reinit = self.x_cfg.path_reinit

        # init VPSD
        pipeline = VectorizedParticleSDSPipeline(vpsd_model_cfg, self.args.diffuser, guidance_cfg, self.device)
        # init reward model
        reward_model = None
        if guidance_cfg.phi_ReFL:
            reward_model = RM.load("ImageReward-v1.0", device=self.device, download_root=self.x_cfg.reward_path)

        # create svg renderer
        if isinstance(init_svg_path, List):  # mode 3
            renderers = [self.load_renderer(init_path) for init_path in init_svg_path]
        elif isinstance(init_svg_path, (str, pathlib.Path, os.PathLike)):  # mode 2
            renderers = [self.load_renderer(init_svg_path) for _ in range(n_particle)]
            init_image = [init_image] * n_particle
        else:  # mode 1
            renderers = [self.load_renderer(init_svg_path) for _ in range(n_particle)]
            if self.x_cfg.color_init == 'rand':  # randomly init
                init_img = torch.randn(1, 3, self.im_size, self.im_size)
            else:  # specified color
                init_img = init_tensor_with_color(self.x_cfg.color_init, 1, self.im_size, self.im_size)
                self.print(f"color: {self.x_cfg.color_init}")
            plot_img(init_img, self.result_path, fname='target_img')
            init_image = [init_img] * n_particle

        # initialize the particles
        for render, gt_ in zip(renderers, init_image):
            render.component_wise_path_init(gt=gt_, pred=None, init_type=self.x_cfg.coord_init)

        # log init images
        for i, r in enumerate(renderers):
            init_imgs = r.init_image(num_paths=self.x_cfg.num_paths)
            plot_img(init_imgs, self.ft_init_dir, fname=f"init_img_stage_two_{i}")

        # init renderer optimizer
        optimizers = []
        for renderer in renderers:
            optim_ = PainterOptimizer(renderer,
                                      self.style,
                                      guidance_cfg.num_iter,
                                      self.vpsd_optim,
                                      self.x_cfg.trainable_bg)
            optim_.init_optimizers()
            optimizers.append(optim_)

        # init phi_model optimizer
        phi_optimizer = get_optimizer('adamW',
                                      pipeline.phi_params,
                                      guidance_cfg.phi_lr,
                                      guidance_cfg.phi_optim)
        # init phi_model lr scheduler
        phi_scheduler = None
        schedule_cfg = guidance_cfg.phi_schedule
        if schedule_cfg.use:
            phi_lr_lambda = CosineWithWarmupLRLambda(num_steps=schedule_cfg.total_step,
                                                     warmup_steps=schedule_cfg.warmup_steps,
                                                     warmup_start_lr=schedule_cfg.warmup_start_lr,
                                                     warmup_end_lr=schedule_cfg.warmup_end_lr,
                                                     cosine_end_lr=schedule_cfg.cosine_end_lr)
            phi_scheduler = LambdaLR(phi_optimizer, lr_lambda=phi_lr_lambda, last_epoch=-1)

        self.print(f"-> Painter point Params: {len(renderers[0].get_point_parameters())}")
        self.print(f"-> Painter color Params: {len(renderers[0].get_color_parameters())}")
        self.print(f"-> Painter width Params: {len(renderers[0].get_width_parameters())}")

        L_reward = torch.tensor(0.)

        self.step = 0  # reset global step
        self.print(f"Total Optimization Steps: {total_step}")
        with tqdm(initial=self.step, total=total_step, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < total_step:
                # set particles
                particles = [renderer.get_image() for renderer in renderers]
                raster_imgs = torch.cat(particles, dim=0)

                if self.make_video and (self.step % self.args.framefreq == 0 or self.step == total_step - 1):
                    plot_img(raster_imgs, self.frame_log_dir, fname=f"iter{self.frame_idx}")
                    self.frame_idx += 1

                L_guide, grad, latents, t_step = pipeline.variational_score_distillation(
                    raster_imgs,
                    self.step,
                    prompt=[text_prompt],
                    negative_prompt=self.args.neg_prompt,
                    grad_scale=guidance_cfg.grad_scale,
                    enhance_particle=guidance_cfg.particle_aug,
                    im_size=model2res(vpsd_model_cfg.model_id)
                )

                # Xing Loss for Self-Interaction Problem
                L_add = torch.tensor(0.)
                if self.style == "iconography" or self.x_cfg.xing_loss.use:
                    for r in renderers:
                        L_add += xing_loss_fn(r.get_point_parameters()) * self.x_cfg.xing_loss.weight

                loss = L_guide + L_add

                # optimization
                for opt_ in optimizers:
                    opt_.zero_grad_()
                loss.backward()
                for opt_ in optimizers:
                    opt_.step_()

                # phi_model optimization
                for _ in range(guidance_cfg.phi_update_step):
                    L_lora = pipeline.train_phi_model(latents, guidance_cfg.phi_t, as_latent=True)

                    phi_optimizer.zero_grad()
                    L_lora.backward()
                    phi_optimizer.step()

                # reward learning
                if guidance_cfg.phi_ReFL and self.step % guidance_cfg.phi_sample_step == 0:
                    with torch.no_grad():
                        phi_outputs = []
                        phi_sample_paths = []
                        for idx in range(guidance_cfg.n_phi_sample):
                            phi_output = pipeline.sample(text_prompt,
                                                         num_inference_steps=guidance_cfg.phi_infer_step,
                                                         generator=self.g_device)
                            sample_path = (self.phi_samples_dir / f'iter{idx}.png').as_posix()
                            phi_output.images[0].save(sample_path)
                            phi_sample_paths.append(sample_path)

                            phi_output_np = np.array(phi_output.images[0])
                            phi_outputs.append(phi_output_np)
                        # save all samples
                        view_images(phi_outputs, save_image=True,
                                    num_rows=max(len(phi_outputs) // 6, 1),
                                    fp=self.phi_samples_dir / f'samples_iter{self.step}.png')

                    ranking, rewards = reward_model.inference_rank(text_prompt, phi_sample_paths)
                    self.print(f"ranking: {ranking}, reward score: {rewards}")

                    for k in range(guidance_cfg.n_phi_sample):
                        phi = self.target_file_preprocess(phi_sample_paths[ranking[k] - 1])
                        L_reward = pipeline.train_phi_model_refl(phi, weight=rewards[k])

                        phi_optimizer.zero_grad()
                        L_reward.backward()
                        phi_optimizer.step()

                # update the learning rate of the phi_model
                if phi_scheduler is not None:
                    phi_scheduler.step()

                # curve regularization
                for r in renderers:
                    r.clip_curve_shape()

                # re-init paths
                if path_reinit.use and self.step % path_reinit.freq == 0 and self.step < path_reinit.stop_step and self.step != 0:
                    for i, r in enumerate(renderers):
                        extra_point_params, extra_color_params, extra_width_params = \
                            r.reinitialize_paths(f"P{i} - Step {self.step}",
                                                 self.reinit_dir / f"reinit-{self.step}_p{i}.svg",
                                                 path_reinit.opacity_threshold,
                                                 path_reinit.area_threshold)
                        optimizers[i].add_params(extra_point_params, extra_color_params, extra_width_params)

                # update lr
                if self.vpsd_optim.lr_schedule:
                    for opt_ in optimizers:
                        opt_.update_lr()

                # log pretrained model lr
                lr_str = ""
                for k, lr in optimizers[0].get_lr().items():
                    lr_str += f"{k}_lr: {lr:.4f}, "
                # log phi model lr
                cur_phi_lr = phi_optimizer.param_groups[0]['lr']
                lr_str += f"phi_lr: {cur_phi_lr:.3e}, "

                pbar.set_description(
                    lr_str +
                    f"t: {t_step.item():.2f}, "
                    f"L_total: {loss.item():.4f}, "
                    f"L_add: {L_add.item():.4e}, "
                    f"L_lora: {L_lora.item():.4f}, "
                    f"L_reward: {L_reward.item():.4f}, "
                    f"grad: {grad.item():.4e}"
                )

                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    # save png
                    torchvision.utils.save_image(raster_imgs,
                                                 fp=self.ft_png_logs_dir / f'iter{self.step}.png')

                    # save svg
                    for i, r in enumerate(renderers):
                        r.pretty_save_svg(self.ft_svg_logs_dir / f"svg_iter{self.step}_p{i}.svg")

                self.step += 1
                pbar.update(1)

        # save final
        for i, r in enumerate(renderers):
            ft_svg_path = self.result_path / f"finetune_final_p_{i}.svg"
            r.pretty_save_svg(ft_svg_path)
        # save SVGs
        torchvision.utils.save_image(raster_imgs, fp=self.result_path / f'all_particles.png')

        if self.make_video:
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.framerate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                (self.result_path / "svgdreamer_rendering.mp4").as_posix()
            ])

    def load_renderer(self, path_svg=None):
        renderer = Painter(self.args.diffvg,
                           self.style,
                           self.x_cfg.num_segments,
                           self.x_cfg.segment_init,
                           self.x_cfg.radius,
                           self.im_size,
                           self.x_cfg.grid,
                           self.x_cfg.trainable_bg,
                           self.x_cfg.width,
                           path_svg=path_svg,
                           device=self.device)
        return renderer

    def target_file_preprocess(self, tar_path: AnyPath):
        process_comp = transforms.Compose([
            transforms.Resize(size=(self.im_size, self.im_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
        ])

        tar_pil = Image.open(tar_path).convert("RGB")  # open file
        target_img = process_comp(tar_pil)  # preprocess
        target_img = target_img.to(self.device)
        return target_img

    def extract_object(self,
                       select_img: torch.Tensor,
                       fg_attn_map: np.ndarray,
                       bg_attn_map: np.ndarray,
                       iter: Union[str, int],
                       tau: float = 0.2):
        # attention to mask
        bool_fg_attn_map = fg_attn_map > tau
        fg_mask = bool_fg_attn_map.astype(int)  # [w, h]

        def shrink_mask_contour(mask, epsilon_factor=0.05, erosion_kernel_size=5, dilation_kernel_size=5):
            """Shrink the contours of a binary mask image.

            Args:
                mask (numpy.ndarray): Binary mask image.
                epsilon_factor (float, optional): Factor for adjusting contour approximation precision. Defaults to 0.01.
                erosion_kernel_size (int, optional): Size of the kernel for erosion operation. Defaults to 3.
                dilation_kernel_size (int, optional): Size of the kernel for dilation operation. Defaults to 3.

            Returns:
                numpy.ndarray: Mask image with shrunk contours.
            """
            mask = mask.astype(np.uint8) * 255

            # Find contours in the input image
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Approximate the contours
            for contour in contours:
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(mask, [approx], 0, (255), -1)

            # Use erosion operation to further shrink the contours
            kernel_erode = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
            mask = cv2.erode(mask, kernel_erode, iterations=1)

            # Use dilation operation to further shrink the contours
            kernel_dilate = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel_dilate, iterations=1)

            mask = (mask / 255).astype(np.uint8)
            return mask

        # shrunk_mask
        fg_mask = shrink_mask_contour(fg_mask)
        # get background mask
        bg_mask = 1 - fg_mask

        # masked image, and save in place
        select_img_np = select_img.cpu().numpy()
        fg_img = fg_mask * select_img_np  # [1, 3, w, h]
        fg_mask_ = np.expand_dims(np.array([fg_mask, fg_mask, fg_mask]), axis=0)  # [w,h] -> [1,3,w,h]
        fg_img[fg_mask_ == 0] = 1
        fg_img = (fg_img / fg_img.max() * 255)
        save_image(fg_img[0], self.mask_dir / f'{iter}_mask_fg.png')

        bg_img = bg_mask * select_img_np
        bg_mask_ = np.expand_dims(np.array([bg_mask, bg_mask, bg_mask]), axis=0)
        bg_img[bg_mask_ == 0] = 1
        bg_img = (bg_img / bg_img.max() * 255)
        save_image(bg_img[0], self.mask_dir / f'{iter}_mask_bg.png')

        # to Tensor
        fg_img_final = self.target_file_preprocess(self.mask_dir / f'{iter}_mask_fg.png')
        bg_img_final = self.target_file_preprocess(self.mask_dir / f'{iter}_mask_bg.png')

        # [1,3,w,h] -> [w,h]
        fg_mask = fg_mask_[0][0, :, :]
        bg_mask = 1 - fg_mask
        return fg_img_final, bg_img_final, fg_mask, bg_mask

    def extract_ldm_attn(self,
                         model_cfg: omegaconf.DictConfig,
                         pipeline: DiffusionPipeline,
                         prompts: str,
                         gen_sample_path: AnyPath,
                         attn_init_cfg: omegaconf.DictConfig,
                         image_size: int,
                         token_ind: int,
                         attn_init: bool = True, ):
        if token_ind <= 0:
            raise ValueError("The 'token_ind' should be greater than 0")

        # init controller
        controller = AttentionStore() if attn_init else EmptyControl()

        # forward once and record attention map
        height = width = model2res(model_cfg.model_id)
        outputs = pipeline.sample(prompt=[prompts],
                                  height=height,
                                  width=width,
                                  num_inference_steps=model_cfg.num_inference_steps,
                                  controller=controller,
                                  guidance_scale=model_cfg.guidance_scale,
                                  negative_prompt=self.args.neg_prompt,
                                  generator=self.g_device)
        outputs_np = [np.array(img) for img in outputs.images]
        view_images(outputs_np, save_image=True, fp=gen_sample_path)
        self.print(f"select_sample shape: {outputs_np[0].shape}")

        if attn_init:
            """ldm cross-attention map"""
            cross_attention_maps, tokens = \
                pipeline.get_cross_attention([prompts],
                                             controller,
                                             res=attn_init_cfg.cross_attn_res,
                                             from_where=("up", "down"),
                                             save_path=self.sive_attn_dir / "cross_attn.png")

            self.print(f"the length of tokens is {len(tokens)}, select {token_ind}-th token")
            # [res, res, seq_len]
            self.print(f"origin cross_attn_map shape: {cross_attention_maps.shape}")
            # [res, res]
            cross_attn_map = cross_attention_maps[:, :, token_ind]
            self.print(f"select cross_attn_map shape: {cross_attn_map.shape}")
            cross_attn_map = 255 * cross_attn_map / cross_attn_map.max()
            # [res, res, 3]
            cross_attn_map = cross_attn_map.unsqueeze(-1).expand(*cross_attn_map.shape, 3)
            # [3, res, res]
            cross_attn_map = cross_attn_map.permute(2, 0, 1).unsqueeze(0)
            # [3, clip_size, clip_size]
            cross_attn_map = F.interpolate(cross_attn_map, size=image_size, mode='bicubic')
            cross_attn_map = torch.clamp(cross_attn_map, min=0, max=255)
            # rgb to gray
            cross_attn_map = rgb2gray(cross_attn_map.squeeze(0).permute(1, 2, 0)).astype(np.float32)
            # torch to numpy
            if cross_attn_map.shape[-1] != image_size and cross_attn_map.shape[-2] != image_size:
                cross_attn_map = cross_attn_map.reshape(image_size, image_size)
            # to [0, 1]
            cross_attn_map = (cross_attn_map - cross_attn_map.min()) / (cross_attn_map.max() - cross_attn_map.min())

            """ldm self-attention map"""
            self_attention_maps, svd, vh_ = \
                pipeline.get_self_attention_comp([prompts],
                                                 controller,
                                                 res=attn_init_cfg.self_attn_res,
                                                 from_where=("up", "down"),
                                                 img_size=image_size,
                                                 max_com=attn_init_cfg.max_com,
                                                 save_path=self.sive_attn_dir)

            # comp self-attention map
            if attn_init_cfg.mean_comp:
                self_attn = np.mean(vh_, axis=0)
                self.print(f"use the mean of {attn_init_cfg.max_com} comps.")
            else:
                self_attn = vh_[attn_init_cfg.comp_idx]
                self.print(f"select {attn_init_cfg.comp_idx}-th comp.")
            # to [0, 1]
            self_attn = (self_attn - self_attn.min()) / (self_attn.max() - self_attn.min())
            # visual final self-attention
            self_attn_vis = np.copy(self_attn)
            self_attn_vis = self_attn_vis * 255
            self_attn_vis = np.repeat(np.expand_dims(self_attn_vis, axis=2), 3, axis=2).astype(np.uint8)
            view_images(self_attn_vis, save_image=True, fp=self.sive_attn_dir / "self-attn-final.png")

            """get final attention map"""
            attn_map = attn_init_cfg.attn_coeff * cross_attn_map + (1 - attn_init_cfg.attn_coeff) * self_attn
            # to [0, 1]
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
            # visual fusion-attention
            attn_map_vis = np.copy(attn_map)
            attn_map_vis = attn_map_vis * 255
            attn_map_vis = np.repeat(np.expand_dims(attn_map_vis, axis=2), 3, axis=2).astype(np.uint8)
            view_images(attn_map_vis, save_image=True, fp=self.sive_attn_dir / 'fusion-attn.png')

            # inverse fusion-attention to [0, 1]
            inverse_attn = 1 - attn_map
            # visual reversed fusion-attention
            reversed_attn_map_vis = np.copy(inverse_attn)
            reversed_attn_map_vis = reversed_attn_map_vis * 255
            reversed_attn_map_vis = np.repeat(np.expand_dims(reversed_attn_map_vis, axis=2), 3, axis=2).astype(np.uint8)
            view_images(reversed_attn_map_vis, save_image=True, fp=self.sive_attn_dir / 'reversed-fusion-attn.png')

            self.print(f"-> fusion attn_map: {attn_map.shape}")
        else:
            attn_map = None
            inverse_attn = None

        return attn_map, inverse_attn, controller

    def get_path_schedule(self,
                          path_schedule: str,
                          schedule_each: Union[int, List],
                          num_paths: int = None):
        if path_schedule == 'repeat':
            assert num_paths is not None
            return int(num_paths / schedule_each) * [schedule_each]
        elif path_schedule == 'list':
            assert isinstance(schedule_each, list) or isinstance(schedule_each, omegaconf.ListConfig)
            return schedule_each
        else:
            raise NotImplementedError
