# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
import PIL
from PIL import Image
from typing import Any, List, Optional, Union, Dict
from omegaconf import DictConfig

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg, StableDiffusionPipelineOutput)

from svgdreamer.diffusers_warp import init_StableDiffusion_pipeline
from svgdreamer.token2attn.attn_control import AttentionStore
from svgdreamer.token2attn.ptp_utils import text_under_image, view_images


class DiffusionPipeline(torch.nn.Module):

    def __init__(self, model_cfg: DictConfig, diffuser_cfg: DictConfig, device: torch.device):
        super().__init__()
        self.device = device

        pipe_kwargs = {
            "device": self.device,
            "torch_dtype": torch.float32,
            "local_files_only": not diffuser_cfg.download,
            "force_download": diffuser_cfg.force_download,
            "resume_download": diffuser_cfg.resume_download,
            "ldm_speed_up": model_cfg.ldm_speed_up,
            "enable_xformers": model_cfg.enable_xformers,
            "gradient_checkpoint": model_cfg.gradient_checkpoint,
            "cpu_offload": model_cfg.cpu_offload,
            "vae_slicing": False
        }

        # load pretrained model
        self.sd_pipeline = init_StableDiffusion_pipeline(
            model_cfg.model_id,
            custom_pipeline=StableDiffusionPipeline,
            custom_scheduler=DDIMScheduler,
            **pipe_kwargs
        )
        # disable grads
        self.sd_pipeline.vae.requires_grad_(False)
        self.sd_pipeline.text_encoder.requires_grad_(False)
        self.sd_pipeline.unet.requires_grad_(False)
        # set components
        self.vae = self.sd_pipeline.vae
        self.unet = self.sd_pipeline.unet
        self.scheduler = self.sd_pipeline.scheduler
        self.tokenizer = self.sd_pipeline.tokenizer
        self.text_encoder = self.sd_pipeline.text_encoder

    @torch.no_grad()
    def encode_prompt(self,
                      prompt,
                      device,
                      do_classifier_free_guidance,
                      negative_prompt=None):
        # text conditional embed
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(device))[0]

        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""]
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt

            # unconditional embed
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=prompt_embeds.shape[1],
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(device))[0]

            concat_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            return concat_prompt_embeds, negative_prompt_embeds, prompt_embeds

        return prompt_embeds, None, None

    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count

    @staticmethod
    def aggregate_attention(prompts,
                            attention_store: AttentionStore,
                            res: int,
                            from_where: List[str],
                            is_cross: bool,
                            select: int):
        if isinstance(prompts, str):
            prompts = [prompts]
        assert isinstance(prompts, list)

        out = []
        attention_maps = attention_store.get_average_attention()
        num_pixels = res ** 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out.cpu()

    def get_cross_attention(self,
                            prompts,
                            attention_store: AttentionStore,
                            res: int,
                            from_where: List[str],
                            select: int = 0,
                            save_path=None):
        tokens = self.tokenizer.encode(prompts[select])
        decoder = self.tokenizer.decode
        # shape: [res ** 2, res ** 2, seq_len]
        attention_maps = self.aggregate_attention(prompts, attention_store, res, from_where, True, select)

        images_text = []
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            images.append(np.copy(image))
            image = text_under_image(image, decoder(int(tokens[i])))
            images_text.append(image)
        image_array = np.stack(images_text, axis=0)
        view_images(image_array, save_image=True, fp=save_path)

        return attention_maps, tokens

    def get_self_attention_comp(self,
                                iter: int,
                                prompts: List[str],
                                attention_store: AttentionStore,
                                res: int,
                                from_where: List[str],
                                img_size: int = 224,
                                max_com=10,
                                select: int = 0,
                                save_path=None):
        attention_maps = self.aggregate_attention(prompts, attention_store, res, from_where, False, select)
        attention_maps = attention_maps.numpy().reshape((res ** 2, res ** 2))
        # shape: [res ** 2, res ** 2]
        u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
        print(f"self-attention maps: {attention_maps.shape}, "
              f"u: {u.shape}, "
              f"s: {s.shape}, "
              f"vh: {vh.shape}")

        images = []
        vh_returns = []
        for i in range(max_com):
            image = vh[i].reshape(res, res)
            image = (image - image.min()) / (image.max() - image.min())
            image = 255 * image

            ret_ = Image.fromarray(image).resize((img_size, img_size), resample=PIL.Image.Resampling.BILINEAR)
            vh_returns.append(np.array(ret_))

            image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
            image = Image.fromarray(image).resize((256, 256))
            image = np.array(image)
            images.append(image)
        image_array = np.stack(images, axis=0)
        view_images(image_array, num_rows=max_com // 10, offset_ratio=0,
                    save_image=True, fp=save_path / f"self-attn-vh-{iter}.png")

        return attention_maps, (u, s, vh), np.stack(vh_returns, axis=0)

    def sampling(self,
                 vae,
                 unet,
                 scheduler,
                 prompt: Union[str, List[str]] = None,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 controller: AttentionStore = None,  # feed attention_store as control of ptp
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 negative_prompt: Optional[Union[str, List[str]]] = None,
                 num_images_per_prompt: Optional[int] = 1,
                 eta: float = 0.0,
                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 latents: Optional[torch.FloatTensor] = None,
                 output_type: Optional[str] = "pil",
                 return_dict: bool = True,
                 cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                 guidance_rescale: float = 0.0):

        # add attention controller
        self.register_attention_control(controller)

        # 0. Default height and width to unet
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        height = height or unet.config.sample_size * vae_scale_factor
        width = width or unet.config.sample_size * vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, _, _ = self.encode_prompt(
            prompt,
            self.device,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # 4. Prepare timesteps
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = unet.config.in_channels
        latents = self.sd_pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.sd_pipeline.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self.sd_pipeline.set_progress_bar_config(desc='DDPM Sampling')
        with self.sd_pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # controller callback
                latents = controller.step_callback(latents)

                # update progress_bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.sd_pipeline.run_safety_checker(image, self.device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.sd_pipeline.image_processor.postprocess(image, output_type=output_type,
                                                             do_denormalize=do_denormalize)
        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def sample(self,
               prompt,
               height: Optional[int] = None,
               width: Optional[int] = None,
               controller: AttentionStore = None,  # feed attention_store as control of ptp
               num_inference_steps: int = 50,
               guidance_scale: float = 7.5,
               negative_prompt: Optional[Union[str, List[str]]] = None,
               generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
               output_type: Optional[str] = "pil"):
        return self.sampling(self.vae, self.unet, self.scheduler,
                             prompt=prompt,
                             height=height, width=width,
                             controller=controller,
                             num_inference_steps=num_inference_steps,
                             guidance_scale=guidance_scale,
                             negative_prompt=negative_prompt,
                             generator=generator,
                             output_type=output_type)

    def encode2latent(self, images):
        images = (2 * images - 1).clamp(-1.0, 1.0)  # images: [B, 3, H, W]
        # encode images
        latents = self.vae.encode(images).latent_dist.sample()
        latents = self.vae.config.scaling_factor * latents
        return latents


class P2PCrossAttnProcessor:

    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
