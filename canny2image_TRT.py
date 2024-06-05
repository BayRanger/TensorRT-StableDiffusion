from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm_trt.model import create_model, load_state_dict
from cldm_trt.ddim_hacked import DDIMSampler
from cldm_trt.clip_engine import ClipEngine
from vae_engine import DecoderEngine

class hackathon():

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.cond_stage_model.cuda()
        self.use_trt = True
        # if not self.use_trt:
        if 1:
            self.model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
            self.model = self.model.cuda()

        self.clip_engine = ClipEngine(self.model)  #TODO: add
        self.clip_trt = self.clip_engine.clip_engine
        self.ddim_sampler = DDIMSampler(self.model)
        self.warm_up()

    def warm_up(self):
        for i in range(2):
            path = "./pictures_croped/bird_"+ str(i) + ".jpg"
            img = cv2.imread(path)
            new_img = self.process(img,
            "a bird",
            "best quality, extremely detailed",
            "longbody, lowres, bad anatomy, bad hands, missing fingers",
            1,
            256,
            20,
            False,
            1,
            9,
            2946901,
            0.0,
            100,
            200)
    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            # import pdb; pdb.set_trace()
            #TOOD: check clip
            cond_crossattn_trt = self.model.get_learned_token([prompt + ', ' + a_prompt] * num_samples)
            un_cond_crossattn_trt = self.model.get_learned_token([n_prompt] * num_samples)

            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            cond_tokens = self.model.get_learned_token([prompt + ', ' + a_prompt] * num_samples)
            un_cond_tokens = self.model.get_learned_token([n_prompt] * num_samples)

            # cond_tokens =  torch.tensor(cond_tokens)
            # un_cond_tokens = torch.tensor(un_cond_tokens)
            #tokens are the input of clip!

            cond_clip_trt_crosssattn = self.clip_trt.infer({"input_ids": cond_tokens})["last_hidden_state"]
            uncond_clip_trt_crosssattn = self.clip_trt.infer({"input_ids": un_cond_tokens})["last_hidden_state"]

            use_torch = False

            #TODO: debug
            if (use_torch):
                cond = {"c_concat": [control], "c_crossattn": \
                        [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
                un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": \
                        [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            else:
                cond = {"c_concat": [control], "c_crossattn":[cond_clip_trt_crosssattn]}
                un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn":[uncond_clip_trt_crosssattn]}
                            #here, we get clip inference result

            print(f"cond shape  {cond['c_crossattn'][0].shape}")
            print(f"un_cond shape  {un_cond['c_crossattn'][0].shape}")




            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            # self.model.control_scales = [strength] * 13
            samples, intermediates = self.ddim_sampler.sample_simple(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)
            # import pdb; pdb.set_trace()
            use_torch = True
            #if use_torch
            np.save("samples.npy", samples.cpu().numpy())
            scale = 1./0.18215
            '''
            x_samples_gt = self.model.decode_first_stage( samples)
            '''

            vae_trt = DecoderEngine(self.model).vae_engine
            vae_engine_dict = vae_trt.infer({"latent": scale *samples})
            x_samples = vae_engine_dict['images']#.cpu().numpy()

            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return results
