"""SAMPLING ONLY."""
import sys
#sys.path.append("../")

import torch
import numpy as np
from tqdm import tqdm
import os
from Engine import Engine
from polygraphy import cuda
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor

import time
'''
How to avoid clip overflow?

https://blog.csdn.net/qq_19859865/article/details/139336523?spm=1001.2014.3001.5501


'''

# my intention is to create a simple interface for engine
class ClipEngine(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        # distingish between fp32,fp16 and int8
        clip_engine_path = "/mnt/data/github/TensorRT-StableDiffusion/engine/CLIP.plan"
        if not os.path.exists(clip_engine_path):
            print("invalid clip engine")
            return
        self.clip_engine = Engine(clip_engine_path)
        self.clip_engine.load()
        print("engine {} load".format(clip_engine_path))
        self.clip_engine.activate()
        clip_shape_dict = self.clip_engine.clip_model_shape_dict(1, 77, 768)
        self.clip_engine.allocate_buffers(clip_shape_dict)
        print("clip engine context load")
        self.stream = cuda.Stream()
        self.clip_engine.get_engine_infor()

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)



def main():
    #TODO: this main function not work yet
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
    from model import create_model, load_state_dict

    cldm_model = create_model('../models/cldm_v15.yaml').cpu()
    cldm_model.load_state_dict(load_state_dict('../models/control_sd15_canny.pth', location='cuda'))
    clip_trt = ClipEngine(cldm_model).clip_engine
    tokens = cldm_model.get_learned_token(['hello'])
    clip_engine_dict = clip_trt.infer({"input_ids": tokens})
    result = clip_engine_dict['last_hidden_state']#.cpu().numpy()
    gt = cldm_model.get_learned_conditioning(['hello'])


if __name__ == "__main__":
    main()