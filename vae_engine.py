"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
import os

from Engine import Engine
from polygraphy import cuda
#from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor

import time
'''
How to avoid vae overflow?

https://blog.csdn.net/qq_19859865/article/details/139336523?spm=1001.2014.3001.5501


'''

# my intention is to create a simple interface for engine
class DecoderEngine(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.vae_engine = None
        # distingish between fp32,fp16 and int8
        vae_engine_path = "/mnt/data/github/trt_stablediffusion_bk/TensorRT-StableDiffusion/engine/Decoder.plan"
        if not os.path.exists(vae_engine_path):
            print("invalid vae engine")
            return
        self.vae_engine = Engine(vae_engine_path)
        self.vae_engine.load()
        print("engine {} load".format(vae_engine_path))
        self.vae_engine.activate()
        vae_shape_dict = self.vae_engine.decoder_model_shape_dict()
        self.vae_engine.allocate_buffers(vae_shape_dict)
        print("vae engine context load")
        self.stream = cuda.Stream()
        self.vae_engine.get_engine_infor()

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)



def main():
    #TODO: this main function not work yet
    #import config
    import cv2
    import einops
    import gradio as gr
    import numpy as np
    import torch
    import random

    from pytorch_lightning import seed_everything
    #from annotator.util import resize_image, HWC3
    #from annotator.canny import CannyDetector
    from cldm_trt.model import create_model, load_state_dict



    data = torch.randn(1, 4, 32, 48, dtype=torch.float32)
    model = create_model('models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('models/control_sd15_canny.pth', location='cuda'))
    vae_trt = DecoderEngine(model).vae_engine
    vae_engine_dict = vae_trt.infer({"latent": data})
    result = vae_engine_dict['images']#.cpu().numpy()

    decode_model = model.first_stage_model
    decode_model.forward = decode_model.decode

    output = decode_model(data)

    ret = np.allclose(output.numpy(), result.detach().cpu().numpy(), rtol=1e-03, atol=1e-05, equal_nan=False)
    if (ret):
        print("======decoder test passed======")
    else:
        print("======decoder test failed======")


#TODO: add benchmark

if __name__ == "__main__":
    main()