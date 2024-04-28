import torch

model_path = "./models/control_sd15_canny.pth"

model = torch.load(model_path)

for key,val in model.items():
    print(f"{key}: {list(val.shape)}")
