import torch
import timm

model = timm.create_model('efficientnetv2_rw_m', pretrained=False, in_chans=1)
model.global_pool = torch.nn.Identity()
model.classifier = torch.nn.Identity()
a = torch.randn(1, 1, 380, 380)

print(model(a).shape)
