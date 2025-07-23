import torch

ckpt = torch.load('../result/test_Varnet/checkpoints/best_model.pt', map_location='cpu')
for key in ckpt['model'].keys():
    if key.startswith('cascades'):
        print(key)
