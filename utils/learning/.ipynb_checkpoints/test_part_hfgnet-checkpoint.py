import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.HFGNet import HFGNet

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    model = HFGNet(
        num_cascades=args.cascade,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
        in_ch=args.in_ch,
        l1_weight=args.l1_weight,
        gamma=args.gamma,
    )
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)

    # 디버깅
    print("Saved keys:")
    for k in checkpoint['model'].keys():
        print(k)
    
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)