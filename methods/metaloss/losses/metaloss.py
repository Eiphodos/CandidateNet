import torch
from einops import rearrange

def compute_meta_loss(loss_target, meta_losses, meta_losses_coords, meta_loss_criterion, patch_sizes_used):
    res_meta_losses = []
    i = 0
    for ml, mlp, ps in zip(meta_losses, meta_losses_coords, patch_sizes_used):
        patched_target = rearrange(loss_target, 'b (nph psh) (npw psw) -> b nph npw (psh psw)', psh=ps, psw=ps)
        target = patched_target.mean(dim=3)
        #target = rearrange(target, 'b nph npw -> b (nph npw)')
        mlp_x = mlp[..., 0] // 2**(len(patch_sizes_used) - i - 1)
        mlp_y = mlp[..., 0] // 2**(len(patch_sizes_used) - i - 1)
        b = torch.arange(mlp.shape[0]).unsqueeze(-1).expand(-1, mlp.shape[1])
        filtered_targets = target[b, mlp_x, mlp_y]
        res = meta_loss_criterion(ml, filtered_targets)
        res_meta_losses.append(res)
        i += 1
    meta_loss = torch.stack(res_meta_losses).mean()
    return meta_loss