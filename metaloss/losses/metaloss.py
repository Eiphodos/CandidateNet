import torch
from einops import rearrange

def compute_meta_loss(outputs, labels, meta_losses, meta_losses_coords, meta_loss_target, meta_loss_criterion, patch_sizes_used):
    full_target = meta_loss_target(outputs.detach(), labels.squeeze(1).long())
    res_meta_losses = []
    for ml, mlp, ps in zip(meta_losses, meta_losses_coords, patch_sizes_used):
        patched_target = rearrange(full_target, 'b (nph psh) (npw psw) -> b nph npw (psh psw)', psh=ps, psw=ps)
        target = patched_target.mean(dim=3)
        target = rearrange(target, 'b nph npw -> b (nph npw)')
        filtered_targets = target[:, mlp]
        res = meta_loss_criterion(ml, filtered_targets)
        res_meta_losses.append(res)
    meta_loss = torch.stack(res_meta_losses).mean()
    return meta_loss