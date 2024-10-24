import torch
import torch.nn
import torch.nn.functional as F
from einops import rearrange

def images_to_patches(images, patch_size, policy_indices):
    # group_quota should be a tuple of integers (base, scale_2, scale_4, potentially more)
    B, C, H, W = images.size()
    assert H % patch_size == 0 and W % patch_size == 0
    base_grid_H, base_grid_W = H // patch_size, W // patch_size

    # prepare all possible patches at different scale
    # base level, no grouping and rescale
    patch_scale_1 = rearrange(images, 'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w', gh=base_grid_H, gw=base_grid_W)
    scale_value_1 = torch.ones([B, base_grid_H, base_grid_W, 1])
    patch_code_scale_1 = torch.cat([scale_value_1,
                                    torch.linspace(0, base_grid_H - 1, base_grid_H).view(-1, 1, 1).expand_as(
                                        scale_value_1),
                                    torch.linspace(0, base_grid_W - 1, base_grid_W).view(1, -1, 1).expand_as(
                                        scale_value_1)], dim=3)

    # group 2*2 patches and resize them to the defined patch size
    patch_scale_2 = rearrange(F.interpolate(images, scale_factor=0.5, mode='bilinear',
                                            align_corners=False, recompute_scale_factor=False),
                              'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w', gh=base_grid_H // 2,
                              gw=base_grid_W // 2)
    patch_code_scale_2 = torch.clone(patch_code_scale_1)[:, ::2, ::2, :]
    patch_code_scale_2[:, :, :, 0] = 2

    (selected_msk_scale_1, selected_msk_scale_2) = policy_indices

    patch_code_scale_2_selected = patch_code_scale_2[selected_msk_scale_2]
    patch_code_scale_2_selected = rearrange(patch_code_scale_2_selected, '(b np) c -> b np c', b=B)
    patch_scale_2_selected = patch_scale_2[selected_msk_scale_2]
    patch_scale_2_selected = rearrange(patch_scale_2_selected, '(b np) c h w -> b np c h w', b=B)

    patch_code_scale_1_selected = patch_code_scale_1[selected_msk_scale_1]
    patch_code_scale_1_selected = rearrange(patch_code_scale_1_selected, '(b np) c -> b np c', b=B)
    patch_scale_1_selected = patch_scale_1[selected_msk_scale_1]
    patch_scale_1_selected = rearrange(patch_scale_1_selected, '(b np) c h w -> b np c h w', b=B)

    patches_total = torch.cat([patch_scale_1_selected, patch_scale_2_selected], dim=1)
    patch_code_total = torch.cat([patch_code_scale_1_selected, patch_code_scale_2_selected], dim=1)

    patches_total = rearrange(patches_total, 'b np c ps_h ps_w -> b c ps_h (np ps_w)')

    return patches_total, patch_code_total


def images_to_patches(images, patch_size, policy_indices):
    # group_quota should be a tuple of integers (base, scale_2, scale_4, potentially more)
    B, C, H, W = images.size()
    assert H % patch_size == 0 and W % patch_size == 0
    base_grid_H, base_grid_W = H // patch_size, W // patch_size

    # prepare all possible patches at different scale
    # base level, no grouping and rescale
    patch_scale_1 = rearrange(images, 'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w', gh=base_grid_H, gw=base_grid_W)
    scale_value_1 = torch.ones([B, base_grid_H, base_grid_W, 1])
    patch_code_scale_1 = torch.cat([scale_value_1,
                                    torch.linspace(0, base_grid_H - 1, base_grid_H).view(-1, 1, 1).expand_as(
                                        scale_value_1),
                                    torch.linspace(0, base_grid_W - 1, base_grid_W).view(1, -1, 1).expand_as(
                                        scale_value_1)], dim=3)

    # group 2*2 patches and resize them to the defined patch size
    patch_scale_2 = rearrange(F.interpolate(images, scale_factor=0.5, mode='bilinear',
                                            align_corners=False, recompute_scale_factor=False),
                              'b c (gh ps_h) (gw ps_w) -> b gh gw c ps_h ps_w', gh=base_grid_H // 2,
                              gw=base_grid_W // 2)
    patch_code_scale_2 = torch.clone(patch_code_scale_1)[:, ::2, ::2, :]
    patch_code_scale_2[:, :, :, 0] = 2

    (selected_msk_scale_1, selected_msk_scale_2) = policy_indices

    patch_code_scale_2_selected = patch_code_scale_2[selected_msk_scale_2]
    patch_code_scale_2_selected = rearrange(patch_code_scale_2_selected, '(b np) c -> b np c', b=B)
    patch_scale_2_selected = patch_scale_2[selected_msk_scale_2]
    patch_scale_2_selected = rearrange(patch_scale_2_selected, '(b np) c h w -> b np c h w', b=B)

    patch_code_scale_1_selected = patch_code_scale_1[selected_msk_scale_1]
    patch_code_scale_1_selected = rearrange(patch_code_scale_1_selected, '(b np) c -> b np c', b=B)
    patch_scale_1_selected = patch_scale_1[selected_msk_scale_1]
    patch_scale_1_selected = rearrange(patch_scale_1_selected, '(b np) c h w -> b np c h w', b=B)

    patches_total = torch.cat([patch_scale_1_selected, patch_scale_2_selected], dim=1)
    patch_code_total = torch.cat([patch_code_scale_1_selected, patch_code_scale_2_selected], dim=1)

    patches_total = rearrange(patches_total, 'b np c ps_h ps_w -> b c ps_h (np ps_w)')

    return patches_total, patch_code_total


def patches_to_images(patches, policy_code, grid_size):
    batch_size, dim_patch, patch_size, ps_times_num_patch = patches.size()
    num_patch = ps_times_num_patch // patch_size
    num_grid_h, num_grid_w = grid_size  # grid size is based on the original base patch size
    patches = rearrange(patches, 'b c hp (np wp) ->b np c hp wp', np=num_patch)
    num_total_grid = num_grid_h * num_grid_w

    scale_value = policy_code[:, :, 0]
    grid_coords = policy_code[:, :, 1:]

    # process patches that stay at the original scale
    scale_1_idx = scale_value == 1

    patch_scale_1 = patches[scale_1_idx]
    patch_scale_1 = rearrange(patch_scale_1, '(b np) c h w -> b np c h w', b=batch_size)
    grid_coord_1 = grid_coords[scale_1_idx].unsqueeze(1)
    grid_coord_1 = rearrange(grid_coord_1, '(b np) ng c -> b (np ng) c', b=batch_size)

    # process patches that are downsize by the factor of 2
    scale_2_idx = scale_value == 2
    # transform 1*1 original grid coord to 2*2 (because it will be upsampled by the factor of 2)
    grid_coord_2 = grid_coords[scale_2_idx].unsqueeze(1)  # coords are stacked across batch dim
    grid_coord_2 = torch.cat([grid_coord_2, grid_coord_2 + torch.tensor([[0, 1]]),
                              grid_coord_2 + torch.tensor([[1, 0]]), grid_coord_2 + torch.tensor([[1, 1]])], dim=1)
    grid_coord_2 = rearrange(grid_coord_2, '(b np) ng c -> b (np ng) c', b=batch_size)  # create batch dim

    # find and enlarege the patches that are downsize before, and break it into 4 pieces
    patch_scale_2 = patches[scale_2_idx]
    patch_scale_2 = F.interpolate(patch_scale_2, scale_factor=2, mode='bilinear', align_corners=False,
                                  recompute_scale_factor=False)  # stacked across batch dim
    patch_scale_2 = rearrange(patch_scale_2, 'b c (h1 h) (w1 w) -> b (h1 w1) c h w', h1=2, w1=2)
    patch_scale_2 = rearrange(patch_scale_2, '(b np) ng c ps_h ps_w  -> b (np ng) c ps_h ps_w', b=batch_size)

    # combine all the restored patches (of the same size and scale) together, total size should be 'num_total_grid'
    # even one shuffle the patches and grid value, it will be sorted later anyway
    patches_uni = torch.cat([patch_scale_1, patch_scale_2], dim=1)
    grid_coord_uni = torch.cat([grid_coord_1, grid_coord_2], dim=1)

    # sort the patches according to grid universal positional value, different samples batch-level have offset values
    grid_uni_value = grid_coord_uni[:, :, 0] * num_grid_w + grid_coord_uni[:, :, 1]
    batch_offset = torch.linspace(0, batch_size - 1, batch_size).view(batch_size, 1).expand_as(
        grid_uni_value) * num_total_grid
    grid_sort_global = batch_offset + grid_uni_value
    grid_sort_global = grid_sort_global.view(-1)
    patch_uni_global = rearrange(patches_uni, 'b np c h w -> (b np) c h w')
    indices_global = torch.argsort(grid_sort_global)
    patch_uni_global = patch_uni_global[indices_global]

    patch_uni_global = rearrange(patch_uni_global, '(b np) c h w  -> b np c h w', b=batch_size)
    images = rearrange(patch_uni_global, 'b (hp wp) c h w -> b c (hp h) (wp w)', hp=num_grid_h, wp=num_grid_w)

    return images


def convert_1d_index_to_2d(one_dim_index, PS):
    x_coord = one_dim_index // PS
    y_coord = one_dim_index % PS
    two_dim_index = torch.stack([x_coord, y_coord])
    two_dim_index = two_dim_index.permute(1, 0)
    return two_dim_index

def convert_2d_index_to_1d(two_dim_index, PS):
    one_dim_index = two_dim_index[:, 0] * PS + two_dim_index[:, 1]
    return one_dim_index

def get_2d_coords_scale_from_h_w_ps(height, width, patch_size, scale):
    patches_coords = torch.meshgrid(torch.arange(0, height // patch_size), torch.arange(0, width // patch_size), indexing='ij')
    patches_coords = torch.stack([patches_coords[0], patches_coords[1]])
    patches_coords = patches_coords.permute(1, 2, 0)
    patches_coords = patches_coords.view(-1, 2)
    n_patches = patches_coords.shape[0]

    scale_lvl = torch.tensor([[scale]] * n_patches)
    patches_scale_coords = torch.cat([scale_lvl, patches_coords], dim=1)
    return patches_scale_coords

def get_1d_coords_scale_from_h_w_ps(height, width, patch_size, scale):
    n_patches = (height // patch_size) * (width // patch_size)
    patches_coords = torch.arange(n_patches).view(-1, 1)

    scale_lvl = torch.tensor([[scale]] * n_patches)
    patches_scale_coords = torch.cat([scale_lvl, patches_coords], dim=1)
    return patches_scale_coords

def convert_scale_to_coords_in_full_res(one_dim_coords, patch_size):
    new_coords_all = []
    for coord in one_dim_coords:
        print(coord)
        new_coords = torch.stack(torch.meshgrid(torch.arange(coord*patch_size, coord*patch_size + patch_size), torch.arange(coord*patch_size, coord*patch_size + patch_size))).permute(1,2,0).view(-1, 2)
        new_coords_one_dim = convert_2d_index_to_1d(new_coords, patch_size)
        new_coords_all.append(new_coords_one_dim)
    return torch.cat(new_coords_all, dim=0)


