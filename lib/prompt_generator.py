import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingAwarePromptGenerator(nn.Module):
    """
    Grounding-Aware Prompts Generation (GPG) module inspired by SGSRF.

    This module takes the OT transport plan (which acts as a structural-semantic
    alignment map) and generates explicit geometric prompts for the SAM3 Decoder.
    
    It extracts:
    1. Sparse Prompts (Points): Coordinates of the highest probability regions.
    2. Dense Prompts (Masks): Low-resolution coarse mask priors.
    """
    def __init__(self, num_points=1):
        super().__init__()
        self.num_points = num_points
        print(f"[GPG] Initialized Grounding-Aware Prompt Generator (num_points={num_points})")

    # Removed @torch.no_grad() so gradients can flow to OT Module!
    def forward(self, P, text_mask, original_image_size, device):
        """
        Generate geometric prompts from the OT transport plan.
        Now uses Differentiable Center of Mass to allow gradient backpropagation.
        """
        B, HW, seq = P.shape
        H = W = int(HW ** 0.5)

        # 1. Mask out padding tokens
        if text_mask is not None:
            valid_mask = (~text_mask).unsqueeze(1)
            P_valid = P * valid_mask.float()
        else:
            P_valid = P

        # 2. Aggregate over sequence to get a global spatial heatmap
        heatmap = P_valid.sum(dim=-1)  # (B, HW)
        heatmap = heatmap.view(B, H, W)
        flat_heatmap = heatmap.view(B, -1)

        # 3. Differentiable Center of Mass (Spatial Expectation) for the primary point
        temperature = 0.1  # Sharpen the peak to act like argmax but differentiable
        prob_map = F.softmax(flat_heatmap / temperature, dim=-1)
        prob_map_2d = prob_map.view(B, H, W)

        # Create coordinate grids
        y_coords = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
        x_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)

        # Expected coordinates (Center of Mass)
        expected_y = (prob_map_2d * y_coords).sum(dim=(1, 2))  # (B,)
        expected_x = (prob_map_2d * x_coords).sum(dim=(1, 2))  # (B,)

        scale_y = original_image_size / H
        scale_x = original_image_size / W

        y_img_com = (expected_y + 0.5) * scale_y
        x_img_com = (expected_x + 0.5) * scale_x
        primary_point = torch.stack((x_img_com, y_img_com), dim=-1)  # (B, 2)

        # 4. Extract remaining points via standard detached top-k
        K_max = 5
        _, topk_idx = torch.topk(flat_heatmap.detach(), K_max, dim=-1)
        y_feat = torch.div(topk_idx, W, rounding_mode='floor')
        x_feat = topk_idx % W

        y_img_topk = (y_feat.float() + 0.5) * scale_y
        x_img_topk = (x_feat.float() + 0.5) * scale_x
        points = torch.stack((x_img_topk, y_img_topk), dim=-1)  # (B, K_max, 2)

        # Replace the first point with our differentiable Center of Mass point!
        # This allows gradients to flow directly through the primary point to the OT map.
        points = points.clone()
        points[:, 0, :] = primary_point

        # 5. Scale-Aware Dynamic Logic (Detached)
        b_max = flat_heatmap.detach().max(dim=1, keepdim=True)[0] + 1e-6
        norm_heatmap = flat_heatmap.detach() / b_max
        active_area = (norm_heatmap > 0.5).sum(dim=1).float() / (H * W)

        points_mask = torch.ones((B, K_max), dtype=torch.bool, device=device)
        point_labels = torch.ones((B, K_max), dtype=torch.long, device=device)

        for b in range(B):
            if active_area[b] < 0.01:
                # Tiny object: Keep only 1 center point to avoid background noise
                points_mask[b, 1:] = False
                point_labels[b, 1:] = -1  # CRITICAL: -1 is padding in SAM, 0 is negative (background) point!

        return points, points_mask, point_labels
