import torch
import torch.nn as nn
import torch.nn.functional as F

def sobel_edges(mask):
    """
    Compute edge map using Sobel filters.
    mask: (B, 1, H, W)
    """
    device = mask.device
    # Sobel kernels
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)
    
    # Pad and conv
    mask_pad = F.pad(mask, (1, 1, 1, 1), mode='replicate')
    edge_x = F.conv2d(mask_pad, kx)
    edge_y = F.conv2d(mask_pad, ky)
    
    edges = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
    # Normalize
    edges = edges / (edges.flatten(2).max(dim=2)[0].view(-1, 1, 1, 1) + 1e-6)
    return edges

class TextGuidedBoundaryLoss(nn.Module):
    """
    L_TBL: Weighted boundary BCE where the text attention map
    modulates edge importance, ensuring focus on the REFERRED
    object's boundary (not background clutter edges).
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_masks, gt_masks, text_feats, vis_feats):
        """
        pred_masks: (B, 1, H, W) logits
        gt_masks: (B, 1, H, W) 0 or 1
        text_feats: (B, seq, C)
        vis_feats: (B, H_v, W_v, C) or flattened (B, HW, C)
        """
        # Ensure mask shapes
        if gt_masks.shape != pred_masks.shape:
            gt_masks = F.interpolate(gt_masks.float(), size=pred_masks.shape[-2:], mode='nearest')
            
        # Extract ground truth boundaries
        boundary = sobel_edges(gt_masks)  # (B, 1, H, W)
        
        # Compute simple text-image attention if features provided
        B, C, H, W = pred_masks.shape
        if text_feats is not None and vis_feats is not None:
            # Flatten spatial feats if needed
            if vis_feats.dim() == 4:
                vis_feats = vis_feats.flatten(2).transpose(1, 2) # (B, HW, C)
            
            # Simple attention: dot product
            # Just take max pool of text feats as global text rep
            global_text = text_feats.max(dim=1)[0].unsqueeze(2) # (B, C, 1)
            
            # Attn: (B, HW, C) x (B, C, 1) -> (B, HW, 1)
            attn = torch.bmm(vis_feats, global_text).squeeze(2) # (B, HW)
            
            # Normalize and reshape
            attn = F.relu(attn)
            attn = attn / (attn.max(dim=1, keepdim=True)[0] + 1e-6)
            
            H_v = int(attn.shape[1] ** 0.5)
            attn = attn.view(B, 1, H_v, H_v)
            
            # Interpolate to match mask size
            attn = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=False)
            
            # Weight is base 1.0 + amplified by boundary and text relevance
            weight = 1.0 + 2.0 * boundary * attn
        else:
            # Fallback if no features: just use boundary weight
            weight = 1.0 + 2.0 * boundary
            
        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(pred_masks, gt_masks.float(), reduction='none')
        loss = (bce * weight).mean()
        
        return loss
