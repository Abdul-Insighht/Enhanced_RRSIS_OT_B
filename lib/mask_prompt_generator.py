import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseMaskPromptGenerator(nn.Module):
    """
    Converts OT transport plan -> dense mask prior of shape (B, 1, 144, 144)
    for SAM3 prompt encoder.
    """
    def __init__(self, d_model=256):
        super().__init__()
        # Learnable refinement convolution
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
    
    def forward(self, P, text_mask):
        """
        P: (B, HW, seq) - the optimal transport plan mapping image pixels to text tokens
        text_mask: (B, seq) - mask for padding text tokens
        """
        B, HW, seq = P.shape
        H = W = int(HW ** 0.5)
        
        # Mask padding tokens
        if text_mask is not None:
            valid = (~text_mask).unsqueeze(1).float()  # (B, 1, seq)
            heatmap = (P * valid).sum(dim=-1)  # (B, HW)
        else:
            heatmap = P.sum(dim=-1)
            
        heatmap = heatmap.view(B, 1, H, W)
        
        # Normalize to [0, 1] per batch item
        hmax = heatmap.flatten(2).max(dim=2)[0].view(B, 1, 1, 1) + 1e-6
        heatmap = heatmap / hmax
        
        # Resize to 144x144 (SAM3 mask encoder expects 4x of 36x36 image embedding size)
        heatmap_144 = F.interpolate(heatmap, size=(144, 144), 
                                     mode='bilinear', align_corners=False)
        
        # Refine (learnable conv)
        refined = self.refine(heatmap_144)
        return refined  # (B, 1, 144, 144)
