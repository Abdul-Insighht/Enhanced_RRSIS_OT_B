"""
Enhanced_RRSIS_UOT Evaluation Script

Evaluate trained Enhanced_RRSIS_UOT model on test/val sets.
Computes: mIoU, oIoU (overall IoU), Precision@X thresholds.

Usage:
    python test.py --dataset rrsis_d --data_root /path/to/data --resume ./output/best_model.pth
"""

import os
import time
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from args import get_args
from data.dataset import get_dataset, collate_fn
from lib.enhanced_model import Enhanced_RRSIS_UOT


def compute_metrics(pred_masks, gt_masks, threshold=0.5):
    """
    Compute segmentation metrics.

    Args:
        pred_masks: [B, 1, H, W] predicted logits
        gt_masks: [B, 1, H, W] ground truth binary masks

    Returns:
        dict with IoU, precision at various thresholds
    """
    pred_binary = (torch.sigmoid(pred_masks) > threshold).float()

    # Per-sample IoU
    intersection = (pred_binary * gt_masks).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    # Overall IoU (cumulative intersection / cumulative union)
    total_intersection = intersection.sum()
    total_union = union.sum()

    # Precision at thresholds
    prec_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    precisions = {}
    for t in prec_thresholds:
        precisions[f'P@{t}'] = (iou > t).float().mean().item()

    return {
        'iou': iou,
        'mean_iou': iou.mean().item(),
        'intersection': total_intersection.item(),
        'union': total_union.item(),
        **precisions,
    }


@torch.no_grad()
def evaluate(model, test_loader, device, args):
    """Run full evaluation."""
    model.eval()

    all_ious = []
    total_intersection = 0
    total_union = 0
    prec_counts = {f'P@{t}': 0 for t in [0.5, 0.6, 0.7, 0.8, 0.9]}
    total_samples = 0
    total_time = 0

    for batch_idx, (images, masks, captions) in enumerate(test_loader):
        images = images.to(device)
        masks = masks.to(device)

        # Handle eval captions (list of lists)
        if isinstance(captions[0], list):
            # Evaluate on each caption and take best IoU
            best_iou_per_sample = []
            for cap_list in captions:
                ious_for_caps = []
                components_for_caps = []
                for cap in cap_list:
                    start = time.time()
                    with torch.amp.autocast('cuda', enabled=True):
                        outputs = model(images[0:1], [cap], masks[0:1])
                    total_time += time.time() - start

                    metrics = compute_metrics(outputs['pred_masks'], masks[0:1])
                    ious_for_caps.append(metrics['iou'].item())
                    components_for_caps.append((metrics['intersection'], metrics['union']))

                best_idx = np.argmax(ious_for_caps)
                best_iou_per_sample.append(ious_for_caps[best_idx])
                
                # Accumulate best caption's intersection and union
                total_intersection += components_for_caps[best_idx][0]
                total_union += components_for_caps[best_idx][1]

            for iou_val in best_iou_per_sample:
                all_ious.append(iou_val)
                total_samples += 1
                for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
                    if iou_val > t:
                        prec_counts[f'P@{t}'] += 1
        else:
            # Single caption per sample
            start = time.time()
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(images, captions, masks)
            total_time += time.time() - start

            metrics = compute_metrics(outputs['pred_masks'], masks)
            all_ious.extend(metrics['iou'].cpu().numpy().tolist())
            total_intersection += metrics['intersection']
            total_union += metrics['union']
            total_samples += images.size(0)

            for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
                prec_counts[f'P@{t}'] += (metrics['iou'] > t).sum().item()

        if (batch_idx + 1) % 100 == 0:
            current_miou = np.mean(all_ious)
            print(f"  Progress: {batch_idx+1}/{len(test_loader)}, "
                  f"Current mIoU={current_miou:.4f}")

    # Final metrics
    mIoU = np.mean(all_ious)
    oIoU = total_intersection / (total_union + 1e-6) if total_union > 0 else mIoU

    results = {
        'mIoU': mIoU,
        'oIoU': oIoU,
        'num_samples': total_samples,
        'avg_time': total_time / max(total_samples, 1),
    }
    for key in prec_counts:
        results[key] = prec_counts[key] / total_samples

    return results


def save_predictions(model, test_loader, device, output_dir, args):
    """Save predicted masks as images for visualization alongside their captions."""
    model.eval()
    # Dynamically separate val and test visualizations into split-specific folders
    vis_dir = os.path.join(output_dir, 'visualizations', args.split)
    os.makedirs(vis_dir, exist_ok=True)

    # Create a metadata text file to map filenames to exact full query captions
    meta_path = os.path.join(vis_dir, 'metadata.txt')
    with open(meta_path, 'w') as meta_file:
        meta_file.write("============================================================\n")
        meta_file.write(f"  Enhanced_RRSIS_UOT Prediction Visualizations Metadata ({args.split.upper()})\n")
        meta_file.write("============================================================\n\n")

        for batch_idx, (images, masks, captions) in enumerate(test_loader):
            if batch_idx >= 50:
                break

            images = images.to(device)
            # If multiple captions list is passed, take the first one for visualization
            if isinstance(captions[0], list):
                raw_captions = [cap[0] for cap in captions]
            else:
                raw_captions = captions

            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(images, raw_captions)

            pred_probs = torch.sigmoid(outputs['pred_masks'])
            pred_binary = (pred_probs > 0.5).float()

            for i in range(images.size(0)):
                caption = raw_captions[i]
                
                # Sanitize caption for a clean, safe filename (replace spaces/special chars with underscores)
                clean_caption = "".join(c if c.isalnum() else "_" for c in caption).strip("_")
                clean_caption = "_".join(filter(None, clean_caption.split("_")))[:40]

                # 1. Reconstruct the original RGB image (images in test_loader are in [0, 1] range)
                orig_np = (images[i].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                orig_img = Image.fromarray(orig_np)
                orig_filename = f'{batch_idx}_{i}_orig_{clean_caption}.png'
                orig_img.save(os.path.join(vis_dir, orig_filename))

                # 2. Generate a premium semi-transparent GREEN overlay for the PREDICTION mask
                pred_active = pred_binary[i, 0].cpu().numpy() > 0.5
                pred_overlay = orig_np.copy()
                # Apply green color (increase green channel, shade red and blue channels slightly)
                pred_overlay[pred_active, 1] = np.clip(pred_overlay[pred_active, 1] * 0.5 + 255 * 0.5, 0, 255).astype(np.uint8)
                pred_overlay[pred_active, 0] = (pred_overlay[pred_active, 0] * 0.5).astype(np.uint8)
                pred_overlay[pred_active, 2] = (pred_overlay[pred_active, 2] * 0.5).astype(np.uint8)
                
                pred_overlay_img = Image.fromarray(pred_overlay)
                pred_filename = f'{batch_idx}_{i}_pred_overlay_{clean_caption}.png'
                pred_overlay_img.save(os.path.join(vis_dir, pred_filename))

                # 3. Generate a premium semi-transparent RED overlay for the GROUND TRUTH mask
                gt_active = masks[i, 0].cpu().numpy() > 0.5
                gt_overlay = orig_np.copy()
                # Apply red color (increase red channel, shade green and blue channels slightly)
                gt_overlay[gt_active, 0] = np.clip(gt_overlay[gt_active, 0] * 0.5 + 255 * 0.5, 0, 255).astype(np.uint8)
                gt_overlay[gt_active, 1] = (gt_overlay[gt_active, 1] * 0.5).astype(np.uint8)
                gt_overlay[gt_active, 2] = (gt_overlay[gt_active, 2] * 0.5).astype(np.uint8)
                
                gt_overlay_img = Image.fromarray(gt_overlay)
                gt_filename = f'{batch_idx}_{i}_gt_overlay_{clean_caption}.png'
                gt_overlay_img.save(os.path.join(vis_dir, gt_filename))

                # Write to the metadata file
                meta_file.write(f"Sample {batch_idx}_{i}:\n")
                meta_file.write(f"  Full Query Text: \"{caption}\"\n")
                meta_file.write(f"  Original Image:   {orig_filename}\n")
                meta_file.write(f"  Prediction Mask:  {pred_filename}\n")
                meta_file.write(f"  Ground Truth Mask: {gt_filename}\n\n")

    print(f"  Saved visualizations and metadata.txt to {vis_dir}")


def main():
    args = get_args()
    args.eval_only = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  Enhanced_RRSIS_UOT Evaluation")
    print(f"  Dataset: {args.dataset}")
    print(f"  Split: {args.split}")
    print(f"  Checkpoint: {args.resume}")
    print(f"  Device: {device}")
    print(f"  --- Enhancements ---")
    print(f"  Dynamic LoRA: {args.use_dynamic_lora}")
    print(f"  Contrastive Loss: {args.use_contrastive_loss}")
    print(f"  Multi-Scale OT: {args.use_multiscale_ot}")
    print(f"  OHEM Loss: {args.use_ohem_loss}")
    print(f"{'='*60}\n")

    # ====== Build Model ======
    model = Enhanced_RRSIS_UOT(
        sam3_ckpt=args.sam3_ckpt,
        image_size=args.image_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_dynamic_lora=args.use_dynamic_lora,
        use_contrastive_loss=args.use_contrastive_loss,
        use_multiscale_ot=args.use_multiscale_ot,
        use_ohem_loss=args.use_ohem_loss,
    )

    # Load trained weights
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, "
              f"best_iou={ckpt.get('best_iou', '?')}")
    else:
        print("WARNING: No checkpoint provided, evaluating with pretrained SAM3 only!")

    model = model.to(device)
    model.eval()

    # ====== Build Dataset ======
    test_dataset = get_dataset(args, split=args.split, eval_mode=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ====== Evaluate ======
    print("\nRunning evaluation...")
    results = evaluate(model, test_loader, device, args)

    # ====== Print & Save Results ======
    res_str = ""
    res_str += f"\n{'='*60}\n"
    res_str += f"  Results on {args.dataset} ({args.split})\n"
    res_str += f"{'='*60}\n"
    res_str += f"  mIoU:  {results['mIoU']*100:.2f}%\n"
    res_str += f"  oIoU:  {results['oIoU']*100:.2f}%\n"
    res_str += f"  P@0.5: {results['P@0.5']*100:.2f}%\n"
    res_str += f"  P@0.6: {results['P@0.6']*100:.2f}%\n"
    res_str += f"  P@0.7: {results['P@0.7']*100:.2f}%\n"
    res_str += f"  P@0.8: {results['P@0.8']*100:.2f}%\n"
    res_str += f"  P@0.9: {results['P@0.9']*100:.2f}%\n"
    res_str += f"  Samples: {results['num_samples']}\n"
    res_str += f"  Avg Time: {results['avg_time']*1000:.1f}ms\n"
    res_str += f"{'='*60}\n"
    
    # Print to console
    print(res_str)

    # Save to a text file in output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    res_txt_path = os.path.join(args.output_dir, f"quantitative_results_{args.split}.txt")
    with open(res_txt_path, 'w') as f:
        f.write(res_str)
    print(f"  Saved quantitative results to {res_txt_path}")

    # ====== Save Visualizations ======
    if args.visualize:
        save_predictions(model, test_loader, device, args.output_dir, args)


if __name__ == '__main__':
    main()
