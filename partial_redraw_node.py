import torch
import torch.nn.functional as F


class PartialRedrawUp:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "imageA": ("IMAGE",),
                "mask": ("MASK",),
                "context_expand_factor": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider"
                }),
                "force_ratio": (["Auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOX")
    RETURN_NAMES = ("imageB", "crop_box")

    FUNCTION = "calculate_crop"
    CATEGORY = "🍎MYAPI"

    def calculate_crop(self, imageA, mask, context_expand_factor, force_ratio):
        RATIOS = {
            "1:1": 1.0, "2:3": 2 / 3, "3:2": 3 / 2, "3:4": 3 / 4, "4:3": 4 / 3,
            "4:5": 4 / 5, "5:4": 5 / 4, "9:16": 9 / 16, "16:9": 16 / 9, "21:9": 21 / 9
        }

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        orig_h, orig_w = imageA.shape[1], imageA.shape[2]

        rows, cols = torch.where(mask[0] > 0.5)
        if len(rows) == 0:
            min_y, max_y = orig_h // 4, orig_h * 3 // 4
            min_x, max_x = orig_w // 4, orig_w * 3 // 4
        else:
            min_y, max_y = rows.min().item(), rows.max().item()
            min_x, max_x = cols.min().item(), cols.max().item()

        box_w = max_x - min_x
        box_h = max_y - min_y
        pad_w = int(box_w * context_expand_factor)
        pad_h = int(box_h * context_expand_factor)

        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        target_w = box_w + (pad_w * 2)
        target_h = box_h + (pad_h * 2)

        current_ratio = target_w / max(1, target_h)
        chosen_ratio_name = "1:1"
        target_ratio_val = 1.0

        if force_ratio != "Auto":
            chosen_ratio_name = force_ratio
            target_ratio_val = RATIOS[force_ratio]
        else:
            closest_diff = float('inf')
            for name, val in RATIOS.items():
                diff = abs(val - current_ratio)
                if diff < closest_diff:
                    closest_diff = diff
                    chosen_ratio_name = name
                    target_ratio_val = val

        if (target_w / target_h) > target_ratio_val:
            new_h = target_w / target_ratio_val
            new_w = target_w
        else:
            new_w = target_h * target_ratio_val
            new_h = target_h

        final_w = int(new_w)
        final_h = int(new_h)

        x1 = center_x - (final_w // 2)
        y1 = center_y - (final_h // 2)
        x2 = x1 + final_w
        y2 = y1 + final_h

        if x1 < 0: x2 += abs(x1); x1 = 0
        if y1 < 0: y2 += abs(y1); y1 = 0
        if x2 > orig_w: x1 -= (x2 - orig_w); x2 = orig_w
        if y2 > orig_h: y1 -= (y2 - orig_h); y2 = orig_h

        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(orig_w, int(x2)), min(orig_h, int(y2))

        real_w = x2 - x1
        real_h = y2 - y1

        imageB = imageA[:, y1:y2, x1:x2, :]
        crop_box = [x1, y1, real_w, real_h]

        return (imageB, crop_box)


class PartialRedrawDown:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "imageB": ("IMAGE",),
                "AIimage": ("IMAGE",),
                "imageA": ("IMAGE",),
                "crop_box": ("BOX",),
                "feather": ("INT", {"default": 16, "min": 0, "max": 200, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("final_image",)
    FUNCTION = "resize_and_paste"
    CATEGORY = "🍎MYAPI"

    def resize_and_paste(self, imageB, AIimage, imageA, crop_box, feather):
        x, y, target_w, target_h = crop_box

        target_h_ref = AIimage.shape[1]
        target_w_ref = AIimage.shape[2]

        src_h = imageB.shape[1]
        src_w = imageB.shape[2]

        target_long = max(target_w_ref, target_h_ref)
        src_long = max(src_w, src_h)

        scale = target_long / src_long

        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))

        img_tensor = imageB.permute(0, 3, 1, 2)

        resized_tensor = F.interpolate(
            img_tensor,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False,
            antialias=True
        )

        canvas = torch.zeros((img_tensor.shape[0], img_tensor.shape[1], target_h_ref, target_w_ref),
                             dtype=resized_tensor.dtype, device=resized_tensor.device)

        src_x_start = max(0, (new_w - target_w_ref) // 2)
        src_y_start = max(0, (new_h - target_h_ref) // 2)
        src_x_end = min(new_w, src_x_start + target_w_ref)
        src_y_end = min(new_h, src_y_start + target_h_ref)

        dst_x_start = max(0, (target_w_ref - new_w) // 2)
        dst_y_start = max(0, (target_h_ref - new_h) // 2)
        dst_x_end = min(target_w_ref, dst_x_start + (src_x_end - src_x_start))
        dst_y_end = min(target_h_ref, dst_y_start + (src_y_end - src_y_start))

        canvas[:, :, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            resized_tensor[:, :, src_y_start:src_y_end, src_x_start:src_x_end]

        resized_image = canvas.permute(0, 2, 3, 1)

        paste_image = resized_image

        if paste_image.shape[1] != target_h or paste_image.shape[2] != target_w:
            img_tensor = paste_image.permute(0, 3, 1, 2)
            paste_image = F.interpolate(
                img_tensor,
                size=(target_h, target_w),
                mode='bicubic',
                align_corners=False
            ).permute(0, 2, 3, 1)

        output_image = imageA.clone()

        if output_image.shape[0] != paste_image.shape[0]:
            if paste_image.shape[0] == 1:
                paste_image = paste_image.repeat(output_image.shape[0], 1, 1, 1)
            else:
                paste_image = paste_image[0].unsqueeze(0)

        if paste_image.shape[3] == 4:
            src_rgb = paste_image[:, :, :, :3]
            src_alpha = paste_image[:, :, :, 3:4]
        else:
            src_rgb = paste_image
            src_alpha = torch.ones((paste_image.shape[0], target_h, target_w, 1),
                                   dtype=paste_image.dtype, device=paste_image.device)

        mask_h, mask_w = target_h, target_w
        if feather > 0:
            Y = torch.linspace(0, mask_h - 1, mask_h, device=paste_image.device).view(mask_h, 1).repeat(1, mask_w)
            X = torch.linspace(0, mask_w - 1, mask_w, device=paste_image.device).view(1, mask_w).repeat(mask_h, 1)

            dist_top = Y
            dist_bottom = mask_h - 1 - Y
            dist_left = X
            dist_right = mask_w - 1 - X

            min_dist = torch.min(torch.min(dist_top, dist_bottom), torch.min(dist_left, dist_right))
            geom_mask = torch.clamp(min_dist / feather, 0.0, 1.0).unsqueeze(0).unsqueeze(-1)
        else:
            geom_mask = torch.ones((1, mask_h, mask_w, 1), dtype=paste_image.dtype, device=paste_image.device)

        geom_mask = geom_mask.to(output_image.device)
        src_rgb = src_rgb.to(output_image.device)
        src_alpha = src_alpha.to(output_image.device)

        final_mask = geom_mask * src_alpha

        bg_slice = output_image[:, y:y + target_h, x:x + target_w, :]

        if bg_slice.shape[3] == 4:
            bg_rgb = bg_slice[:, :, :, :3]
            bg_a = bg_slice[:, :, :, 3:4]

            blended_rgb = src_rgb * final_mask + bg_rgb * (1.0 - final_mask)
            blended_slice = torch.cat((blended_rgb, bg_a), dim=3)
        else:
            blended_slice = src_rgb * final_mask + bg_slice * (1.0 - final_mask)

        output_image[:, y:y + target_h, x:x + target_w, :] = blended_slice

        return (output_image,)


NODE_CLASS_MAPPINGS = {
    "PartialRedrawUp": PartialRedrawUp,
    "PartialRedrawDown": PartialRedrawDown
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PartialRedrawUp": "🎨局部重绘上",
    "PartialRedrawDown": "🎨局部重绘下"
}
