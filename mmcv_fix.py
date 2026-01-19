"""
Workaround for missing MMCV CUDA extensions.
This module patches MMCV to skip loading the _ext module if unavailable and
adds a minimal CPU/GPU NMS fallback when the extension is missing.
"""

import sys
import warnings

# Patch the ext_loader to handle missing extensions gracefully
try:
    from mmcv.utils import ext_loader
    
    original_load_ext = ext_loader.load_ext

    def patched_load_ext(name, funcs):
        try:
            return original_load_ext(name, funcs)
        except (ModuleNotFoundError, ImportError) as e:
            warnings.warn(f"Failed to load mmcv extension '{name}': {e}. Continuing without it.")
            return None

    ext_loader.load_ext = patched_load_ext
except Exception as e:
    warnings.warn(f"Could not apply MMCV extension patch: {e}")
    pass

# Patch NMS to avoid NoneType errors when MMCV ops are unavailable
try:
    import importlib
    import torch
    nms_module = importlib.import_module('mmcv.ops.nms')

    if getattr(nms_module, 'ext_module', None) is None or not hasattr(nms_module.ext_module, 'nms'):
        class _ExtFallback:
            """Minimal fallback for mmcv.ops._ext with NMS support."""

            @staticmethod
            def nms(bboxes, scores, iou_threshold=0.5, offset=0):
                if bboxes.numel() == 0:
                    return bboxes.new_zeros((0,), dtype=torch.long)

                if offset:
                    bboxes = bboxes.clone()
                    bboxes[:, 2] = bboxes[:, 2] + offset
                    bboxes[:, 3] = bboxes[:, 3] + offset

                x1 = bboxes[:, 0]
                y1 = bboxes[:, 1]
                x2 = bboxes[:, 2]
                y2 = bboxes[:, 3]

                areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
                _, order = scores.sort(descending=True)

                keep = []
                while order.numel() > 0:
                    i = order[0]
                    keep.append(i)
                    if order.numel() == 1:
                        break

                    xx1 = torch.maximum(x1[i], x1[order[1:]])
                    yy1 = torch.maximum(y1[i], y1[order[1:]])
                    xx2 = torch.minimum(x2[i], x2[order[1:]])
                    yy2 = torch.minimum(y2[i], y2[order[1:]])

                    w = (xx2 - xx1).clamp(min=0)
                    h = (yy2 - yy1).clamp(min=0)
                    inter = w * h
                    union = areas[i] + areas[order[1:]] - inter
                    iou = inter / (union + 1e-6)

                    order = order[1:][iou <= iou_threshold]

                return torch.stack(keep) if keep else bboxes.new_zeros((0,), dtype=torch.long)

            @staticmethod
            def softnms(*args, **kwargs):
                raise NotImplementedError("softnms requires MMCV with CUDA extensions")

            @staticmethod
            def nms_match(*args, **kwargs):
                raise NotImplementedError("nms_match requires MMCV with CUDA extensions")

            @staticmethod
            def nms_rotated(*args, **kwargs):
                raise NotImplementedError("nms_rotated requires MMCV with CUDA extensions")

            @staticmethod
            def nms_quadri(*args, **kwargs):
                raise NotImplementedError("nms_quadri requires MMCV with CUDA extensions")

        nms_module.ext_module = _ExtFallback()
        warnings.warn("MMCV NMS extension missing; using a Python NMS fallback.")
except Exception as e:
    warnings.warn(f"Could not apply MMCV NMS fallback patch: {e}")
    pass

# Patch active_rotated_filter to handle missing extension
try:
    import importlib
    arf_module = importlib.import_module('mmcv.ops.active_rotated_filter')

    # Create a dummy implementation
    def dummy_active_rotated_filter(*args, **kwargs):
        raise NotImplementedError("active_rotated_filter requires MMCV with CUDA extensions")

    ext_module = getattr(arf_module, 'ext_module', None)
    if ext_module is None or not hasattr(ext_module, 'active_rotated_filter_forward'):
        arf_module.ext_module = None
        arf_module.active_rotated_filter = dummy_active_rotated_filter
except Exception as e:
    warnings.warn(f"Could not apply active_rotated_filter patch: {e}")
    pass
