import numpy as np
import torch


def safe_cast_back(arr, ref):
    """
    Safely cast `arr` (torch.Tensor or np.ndarray, typically float32)
    back to the dtype of `ref` (torch.Tensor or np.ndarray).
    Clipping is applied to avoid overflow.
    """
    if not isinstance(ref, (np.ndarray, torch.Tensor)):
        raise TypeError(f"Unsupported reference type: {type(ref)}")

    target_dtype = ref.dtype

    if isinstance(ref, torch.Tensor):
        arr = torch.as_tensor(arr, device=ref.device)
        if torch.is_floating_point(ref):
            return arr.to(dtype=target_dtype)
        elif target_dtype == torch.uint8:
            arr = arr + max(-arr.min() + 1, 0)
            if arr.max() > 255:
                arr = arr / arr.max() * 255
            return torch.clamp(arr, 0, 255).to(dtype=target_dtype)
        elif target_dtype == torch.uint16:
            arr = arr + max(-arr.min() + 1, 0)
            if arr.max() > 65535:
                arr = arr / arr.max() * 65535
            return torch.clamp(arr, 0, 65535).to(dtype=target_dtype)
        elif target_dtype == torch.int16:
            return torch.clamp(arr, -32768, 32767).to(dtype=target_dtype)
        elif target_dtype == torch.int32:
            return torch.clamp(arr, -2147483648, 2147483647).to(dtype=target_dtype)
        elif target_dtype == torch.int64:
            return torch.clamp(arr, -9223372036854775808, 9223372036854775807).to(
                dtype=target_dtype
            )
        else:
            raise ValueError(f"Unsupported torch dtype: {target_dtype}")

    else:
        arr = np.asarray(arr)
        if np.issubdtype(target_dtype, np.floating):
            return arr.astype(target_dtype)
        elif target_dtype == np.uint8:
            arr = arr + max(-arr.min() + 1, 0)
            if arr.max() > 255:
                arr = arr / arr.max() * 255
            return np.clip(arr, 0, 255).astype(np.uint8)
        elif target_dtype == np.uint16:
            arr = arr + max(-arr.min() + 1, 0)
            if arr.max() > 65535:
                arr = arr / arr.max() * 65535
            return np.clip(arr, 0, 65535).astype(np.uint16)
        elif target_dtype == np.int16:
            return np.clip(arr, -32768, 32767).astype(np.int16)
        elif target_dtype == np.int32:
            return np.clip(arr, -2147483648, 2147483647).astype(np.int32)
        elif target_dtype == np.int64:
            return np.clip(arr, -9223372036854775808, 9223372036854775807).astype(
                np.int64
            )
        else:
            raise ValueError(f"Unsupported numpy dtype: {target_dtype}")
