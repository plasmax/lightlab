from __future__ import annotations
import random
import os, glob
from typing import Dict, Iterable, List, Optional, Tuple
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset


class LightLabDataset(Dataset):
    """
    Builds samples of:
        inp   = [depth_L, mask_L, normal_RGB, light_off_RGB] (C=1+1+3+3=8 channels)
        target= light_only_RGB
    Returns: (inp, target, idx, paths_dict)
    """
    def __init__(
        self,
        patterns: Dict[str, str],
        frame_range: Optional[Tuple[int, int]] = None,  # e.g. (1001, 1458), inclusive
        required_keys: Iterable[str] = ("depth", "mask", "normal", "off", "only"),
        strict: bool = True,   # if True: only frames where ALL files exist
    ):
        """
        patterns keys expected:
            "normal", "depth", "only", "off", "mask"
        values are printf-style patterns with %04d, e.g. "....%04d.jpeg"
        """
        self.patterns = patterns
        self.required_keys = tuple(required_keys)
        self.strict = strict

        # Precompile ToTensor
        self.to_tensor = transforms.ToTensor()

        # Discover candidate frames
        if frame_range is None:
            # auto-discover from the 'only' (or any available) pattern, then intersect across all required keys
            idx_sets: List[set] = []
            for k in self.required_keys:
                pat = self.patterns[k]
                # Expand by globbing the numeric slot
                # Turn "...%04d.jpeg" into "...[0-9][0-9][0-9][0-9].jpeg"
                gpat = pat.replace("%04d", "[0-9][0-9][0-9][0-9]")
                indices = set()
                for p in glob.glob(gpat):
                    base = os.path.basename(p)
                    # extract the number from the last dot-separated token (before extension)
                    # safer: read digits before extension
                    stem, ext = os.path.splitext(base)
                    num_str = ''.join(ch for ch in stem.split('.')[-1] if ch.isdigit())
                    if len(num_str) == 4:  # what we expect here
                        indices.add(int(num_str))
                idx_sets.append(indices)
            candidate = set.intersection(*idx_sets) if idx_sets else set()
            self.indices = sorted(candidate)
        else:
            start, end = frame_range
            self.indices = list(range(start, end + 1))

        # If strict, filter to frames where *all* required files exist
        if self.strict:
            filtered = []
            for i in self.indices:
                if all(os.path.exists(self.patterns[k] % i) for k in self.required_keys):
                    filtered.append(i)
            self.indices = filtered

        if not self.indices:
            raise RuntimeError("No frames found that satisfy the provided patterns/range.")

    def __len__(self) -> int:
        return len(self.indices)

    def _read(self, path: str, mode: str) -> Image.Image:
        img = Image.open(path)
        if mode is not None:
            img = img.convert(mode)
        return img

    def __getitem__(self, i: int):
        frame = self.indices[i]

        paths = {
            "normal": self.patterns["normal"] % frame,
            "depth":  self.patterns["depth"]  % frame,
            "only":   self.patterns["only"]   % frame,
            "off":    self.patterns["off"]    % frame,
            "mask":   self.patterns["mask"]   % frame,
        }

        # Load
        depth     = self._read(paths["depth"],  "L")   # 1ch
        mask      = self._read(paths["mask"],   "L")   # 1ch
        normal    = self._read(paths["normal"], "RGB") # 3ch
        light_off = self._read(paths["off"],    "RGB") # 3ch
        light_only= self._read(paths["only"],   "RGB") # 3ch (target)

        # To tensors (C,H,W)
        t_depth      = self.to_tensor(depth)
        t_mask       = self.to_tensor(mask)
        t_normal     = self.to_tensor(normal)
        t_light_off  = self.to_tensor(light_off)
        t_light_only = self.to_tensor(light_only)

        # Stack input
        inp = torch.cat([t_depth, t_mask, t_normal, t_light_off], dim=0)  # (8,H,W)

        return inp, t_light_only, f"ll_light_only.{frame}"



def split_dataset(dataset, val_fraction=0.01, seed=42):
    n = len(dataset)
    val_count = max(1, int(n * val_fraction))
    all_indices = list(range(n))
    random.seed(seed)
    val_indices = set(random.sample(all_indices, val_count))
    train_indices = list(set(all_indices) - val_indices)
    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, list(val_indices)),
    )


if __name__ == "__main__":
    # test

    from file_paths import patterns
    """
    # example expected patterns
    patterns = {
        "normal": "/path/to/ll_normal.%04d.jpeg",
        "depth":  "/path/to/ll_depth.%04d.jpeg",
        "only":   "/path/to/ll_light_only.%04d.jpeg",
        "off":    "/path/to/ll_light_off.%04d.jpeg",
        "mask":   "/path/to/ll_light_mask.%04d.jpeg",
    }
    """

    # Pin to 1001–1458 explicitly:
    ds = LightLabDataset(patterns, frame_range=(1001, 1458), strict=True)

    # Or auto-discover across the five patterns:
    # ds = LightLabDataset(patterns, frame_range=None, strict=True)

    inp, target, path = ds[5]
    print(inp.shape, target.shape, path)
    # -> torch.Size([8, 256, 256]) torch.Size([3, 256, 256]) sandbox_mlast_precomp_ll_light_only_v001_main.1006.jpeg
