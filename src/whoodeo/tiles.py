
import torch
import torch.nn.functional as F

class Tiles():
    def __init__(self, device, tile_size: int, overlap: int, frame_width: int, frame_height: int):
        assert overlap < tile_size, "overlap must be less than tile size"

        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.step = self.tile_size - self.overlap

    def dissect(self, frame: torch.Tensor):
        tiles = []

        for top in range(0, self.frame_height, self.step):
            for left in range(0, self.frame_width, self.step):
                # Calculate effective dimensions within frame boundaries
                eff_h = min(self.tile_size, self.frame_height - top)
                eff_w = min(self.tile_size, self.frame_width - left)

                # Extract tile
                tile = frame[:, top:top+eff_h, left:left+eff_w]

                # Pad tile to required size if it extends beyond frame
                pad_bottom = self.tile_size - eff_h
                pad_right = self.tile_size - eff_w
                if pad_bottom > 0 or pad_right > 0:
                    tile = F.pad(tile, (0, pad_right, 0, pad_bottom), mode='constant', value=0)

                tiles.append(tile)

        return torch.stack(tiles)

    def reconstruct(self, tiles: torch.Tensor):
        # Initialize output frame and count tensor for averaging overlaps
        full_img = torch.zeros(tiles.shape[1], self.frame_height, self.frame_width, device=self.device)
        count = torch.zeros(self.frame_height, self.frame_width, device=self.device)
        tile_idx = 0

        for top in range(0, self.frame_height, self.step):
            for left in range(0, self.frame_width, self.step):
                # Calculate effective tile size within frame boundaries
                eff_h = min(self.tile_size, self.frame_height - top)
                eff_w = min(self.tile_size, self.frame_width - left)

                # Add tile to the full image (only the effective region)
                full_img[:, top:top + eff_h, left:left + eff_w] += tiles[tile_idx, :, :eff_h, :eff_w]
                # Increment count for averaging overlapping regions
                count[top:top + eff_h, left:left + eff_w] += 1
                tile_idx += 1

        # Average overlapping regions
        full_img /= count[None, :, :]
        return full_img
