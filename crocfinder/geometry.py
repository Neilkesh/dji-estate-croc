"""360 frame -> perspective tiles.

World axes: x right, y down, z forward. Yaw turns right, pitch < 0 looks down.
A tile is a pinhole view; for each tile pixel we compute a ray, then ask the
source model (equirect or dual fisheye) which source pixel that ray comes from.
The result is a sampling grid for torch.nn.functional.grid_sample.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Tile:
    yaw: float    # degrees, 0 = forward, positive = right
    pitch: float  # degrees, 0 = horizon, -90 = straight down
    hfov: float   # degrees
    size: int     # square tile side in pixels


@dataclass(frozen=True)
class Source:
    """How the 360 image is laid out in the decoded frame."""
    kind: str = "equirect"        # "equirect", "dual_fisheye" or "fisheye_down"
    # fisheye_down: one lens looking straight down that fills the whole frame
    # (DJI Avata 360 .OSV, video stream 1). Image top = drone forward when roll_front = 0.
    lens_fov: float = 190.0       # dual_fisheye: field of view of each lens, degrees
    # dual_fisheye: front lens is the left half of the frame, back lens the right half
    swap: bool = False            # back lens on the left instead
    roll_front: float = 0.0       # in-plane rotation of each lens image, degrees
    roll_back: float = 0.0
    radius_frac: float = 1.0      # image-circle radius as a fraction of half the lens image height


def default_tiles(hfov: float = 48.0, size: int = 1024, top_pitch: float = -5.0,
                  overlap: float = 0.2) -> list[Tile]:
    """Rings of tiles from just below the horizon down to straight down.

    hfov/size are chosen so tile pixel density is close to an 8K 360 frame
    (about 21 px per degree), i.e. no downscaling of small subjects.
    """
    step = hfov * (1 - overlap)
    n_rings = math.ceil((90 + top_pitch) / step)      # intervals between top ring and nadir
    spacing = (90 + top_pitch) / n_rings
    tiles: list[Tile] = []
    for r in range(n_rings):
        pitch = top_pitch - r * spacing
        # a ring's circumference shrinks with cos(pitch); size it at the ring's upper edge
        edge = math.cos(math.radians(min(0.0, pitch + hfov / 2)))
        n = max(1, math.ceil(360 * edge / step))
        tiles += [Tile(yaw=i * 360 / n - 180, pitch=pitch, hfov=hfov, size=size) for i in range(n)]
    tiles.append(Tile(yaw=0.0, pitch=-90.0, hfov=hfov, size=size))
    return tiles


def tile_rays(t: Tile) -> torch.Tensor:
    """Unit rays for every tile pixel, shape (size, size, 3)."""
    f = (t.size / 2) / math.tan(math.radians(t.hfov) / 2)
    c = torch.arange(t.size, dtype=torch.float32) + 0.5 - t.size / 2
    y, x = torch.meshgrid(c, c, indexing="ij")
    d = torch.stack([x, y, torch.full_like(x, f)], -1)
    d = d / d.norm(dim=-1, keepdim=True)
    p, w = math.radians(t.pitch), math.radians(t.yaw)
    # pitch about x (negative pitch tilts the view toward +y = down), then yaw about y
    rx = torch.tensor([[1, 0, 0],
                       [0, math.cos(p), -math.sin(p)],
                       [0, math.sin(p), math.cos(p)]], dtype=torch.float32)
    ry = torch.tensor([[math.cos(w), 0, math.sin(w)],
                       [0, 1, 0],
                       [-math.sin(w), 0, math.cos(w)]], dtype=torch.float32)
    return d @ rx.T @ ry.T


def rays_to_source(d: torch.Tensor, src: Source) -> torch.Tensor:
    """Rays (...,3) -> normalised source coords (...,2) in [-1, 1] for grid_sample."""
    x, y, z = d.unbind(-1)
    if src.kind == "equirect":
        lon = torch.atan2(x, z)                    # -pi..pi, 0 = forward
        lat = torch.asin((-y).clamp(-1, 1))        # +pi/2 = up
        return torch.stack([lon / math.pi, -lat / (math.pi / 2)], -1)
    if src.kind == "dual_fisheye":
        front = z >= 0
        # in the back lens's own frame: it looks along -z and its right is world -x
        lx = torch.where(front, x, -x)
        lz = torch.where(front, z, -z)
        theta = torch.acos(lz.clamp(-1, 1))        # angle off the lens axis
        phi = torch.atan2(y, lx)
        roll = torch.where(front, torch.tensor(math.radians(src.roll_front)),
                           torch.tensor(math.radians(src.roll_back)))
        phi = phi + roll
        r = theta / math.radians(src.lens_fov / 2) * src.radius_frac  # 1.0 = edge of circle
        u, v = r * torch.cos(phi), r * torch.sin(phi)   # -1..1 inside that lens's half
        left = front ^ src.swap
        gx = torch.where(left, u * 0.5 - 0.5, u * 0.5 + 0.5)
        return torch.stack([gx, v], -1)
    if src.kind == "fisheye_down":
        theta = torch.acos(y.clamp(-1, 1))             # angle off straight down
        phi = torch.atan2(-z, x) + math.radians(src.roll_front)
        r = theta / math.radians(src.lens_fov / 2) * src.radius_frac
        return torch.stack([r * torch.cos(phi), r * torch.sin(phi)], -1)
    raise ValueError(f"unknown source kind {src.kind!r}")


def build_grids(tiles: list[Tile], src: Source, device: str = "cpu") -> torch.Tensor:
    """Sampling grids for all tiles, shape (n_tiles, size, size, 2)."""
    return torch.stack([rays_to_source(tile_rays(t), src) for t in tiles]).to(device)


def reproject(frame: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
    """frame (3,H,W) float -> tiles (n,3,size,size)."""
    n = grids.shape[0]
    return torch.nn.functional.grid_sample(
        frame.unsqueeze(0).expand(n, -1, -1, -1), grids,
        mode="bilinear", padding_mode="border", align_corners=False)


def pixel_direction(t: Tile, px: float, py: float) -> tuple[float, float]:
    """Tile pixel -> (yaw, pitch) in degrees in the 360 frame. Used for reporting."""
    f = (t.size / 2) / math.tan(math.radians(t.hfov) / 2)
    d = torch.tensor([px - t.size / 2, py - t.size / 2, f], dtype=torch.float32)
    d = d / d.norm()
    p, w = math.radians(t.pitch), math.radians(t.yaw)
    x, y, z = d.tolist()
    y, z = y * math.cos(p) - z * math.sin(p), y * math.sin(p) + z * math.cos(p)
    x, z = x * math.cos(w) + z * math.sin(w), -x * math.sin(w) + z * math.cos(w)
    return math.degrees(math.atan2(x, z)), math.degrees(math.asin(max(-1.0, min(1.0, -y))))


def angular_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle angle in degrees between two (yaw, pitch) directions."""
    ya, pa, yb, pb = map(math.radians, (*a, *b))
    c = math.sin(pa) * math.sin(pb) + math.cos(pa) * math.cos(pb) * math.cos(ya - yb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))
