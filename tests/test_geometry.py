import math

import torch

from crocfinder.geometry import (Source, Tile, angular_distance, build_grids, default_tiles,
                                 pixel_direction, rays_to_source, reproject, tile_rays)


def test_tile_centre_ray_points_where_the_tile_looks():
    for yaw, pitch in [(0, 0), (90, 0), (-135, -40), (30, -90)]:
        t = Tile(yaw, pitch, 48, 64)
        d = tile_rays(t)[31:33, 31:33].mean((0, 1))
        d = d / d.norm()
        want = (math.cos(math.radians(pitch)) * math.sin(math.radians(yaw)),
                -math.sin(math.radians(pitch)),
                math.cos(math.radians(pitch)) * math.cos(math.radians(yaw)))
        assert torch.allclose(d, torch.tensor(want), atol=1e-4), (yaw, pitch, d)


def test_equirect_mapping_known_points():
    eq = Source("equirect")
    f = lambda v: rays_to_source(torch.tensor(v, dtype=torch.float32), eq).tolist()
    assert f([0, 0, 1]) == [0, 0]                       # forward = image centre
    assert abs(f([1, 0, 0])[0] - 0.5) < 1e-6            # right = 3/4 across
    assert abs(f([0, 1, 0])[1] - 1.0) < 1e-6            # down = bottom row


def test_dual_fisheye_lens_axes_hit_lens_centres():
    fe = Source("dual_fisheye")
    f = lambda v: rays_to_source(torch.tensor(v, dtype=torch.float32), fe).tolist()
    assert f([0, 0, 1]) == [-0.5, 0]                    # front lens centre = left half centre
    assert f([0, 0, -1]) == [0.5, 0]                    # back lens centre = right half centre


def test_a_marked_spot_lands_in_the_right_tile_pixel():
    # paint a dot in an equirect frame at yaw 40, pitch -30 and find it again in a tile
    h, w = 512, 1024
    frame = torch.zeros(3, h, w)
    yaw, pitch = 40.0, -30.0
    u = int((yaw / 360 + 0.5) * w)
    v = int((0.5 - pitch / 180) * h)
    frame[:, v - 2:v + 3, u - 2:u + 3] = 255
    t = Tile(yaw=30, pitch=-35, hfov=48, size=256)
    tile = reproject(frame, build_grids([t], Source("equirect")))[0, 0]
    ys, xs = torch.nonzero(tile > 128, as_tuple=True)
    assert len(ys) > 0
    got = pixel_direction(t, xs.float().mean().item(), ys.float().mean().item())
    assert angular_distance(got, (yaw, pitch)) < 1.0, got


def test_default_tiles_cover_everything_below_the_horizon():
    tiles = default_tiles()
    for yaw in range(-180, 180, 7):
        for pitch in range(-90, -4, 5):
            # covered if within the inscribed cone of some tile
            assert any(angular_distance((yaw, pitch), (t.yaw, t.pitch)) < t.hfov / 2 for t in tiles), (yaw, pitch)


def test_fisheye_down_centre_is_straight_down_and_horizon_is_near_the_rim():
    fe = Source("fisheye_down", lens_fov=180, radius_frac=1.0)
    f = lambda v: rays_to_source(torch.tensor(v, dtype=torch.float32), fe)
    assert f([0, 1, 0]).abs().max() < 1e-6              # straight down = image centre
    assert abs(f([0, 0, 1]).norm().item() - 1.0) < 1e-5  # horizon = edge of a 180 degree lens
    assert f([0, 0, 1])[1] < 0                           # forward = top of image
    assert f([1, 0, 0])[0] > 0                           # right = right of image
