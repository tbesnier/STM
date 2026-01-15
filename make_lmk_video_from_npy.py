"""
Make an .mp4 video from a (T, N, 3) landmark sequence stored as .npy using PyVista.

Requirements:
  pip install pyvista imageio imageio-ffmpeg numpy

Usage:
  python make_landmark_video.py --inp path/to/seq.npy --out out.mp4
"""

import argparse
import numpy as np
import pyvista as pv
import imageio.v2 as imageio


def make_video(
    npy_path: str,
    out_mp4: str,
    fps: int = 30,
    point_size: float = 10.0,
    color: str = "tomato",
    bg: str = "white",
    resolution=(1280, 720),
):
    seq = np.load(npy_path)  # (T, N, 3)
    if seq.ndim != 3 or seq.shape[-1] != 3:
        raise ValueError(f"Expected shape (T, N, 3), got {seq.shape}")

    T, N, _ = seq.shape

    # Global bounds across time so the camera doesn't "jump"
    mins = seq.reshape(-1, 3).min(axis=0)
    maxs = seq.reshape(-1, 3).max(axis=0)
    center = (mins + maxs) / 2.0
    diag = np.linalg.norm(maxs - mins)
    if diag == 0:
        diag = 1.0

    # Off-screen render
    pv.global_theme.window_size = resolution
    plotter = pv.Plotter(off_screen=True, window_size=resolution)
    plotter.set_background(bg)

    # Add initial points
    points = seq[0]
    pdata = pv.PolyData(points)
    actor = plotter.add_points(
        pdata,
        color=color,
        point_size=point_size,
        render_points_as_spheres=True,
    )

    # Set a stable camera
    plotter.camera_position = "xy"   # quick default
    plotter.camera.focal_point = center.tolist()
    # Pull camera back so everything fits
    plotter.camera.position = (center + np.array([0, 0, 2.5 * diag])).tolist()
    plotter.camera.up = (0, 1, 0)

    # Optional: add axes
    plotter.add_axes()

    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8)

    try:
        for t in range(T):
            pdata.points = seq[t]  # update geometry in-place
            plotter.render()
            frame = plotter.screenshot(return_img=True)  # HxWx3 uint8
            writer.append_data(frame)
    finally:
        writer.close()
        plotter.close()

    print(f"Saved: {out_mp4}  (T={T}, N={N}, fps={fps})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="../datasets/ravdess/tracking_npy_aligned/Actor_01/01-01-01-01-01-01-01_aligned.npy", help="Input .npy file with shape (T, N, 3)")
    ap.add_argument("--out", default="./test_lmk_vid_bis.mp4", help="Output .mp4 path")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--point_size", type=float, default=15.0)
    ap.add_argument("--color", default="tomato")
    ap.add_argument("--bg", default="white")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    args = ap.parse_args()

    make_video(
        npy_path=args.inp,
        out_mp4=args.out,
        fps=args.fps,
        point_size=args.point_size,
        color=args.color,
        bg=args.bg,
        resolution=(args.w, args.h),
    )


if __name__ == "__main__":
    main()
