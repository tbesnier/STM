#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def write_ply_xyz_ascii(path: Path, points_xyz: np.ndarray) -> None:
    """
    Write Nx3 float point cloud to an ASCII PLY file (vertices only, no faces).
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected (N,3) points, got {points_xyz.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    n = points_xyz.shape[0]

    # PLY ASCII header: vertices with x,y,z float properties. :contentReference[oaicite:1]{index=1}
    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "end_header"
    ]) + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for x, y, z in points_xyz:
            f.write(f"{x} {y} {z}\n")


def first_frame_points(arr: np.ndarray) -> np.ndarray:
    """
    Accepts:
      - (T, N, 3) -> returns (N, 3) from frame 0
      - (N, 3)    -> returns (N, 3)
      - (T, N, 2) or (N, 2) -> returns (N, 3) with z=0
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        pts = arr[0]  # first frame
    elif arr.ndim == 2:
        pts = arr
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"Expected (N,2) or (N,3), got {pts.shape}")

    if pts.shape[1] == 2:
        z = np.zeros((pts.shape[0], 1), dtype=pts.dtype)
        pts = np.concatenate([pts, z], axis=1)

    return pts


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert first frame of landmark .npy sequences into .ply point clouds.")
    ap.add_argument("--input_dir", type=Path, default="../datasets/ravdess/tracking_npy/Actor_01", help="Directory containing .npy files.")
    ap.add_argument("--output_dir", type=Path, default="../datasets/ravdess/lmk_ply/Actor_01", help="Directory to write .ply files into.")
    ap.add_argument("--recursive", action="store_true", help="Search input_dir recursively.")
    args = ap.parse_args()

    in_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.npy" if args.recursive else "*.npy"
    npy_files = sorted(in_dir.glob(pattern))
    if not npy_files:
        print(f"No .npy files found in {in_dir} (recursive={args.recursive}).")
        return 0

    ok, failed = 0, 0
    for npy_path in npy_files:
        try:
            arr = np.load(npy_path)
            pts = first_frame_points(arr)

            rel = npy_path.relative_to(in_dir)
            ply_path = (out_dir / rel).with_suffix(".ply")
            write_ply_xyz_ascii(ply_path, pts)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {npy_path}: {e}")

    print(f"Done. Wrote {ok} PLY file(s). Failed: {failed}.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
