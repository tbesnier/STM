#!/usr/bin/env python3
"""
Batch rigid alignment of point-cloud sequences stored as .npy arrays of shape (T, N, 3).

For each file:
  - loads seq: (T,N,3)
  - uses frame 0 as reference A
  - for each frame t, computes rigid transform aligning frame t -> frame 0 (Kabsch/SVD)
  - applies it to all points in that frame
  - saves aligned sequence as .npy (same shape) in output directory

Assumes pointwise correspondences by index across frames (same N and consistent ordering).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def rigid_align(A: np.ndarray, B: np.ndarray):
    """
    Best-fit rigid transform aligning B to A with pointwise correspondences.
    Solves: A ≈ R @ B + t
    A, B: (N,3)
    Returns: R (3,3), t (3,)
    """
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("A and B must be the same shape (N,3).")

    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    AA = A - cA
    BB = B - cB

    H = BB.T @ AA
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = cA - R @ cB
    return R, t


def apply_transform(P: np.ndarray, R: np.ndarray, t: np.ndarray):
    """Apply rigid transform to Nx3 points."""
    return (R @ P.T).T + t


def align_sequence_to_first(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Align every frame of seq (T,N,3) to frame 0.
    Returns:
      aligned_seq: (T,N,3)
      rms_per_frame: (T,) RMS error vs frame 0 after alignment (frame 0 will be 0)
    """
    if seq.ndim != 3 or seq.shape[2] != 3:
        raise ValueError(f"Expected (T,N,3), got {seq.shape}")

    T, N, _ = seq.shape
    ref = np.asarray(seq[0], dtype=np.float64)
    aligned = np.empty_like(seq, dtype=np.float64)
    rms = np.empty((T,), dtype=np.float64)

    # Reference frame unchanged
    aligned[0] = ref
    rms[0] = 0.0

    for t in range(1, T):
        frame = np.asarray(seq[t], dtype=np.float64)
        if frame.shape != ref.shape:
            raise ValueError("Frame shape mismatch inside sequence.")
        R, tt = rigid_align(ref, frame)
        aligned_frame = apply_transform(frame, R, tt)
        aligned[t] = aligned_frame
        rms[t] = np.sqrt(np.mean(np.sum((ref - aligned_frame) ** 2, axis=1)))

    return aligned, rms


def main():
    parser = argparse.ArgumentParser(
        description="Rigidly align all frames of each (T,N,3) .npy sequence to its first frame."
    )
    parser.add_argument("--in_dir", default="../datasets/ravdess/tracking_npy/Actor_01", type=str, help="Directory containing .npy sequences")
    parser.add_argument("--out_dir", default="../datasets/ravdess/tracking_npy_aligned/Actor_01", type=str, help="Directory to write aligned .npy sequences")
    parser.add_argument("--pattern", default="*.npy", help="Glob pattern to match input files (default: *.npy)")
    parser.add_argument("--suffix", default="_aligned", help="Suffix appended before .npy (default: _aligned)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--report", action="store_true", help="Print per-file RMS stats")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.pattern} in {in_dir}")

    for fp in files:
        try:
            seq = np.load(fp)
            aligned, rms = align_sequence_to_first(seq)

            out_name = fp.stem + args.suffix + ".npy"
            out_path = out_dir / out_name

            if out_path.exists() and not args.overwrite:
                print(f"Skip (exists): {out_path}")
                continue

            np.save(out_path, aligned)

            if args.report:
                # Report summary RMS excluding frame 0
                if aligned.shape[0] > 1:
                    rms_non0 = rms[1:]
                    msg = (f"{fp.name}: saved -> {out_name} | "
                           f"T={aligned.shape[0]} N={aligned.shape[1]} | "
                           f"RMS min/mean/max = "
                           f"{rms_non0.min():.6f}/{rms_non0.mean():.6f}/{rms_non0.max():.6f}")
                else:
                    msg = f"{fp.name}: saved -> {out_name} | T=1 N={aligned.shape[1]} | RMS=0"
                print(msg)

        except Exception as e:
            print(f"ERROR on {fp}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
