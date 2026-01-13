import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def extract_landmarks(df: pd.DataFrame, mode: str = "3d") -> np.ndarray:
    """Return landmarks as (T, 68, C) where C=3 (3d) or C=2 (2d)."""
    if mode == "2d":
        xs = df[[f"x_{i}" for i in range(68)]].to_numpy()
        ys = df[[f"y_{i}" for i in range(68)]].to_numpy()
        arr = np.stack([xs, ys], axis=-1)  # (T,68,2)
    elif mode == "3d":
        X = df[[f"X_{i}" for i in range(68)]].to_numpy()
        Y = df[[f"Y_{i}" for i in range(68)]].to_numpy()
        Z = df[[f"Z_{i}" for i in range(68)]].to_numpy()
        arr = np.stack([X, Y, Z], axis=-1)  # (T,68,3)
    else:
        raise ValueError("mode must be '2d' or '3d'")
    return arr.astype(np.float32)


def apply_global_transform_xyz(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T,68,3)
    Rotate X -180°, then Y -180° (extrinsic axes via 'xy'), scale 0.001, translate Z +1.43.
    """
    if seq.ndim != 3 or seq.shape[-1] != 3:
        raise ValueError(f"Expected (T,68,3), got {seq.shape}")

    pts = seq.reshape(-1, 3).astype(np.float32, copy=False)

    # SciPy: lowercase axis letters => extrinsic rotations. :contentReference[oaicite:2]{index=2}
    rot = R.from_euler("x", 180.0, degrees=True)
    pts = rot.apply(pts)  # equivalent to self.as_matrix() @ vectors :contentReference[oaicite:3]{index=3}

    pts *= 0.001
    pts[:, 2] += 1.43

    return pts.reshape(seq.shape).astype(np.float32, copy=False)


def kabsch_align_to_ref(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Align P to Q with a rigid transform (rotation+translation), minimizing RMSD (Kabsch). :contentReference[oaicite:4]{index=4}
    P, Q: (N,3)
    Returns aligned P': (N,3)
    """
    P = np.asarray(P, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Expected P,Q as (N,3) same shape, got {P.shape} vs {Q.shape}")

    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    P0 = P - cP
    Q0 = Q - cQ

    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T

    # Reflection fix: enforce det(R)=+1
    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1.0
        Rm = Vt.T @ U.T

    return (P0 @ Rm) + cQ


def rigid_align_sequence_to_reference(seq: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
    """
    seq: (T,68,3), ref_frame: (68,3)
    Returns aligned seq where each frame is rigidly aligned to ref_frame.
    """
    if seq.ndim != 3 or seq.shape[-1] != 3:
        raise ValueError(f"Expected (T,68,3), got {seq.shape}")
    if ref_frame.shape != (seq.shape[1], 3):
        raise ValueError(f"Expected ref_frame shape ({seq.shape[1]},3), got {ref_frame.shape}")

    out = np.empty_like(seq, dtype=np.float32)
    for t in range(seq.shape[0]):
        out[t] = kabsch_align_to_ref(seq[t], ref_frame)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export RAVDESS landmark CSVs to .npy sequences with global transform + rigid alignment."
    )
    ap.add_argument("--input_dir", type=Path, default="../datasets/ravdess/tracking/Actor_01",
                    help="Directory containing RAVDESS tracking CSV files.")
    ap.add_argument("--output_dir", type=Path, default="../datasets/ravdess/tracking_npy/Actor_01",
                    help="Directory to write .npy files into.")
    ap.add_argument("--mode", choices=["2d", "3d"], default="3d",
                    help="Export 2D or 3D landmarks.")
    ap.add_argument("--recursive", action="store_true", help="Search input_dir recursively.")
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.csv" if args.recursive else "*.csv"
    csv_files = sorted(input_dir.glob(pattern))

    if not csv_files:
        print(f"No CSV files found in {input_dir} (recursive={args.recursive}).")
        return 0

    # --- Build the global reference: first frame of the first sequence (after global transform)
    ref_frame = None
    if args.mode == "3d":
        df0 = pd.read_csv(csv_files[0])
        seq0 = extract_landmarks(df0, mode="3d")
        seq0 = apply_global_transform_xyz(seq0)
        ref_frame = seq0[0].copy()  # (68,3)

    ok, failed = 0, 0
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            landmarks = extract_landmarks(df, mode=args.mode)

            if args.mode == "3d":
                landmarks = apply_global_transform_xyz(landmarks)
                landmarks = rigid_align_sequence_to_reference(landmarks, ref_frame)

            rel = csv_path.relative_to(input_dir)
            out_path = (output_dir / rel).with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(out_path, landmarks.astype(np.float32, copy=False))
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {csv_path}: {e}")

    print(f"Done. Saved {ok} file(s). Failed: {failed}.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
