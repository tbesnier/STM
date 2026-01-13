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


def transform_landmarks_3d(landmarks: np.ndarray) -> np.ndarray:
    """
    landmarks: (T, 68, 3)
    Apply per-point transform to all frames:
      rotate X -180°, then Y -180° (extrinsic/global axes),
      scale 0.001,
      translate Z + 1.43
    """
    if landmarks.ndim != 3 or landmarks.shape[-1] != 3:
        raise ValueError(f"Expected (T,68,3), got {landmarks.shape}")

    pts = landmarks.reshape(-1, 3).astype(np.float32, copy=False)

    # Lowercase axes => extrinsic rotations in SciPy. :contentReference[oaicite:3]{index=3}
    rot = R.from_euler("x", 180.0, degrees=True)
    pts = rot.apply(pts)  # apply rotation to all vectors :contentReference[oaicite:4]{index=4}

    pts *= 0.001
    pts[:, 2] += 1.43

    return pts.reshape(landmarks.shape).astype(np.float32, copy=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export RAVDESS landmark CSVs to .npy sequences (with optional transform).")
    ap.add_argument("--input_dir", type=Path, default="../datasets/ravdess/tracking/Actor_01",
                    help="Directory containing RAVDESS tracking CSV files.")
    ap.add_argument("--output_dir", type=Path, default="../datasets/ravdess/tracking_npy/Actor_01",
                    help="Directory to write .npy files into.")
    ap.add_argument("--mode", choices=["2d", "3d"], default="3d", help="Export 2D or 3D landmarks.")
    ap.add_argument("--recursive", action="store_true", help="Search input_dir recursively.")
    ap.add_argument("--no_transform", action="store_true",
                    help="Disable the 3D transform (rotation/scale/translation).")
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.csv" if args.recursive else "*.csv"
    csv_files = sorted(input_dir.glob(pattern))

    if not csv_files:
        print(f"No CSV files found in {input_dir} (recursive={args.recursive}).")
        return 0

    ok, failed = 0, 0
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            landmarks = extract_landmarks(df, mode=args.mode)

            # Apply your requested transform to 3D landmarks
            if args.mode == "3d" and not args.no_transform:
                landmarks = transform_landmarks_3d(landmarks)

            rel = csv_path.relative_to(input_dir)
            out_path = (output_dir / rel).with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(out_path, landmarks)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {csv_path}: {e}")

    print(f"Done. Saved {ok} file(s). Failed: {failed}.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
