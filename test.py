#!/usr/bin/env python3
"""
Given:
  1) a directory of *template meshes* (e.g., 12 named meshes)
  2) a directory of *deformation sequences*, where each sequence is a directory
     whose name contains the corresponding template name

This script:
  - loads each template mesh
  - finds all deformation-sequence directories that match that template
  - prints all mesh files (frames) inside each matching sequence directory

Usage:
  python list_sequences.py /path/to/templates /path/to/deformations

Notes:
  - Matching is done by substring: template_stem in sequence_dir_name (case-insensitive).
  - Mesh loading uses trimesh if available (recommended). Install with:
      pip install trimesh
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import trimesh

DEFAULT_MESH_EXTS = (".ply", ".obj", ".off", ".stl", ".glb", ".gltf")


def iter_mesh_files(folder: Path, exts: Tuple[str, ...]) -> List[Path]:
    exts_lc = tuple(e.lower() for e in exts)
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts_lc])

def iter_template_meshes(templates_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    files = []
    for ext in exts:
        files.extend(templates_dir.glob(f"*{ext}"))
        files.extend(templates_dir.glob(f"*{ext.upper()}"))
    # Deduplicate & sort
    return sorted(set(files), key=lambda p: p.name.lower())

def find_matching_sequence_dirs(seqs_root: Path, template_key: str) -> List[Path]:
    key = template_key.lower()
    matches = [d for d in seqs_root.iterdir() if d.is_dir() and key in d.name.lower()]
    return sorted(matches, key=lambda p: p.name.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates_dir", default="D:/phd_data/ICT_templates", type=Path, help="Directory containing template mesh files")
    ap.add_argument("--deformations_dir", default="D:/phd_data/ICT_interp", type=Path, help="Directory containing deformation sequence directories")
    ap.add_argument(
        "--exts",
        nargs="*",
        default=list(DEFAULT_MESH_EXTS),
        help=f"Mesh file extensions to consider (default: {', '.join(DEFAULT_MESH_EXTS)})",
    )
    ap.add_argument(
        "--print-full-paths",
        action="store_true",
        help="Print full paths instead of just file names for frames",
    )
    args = ap.parse_args()

    templates_dir: Path = args.templates_dir
    deformations_dir: Path = args.deformations_dir
    exts: Tuple[str, ...] = tuple(args.exts)
    template_files = iter_template_meshes(templates_dir, exts)

    for tmpl_path in template_files:
        template_name = tmpl_path.stem  # key used for matching sequence dir names

        # Load template mesh (as requested)
        mesh = trimesh.load(tmpl_path, process=False)

        # Print template info
        verts = getattr(mesh, "vertices", None)
        faces = getattr(mesh, "faces", None)
        v_count = len(verts) if verts is not None else "?"
        f_count = len(faces) if faces is not None else "?"
        print("\n" + "=" * 80)
        print(f"Template: {tmpl_path.name}")
        print(f"  key (stem): {template_name}")
        print(f"  loaded: vertices={v_count}, faces={f_count}")

        # Find deformation sequences for this template
        seq_dirs = find_matching_sequence_dirs(deformations_dir, template_name)
        if not seq_dirs:
            print("  No matching deformation sequences found.")
            continue

        print(f"  Matching deformation sequences: {len(seq_dirs)}")
        for seq_dir in seq_dirs:
            frame_files = iter_mesh_files(seq_dir, exts)
            print("-" * 80)
            print(f"  Sequence dir: {seq_dir.name}  ({len(frame_files)} frames)")

            if not frame_files:
                print("    [No mesh frames found in this sequence directory]")
                continue

            for f in frame_files:
                print(f"    {str(f) if args.print_full_paths else f.name}")


if __name__ == "__main__":
    main()
