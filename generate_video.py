import os
import glob
import subprocess
from pathlib import Path

import numpy as np
import pyvista as pv
import imageio.v2 as imageio

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
    'cornsilk', 'crimson', 'cyan', 'darkblue',
    'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
    'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
    'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
    'darkslateblue', 'darkslategray', 'darkslategrey',
    'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
    'dimgray', 'dimgrey', 'dodgerblue', '#000000'
]
i=0
color = tuple(int(colors[-1][i:i + 2], 16) / 255.0 for i in (1, 3, 5))


def _clear_scene_keep_lights(plotter: pv.Plotter) -> None:
    """Clear actors/props while keeping lights intact (when supported)."""
    if hasattr(plotter, "clear_actors"):
        plotter.clear_actors()
    else:
        # Fallback: clear everything (may also clear lights on some versions)
        plotter.clear()


def _is_point_cloud(mesh: pv.DataSet) -> bool:
    """Heuristic: treat as point cloud if there are no polygon faces."""
    faces = getattr(mesh, "faces", None)
    return faces is None or len(faces) == 0


def _update_three_point_lights(key: pv.Light, fill: pv.Light, rim: pv.Light, center, scale: float) -> None:
    """Reposition a simple 3-point lighting rig around the mesh center."""
    cx, cy, cz = center
    s = max(float(scale), 1e-6)

    key.position = (cx + 1.2 * s, cy + 1.0 * s, cz + 1.2 * s)
    fill.position = (cx - 1.2 * s, cy + 0.6 * s, cz + 0.8 * s)
    rim.position = (cx + 0.0 * s, cy - 1.4 * s, cz + 1.2 * s)

    key.focal_point = (cx, cy, cz)
    fill.focal_point = (cx, cy, cz)
    rim.focal_point = (cx, cy, cz)


def _add_mesh_with_visible_geometry(plotter: pv.Plotter, mesh: pv.DataSet) -> None:
    """Render mesh with depth cues (edges/material) and handle point clouds."""
    # Force perspective projection (not orthographic)
    plotter.camera.parallel_projection = False

    if _is_point_cloud(mesh):
        # Point clouds: spheres + eye-dome lighting gives strong depth cues
        if hasattr(plotter, "enable_eye_dome_lighting"):
            plotter.enable_eye_dome_lighting()
        plotter.add_mesh(
            mesh,
            render_points_as_spheres=True,
            point_size=6,
            ambient=0.05,
            diffuse=0.95,
            specular=0.25,
            specular_power=25,
        )
    else:
        # Surface meshes: edges + lower ambient makes shape readable
        if hasattr(plotter, "disable_eye_dome_lighting"):
            plotter.disable_eye_dome_lighting()
        try:
            mesh = mesh.triangulate()
        except Exception:
            pass
        plotter.add_mesh(
            mesh,
            color="17becf",
            smooth_shading=False,  # facet cues help a lot
            show_edges=False,
            line_width=0.1,
            ambient=0.08,
            diffuse=0.9,
            specular=0.35,
            specular_power=40,
        )


def make_video_with_audio(ply_dir, wav_path, out_path="out.mp4", fps=30, width=1280, height=720):
    ply_files = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))
    if not ply_files:
        raise FileNotFoundError(f"No .ply files in {ply_dir}")

    tmp_video = str(Path(out_path).with_suffix("").as_posix() + "_noaudio.mp4")

    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background("white")

    # Ensure we're using perspective
    plotter.camera.parallel_projection = False

    # ---------- CAMERA (paste from the interactive step) ----------
    # Example placeholder; REPLACE with what you printed:
    # CAM = plotter.camera_position  # fallback if you forget to paste
    CAM = [
        (0.01969904175276854, -0.011327166386706219, 0.8570828831406961),
        (0.0017170123755931854, -0.026146210730075836, -0.039116524159908295),
        (0.013193407296681715, 0.9997718817495934, -0.016796382550436362),
    ]
    # -------------------------------------------------------------

    # ---------- LIGHTING ----------
    # Remove default lights and add a simple 3-point setup
    plotter.remove_all_lights()

    key = pv.Light(light_type="scene light")
    key.position = (2, 2, 2)
    key.focal_point = (0, 0, 0)
    key.intensity = 1.0
    plotter.add_light(key)

    fill = pv.Light(light_type="scene light")
    fill.position = (-2, 1, 1.5)
    fill.focal_point = (0, 0, 0)
    fill.intensity = 0.5
    plotter.add_light(fill)

    rim = pv.Light(light_type="scene light")
    rim.position = (0, -2, 2)
    rim.focal_point = (0, 0, 0)
    rim.intensity = 0.0
    plotter.add_light(rim)
    # -----------------------------

    writer = imageio.get_writer(tmp_video, fps=fps, format="FFMPEG", codec="libx264", quality=8)

    try:
        for i, f in enumerate(ply_files):
            mesh = pv.read(f)

            # Keep lights; only clear actors
            _clear_scene_keep_lights(plotter)
            plotter.camera_position = CAM  # lock view every frame

            # Update light positions to follow the mesh (helps if meshes aren't centered)
            try:
                center = mesh.center
                scale = getattr(mesh, "length", None)
                if scale is None:
                    # Fallback: use bounding box diagonal-ish
                    bounds = mesh.bounds
                    scale = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
                _update_three_point_lights(key, fill, rim, center, scale)
            except Exception:
                pass

            # Render with strong geometry/perspective cues
            _add_mesh_with_visible_geometry(plotter, mesh)

            img = plotter.screenshot(return_img=True)
            writer.append_data(img)

            if (i + 1) % 50 == 0:
                print(f"Rendered {i+1}/{len(ply_files)} frames...")
    finally:
        writer.close()
        plotter.close()

    subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-i", wav_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_path
    ], check=True)

    try:
        os.remove(tmp_video)
    except OSError:
        pass

    print("Wrote:", out_path)


if __name__ == "__main__":
    # EDIT THESE:
    PLY_DIR = "../Data/STM/test_dn_mlp_seq/infer"
    WAV_PATH = "./data/ravdess/ex_angry.wav"#"../datasets/VOCA_training/wav/FaceTalk_170725_00137_TA_sentence01.wav"#"D:/phd_data/ravdess/wav/01-02-05-02-02-02-01.wav"#01-02-03-02-02-01-01.wav"#01-02-05-02-02-02-01.wav"#"../datasets/MEAD/audio/W009_fear_3_028.wav" #"../datasets/VOCA_training/wav/FaceTalk_170725_00137_TA_sentence01.wav"
    OUT_PATH = "test_pred_ravdess_angry.mp4"

    make_video_with_audio(PLY_DIR, WAV_PATH, OUT_PATH, fps=25)
