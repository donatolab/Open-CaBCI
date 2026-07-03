import argparse
import gc
import gzip
import os
import shutil
import sys
import time
from multiprocessing import Process, shared_memory
from pathlib import Path

import numpy as np


if "--gui" in sys.argv:
    import matplotlib

    matplotlib.use("TkAgg", force=True)


REPOSITORY_ROOT = Path(__file__).resolve().parent
OPENBMI_ROOT = REPOSITORY_ROOT / "openbmi"
DEFAULT_DATASET = OPENBMI_ROOT / "data_samples" / "demo_5min"
sys.path.insert(0, str(OPENBMI_ROOT))

from bmi.bmi import BMI


REQUIRED_ROI_KEYS = {
    "calibration_template",
    "contours_all_cells",
    "ensemble1_contours",
    "ensemble1_f0s",
    "ensemble1_footprints",
    "ensemble2_contours",
    "ensemble2_f0s",
    "ensemble2_footprints",
    "high_freq",
    "high_threshold",
    "low_freq",
    "low_threshold",
    "post_reward_lockout",
    "rois_smooth_window",
    "smooth_diff_function_flag",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the bundled Open-CaBCI dataset without acquisition hardware."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--frames", type=int, default=9000)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--gui", action="store_true", help="show the live BMI GUI")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="pace imaging frames at the recorded 30 Hz rate",
    )
    return parser.parse_args()


def prepare_movie(dataset):
    movie_path = dataset / "data" / "Image_001_001.raw"
    if movie_path.is_file():
        return movie_path

    movie_parts = sorted((dataset / "data" / "movie_parts").rglob("*.raw.gz"))
    if not movie_parts:
        raise FileNotFoundError(
            f"Missing {movie_path} and compressed movie parts"
        )

    temporary_path = movie_path.with_suffix(".raw.tmp")
    movie_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_path.open("wb") as destination:
            for part in movie_parts:
                print(f"Extracting {part.name}")
                with gzip.open(part, "rb") as source:
                    shutil.copyfileobj(source, destination)
        os.replace(temporary_path, movie_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return movie_path


def validate_dataset(dataset, frame_count):
    movie_path = prepare_movie(dataset)
    roi_path = dataset / "rois_pixels_and_thresholds.npz"
    ttl_path = dataset / "ttl_pulses.npy"

    missing_files = [
        path for path in (movie_path, roi_path, ttl_path) if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing demo files: " + ", ".join(str(path) for path in missing_files)
        )

    bytes_per_frame = 512 * 512 * np.dtype("uint16").itemsize
    movie_frame_count, remainder = divmod(movie_path.stat().st_size, bytes_per_frame)
    if remainder or movie_frame_count < 1:
        raise ValueError(
            f"Movie size {movie_path.stat().st_size} is not a positive number "
            "of 512x512 uint16 frames"
        )

    with np.load(roi_path, allow_pickle=True) as roi_data:
        missing_keys = REQUIRED_ROI_KEYS.difference(roi_data.files)
        if missing_keys:
            raise ValueError(
                "ROI archive is missing keys: " + ", ".join(sorted(missing_keys))
            )

    ttl = np.load(ttl_path).reshape(-1)
    falling_edges = np.count_nonzero((ttl[:-1] >= 1) & (ttl[1:] < 1))
    if falling_edges < 1:
        raise ValueError(
            "TTL file must contain at least one falling edge"
        )

    if frame_count > movie_frame_count:
        raise ValueError(
            f"Movie contains {movie_frame_count} frames; {frame_count} were requested"
        )

    return movie_path, roi_path, ttl_path


def unlink_shared_memory(bmi):
    shared_blocks = [
        value
        for value in vars(bmi).values()
        if isinstance(value, shared_memory.SharedMemory)
    ]
    for block in shared_blocks:
        try:
            block.unlink()
        except FileNotFoundError:
            pass


def start_plotter(bmi, roi_path):
    from plotter.plotter import PlotROIs

    process = Process(
        target=PlotROIs,
        args=(
            False,
            str(roi_path),
            bmi.shmem_rois_traces_ensemble1.name,
            bmi.shmem_rois_traces_ensemble2.name,
            bmi.shmem_n_ttl.name,
            bmi.rois_traces_raw_ensemble1.shape,
            bmi.rois_traces_raw_ensemble2.shape,
            bmi.shmem_reward_times.name,
            bmi.shmem_tone_state.name,
            bmi.shmem_live_frame_plotter.name,
            bmi.shmem_ensemble_state.name,
            bmi.high_threshold,
            bmi.shmem_termination_flag.name,
            bmi.shmem_live_video_frame.name,
            bmi.shmem_high_threshold_state.name,
            bmi.video_width,
            bmi.video_length,
            bmi.shmem_motion_correction_flag.name,
            False,
            bmi.shmem_dynamic_f0_flag.name,
            bmi.shmem_manual_motion_correction_array.name,
            bmi.shmem_contingency_degradation.name,
        ),
    )
    process.start()
    time.sleep(2)
    if not process.is_alive():
        raise RuntimeError(f"GUI process exited during startup ({process.exitcode})")
    return process


def validate_results(result_path, frame_count):
    if not result_path.is_file():
        raise RuntimeError(f"Simulation did not create {result_path}")

    with np.load(result_path, allow_pickle=True) as results:
        saved_frame_count = int(results["n_frames"])
        detected_frames = len(results["ttl_n_detected"])
        raw_ensemble1 = results["rois_traces_raw_ensemble1"]
        raw_ensemble2 = results["rois_traces_raw_ensemble2"]

        if saved_frame_count != frame_count or detected_frames != frame_count:
            raise RuntimeError(
                f"Expected {frame_count} frames, saved {saved_frame_count}, "
                f"detected {detected_frames}"
            )
        if not np.isfinite(raw_ensemble1.astype(float)).all():
            raise RuntimeError("Ensemble 1 contains non-finite ROI values")
        if not np.isfinite(raw_ensemble2.astype(float)).all():
            raise RuntimeError("Ensemble 2 contains non-finite ROI values")

        print(f"Validated {detected_frames} processed frames")
        print(
            "ROI ranges: "
            f"E1={float(raw_ensemble1.min()):.3f}..{float(raw_ensemble1.max()):.3f}, "
            f"E2={float(raw_ensemble2.min()):.3f}..{float(raw_ensemble2.max()):.3f}"
        )


def main():
    args = parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be positive")
    if args.sleep < 0:
        raise ValueError("--sleep cannot be negative")
    if args.gui and args.frames < 900:
        raise ValueError("--gui requires at least 900 frames for its 30-second plot window")

    dataset = args.dataset.resolve()
    movie_path, roi_path, ttl_path = validate_dataset(dataset, args.frames)
    result_path = movie_path.parent / "results.npz"

    print(f"Dataset: {dataset}")
    print(f"Running {args.frames} frames in hardware-free simulation mode")

    bmi = None
    plotter = None
    try:
        bmi = BMI(
            True,
            True,
            str(dataset),
            str(movie_path),
            str(ttl_path),
            30,
            str(roi_path),
            max(60, args.frames // 30 + 30),
            args.frames,
            100,
            100,
            False,
            False,
        )
        bmi.sleep_time_sec = args.sleep
        bmi.simulation_realtime = args.realtime
        bmi.verbose = False
        bmi.verbose2 = False
        if args.gui:
            plotter = start_plotter(bmi, roi_path)
        bmi.run_BMI()
        validate_results(result_path, args.frames)
    finally:
        if bmi is not None:
            bmi.termination_flag[0] = 1
            if plotter is not None:
                plotter.join(timeout=5)
                if plotter.is_alive():
                    plotter.terminate()
                    plotter.join(timeout=5)
            unlink_shared_memory(bmi)
            del bmi
            gc.collect()

    print(f"Simulation output: {result_path}")


if __name__ == "__main__":
    main()
