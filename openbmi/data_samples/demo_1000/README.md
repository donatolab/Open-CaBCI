# Open-CaBCI Lightweight Demonstration Dataset

This reviewer dataset contains the first 1,000 consecutive frames (33.3 seconds at 30 Hz) from source session `DON-019683/20240120`. It exercises the complete hardware-free online pipeline while downloading only the first ten movie chunks from the full five-minute demonstration.

## Contents

- `data/movie_parts/*.raw.gz`: frames 0–999 in ten sequential 100-frame chunks.
- `rois_pixels_and_thresholds.npz`: matching ROI footprints, calibration baselines, contours, thresholds, frequencies, and template.
- `ttl_pulses.npy`: a recorded 1,000-edge TTL waveform.
- `day0/day0_ca_mask.npz`: matching day-0 calcium mask from source session `DON-019683/20240119`.

The compressed movie is 338 MiB. Its files are identical Git blobs reused from the full dataset, so this option does not duplicate repository storage.

## Run

From the repository root:

```bash
python run_demo.py --dataset openbmi/data_samples/demo_1000 --frames 1000
```

The first run reconstructs `data/Image_001_001.raw`, writes `data/results.npz` and `data/results.xlsx`, and validates all 1,000 processed frames. These generated files are ignored by Git and can be deleted safely.

## Integrity

- `Image_001_001.raw`: `37a1a12ee28b9d20379497b2ee736cddd575217016cb026ac0ce10d645fb4877`
- `rois_pixels_and_thresholds.npz`: `c9c356aa831054ecee99128bf33330887555d5d0e9d250f05e7f4881355ef227`
- `ttl_pulses.npy`: `1dd80e904eea8b5637a7bbbfc4458320a04168550cb1554877e93301b4ad2641`
- `day0/day0_ca_mask.npz`: `e57795a88e58fc9e6176a8c045027ba789d4261e7368ef9c79533cf86ae1f76b`
