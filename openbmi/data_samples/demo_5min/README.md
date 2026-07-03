# Open-CaBCI Five-Minute Demonstration Dataset

This dataset contains the first 9,000 consecutive frames (five minutes at 30 Hz) from source session `DON-019683/20240120`. It validates the Open-CaBCI online pipeline without microscope, NI-DAQ, camera, tone, water, or animal hardware.

## Contents

- `data/movie_parts/batch_01/*.raw.gz`: frames 0–4,499 in 45 sequential chunks.
- `data/movie_parts/batch_02/*.raw.gz`: frames 4,500–8,999 in 45 sequential chunks.
- `rois_pixels_and_thresholds.npz`: matching ROI footprints, calibration baselines, contours, thresholds, frequencies, and template.
- `ttl_pulses.npy`: a recorded 1,000-edge TTL waveform replayed cyclically by simulation for the 9,000-frame run.
- `day0/day0_ca_mask.npz`: matching day-0 calcium mask from source session `DON-019683/20240119`.

Every compressed movie chunk is below 36 MB, safely below [GitHub's enforced 100 MB per-file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Each batch is below 1.6 GB, so maintainers can commit and push `batch_01` first and `batch_02` second without exceeding GitHub's [2 GB push limit](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits). The complete dataset is approximately 3.1 GB.

## Run

From the repository root:

```bash
python run_demo.py
```

For a five-minute realtime replay with the live GUI:

```bash
python run_demo.py --gui --realtime
```

The first run reconstructs `data/Image_001_001.raw` and requires approximately 4.7 GB of additional free space. It writes `data/results.npz` and `data/results.xlsx`. These generated files are ignored by Git.

## Integrity

- `Image_001_001.raw`: `b829f0e795d516efda790554c0a997d8f99e12eed0f6c5b099b5131e143af36c`
- `rois_pixels_and_thresholds.npz`: `c9c356aa831054ecee99128bf33330887555d5d0e9d250f05e7f4881355ef227`
- `ttl_pulses.npy`: `1dd80e904eea8b5637a7bbbfc4458320a04168550cb1554877e93301b4ad2641`
- `day0/day0_ca_mask.npz`: `e57795a88e58fc9e6176a8c045027ba789d4261e7368ef9c79533cf86ae1f76b`
