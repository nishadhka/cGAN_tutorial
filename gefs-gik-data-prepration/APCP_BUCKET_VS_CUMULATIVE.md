# APCP Bucket-Incremental vs Cumulative: Impact on cGAN Inference

## Summary

The cGAN model was **trained on per-bucket APCP** (6-hour accumulation windows,
~0–30 mm range). Feeding it **total-cumulative APCP** (accumulated from hour 0,
~0–150 mm) inflates the APCP input by ~5–9x in raw values and ~1.5–2.1x after
log normalization, pushing the model outside its training distribution and producing
incorrect output intensities.

## How GEFS APCP Works

GEFS precipitation (APCP) uses **6-hour accumulation buckets** that reset at
hours 0, 6, 12, 18, 24, 30, 36, ...

At each forecast step, the APCP value represents precipitation accumulated since
the **last bucket boundary**:

```
Bucket boundaries:  0    6    12   18   24   30   36   42   48   54   60
                    |----|----|----|----|----|----|----|----|----|----|
Step T+30h: accum from 24h to 30h  (one 6h window)
Step T+33h: accum from 30h to 33h  (partial, 3h into new bucket)
Step T+36h: accum from 30h to 36h  (full 6h bucket)
```

**Per-bucket values** at boundary steps (T+30h, T+36h, etc.) represent exactly
one 6-hour window of rainfall. Typical maximum: ~20–30 mm for East Africa.

**Cumulative values** (total from hour 0) sum all preceding buckets:
- T+30h cumulative = sum of 5 buckets (hours 0→30) → ~5x larger
- T+48h cumulative = sum of 8 buckets (hours 0→48) → ~8x larger

## How APCP Flows Through cGAN Inference

### Input normalization (in `run_gefs_inference_raw.py`)

APCP has a **unique normalization** compared to other fields:

```python
# APCP — log transform only, NO linear normalization
if field == "apcp":
    data = np.log10(1 + data)          # log compress
    data_mean = np.nanmean(data, axis=-2)  # ensemble mean
    data_std  = np.nanstd(data, axis=-2)   # ensemble spread
    # → 4 channels: mean_step1, std_step1, mean_step2, std_step2
```

Other fields use linear normalization with statistics from `FCSTNorm_GEFS_2018.pkl`:
- Bounded fields (pres, tmp, msl): `(data - mean) / std`
- Non-negative fields (cape, pwat): `data / max`
- Wind fields (ugrd, vgrd): `data / max(|min|, |max|)`

**APCP is not in the normalization pickle at all** — it relies solely on `log10(1+x)`.

### Output denormalization

```python
def denormalise(data):
    return np.minimum(np.power(10.0, data) - 1.0, 100.0)  # mm/h, capped at 100
```

### Lookup table: input → normalized → output

```
APCP input (mm)  →  log10(1+x)  →  network sees
       0.0       →    0.0000
       0.1       →    0.0414
       1.0       →    0.3010
       5.0       →    0.7782
      10.0       →    1.0414
      20.0       →    1.3222      ← typical bucket maximum
      30.0       →    1.4914      ← model training range ceiling
      50.0       →    1.7076      ← cumulative starts here
     100.0       →    2.0043
     150.0       →    2.1790      ← cumulative for 30mm × 5 buckets

network output   →  10^x - 1     →  precipitation (mm/h)
      0.0        →    0.00 mm/h
      0.1        →    0.26 mm/h
      0.5        →    2.16 mm/h
      1.0        →    9.00 mm/h
      1.5        →   30.62 mm/h
      2.0        →   99.00 mm/h   ← cap at 100 mm/h
```

## Impact of Cumulative vs Bucket

The model was trained to see `log10(1+bucket_mm)` values in the range **0 to ~1.5**.

| Pixel rain | Bucket (mm) | log10(1+bucket) | Cumul@30h (5x) | log10(1+cumul) | Inflation |
|------------|-------------|-----------------|----------------|----------------|-----------|
| Light      | 5           | 0.778           | 25             | 1.415          | 1.8x      |
| Moderate   | 10          | 1.041           | 50             | 1.708          | 1.6x      |
| Heavy      | 20          | 1.322           | 100            | 2.004          | 1.5x      |
| Extreme    | 30          | 1.491           | 150            | 2.179          | 1.5x      |

At T+48h (8 buckets cumulated), the inflation is even larger:

| Pixel rain | Bucket (mm) | log10(1+bucket) | Cumul@48h (8x) | log10(1+cumul) | Inflation |
|------------|-------------|-----------------|----------------|----------------|-----------|
| Light      | 5           | 0.778           | 40             | 1.613          | 2.1x      |
| Heavy      | 20          | 1.322           | 160            | 2.207          | 1.7x      |

Even though `log10` compresses the ratio, cumulative APCP pushes the network
input 1.5–2.1x beyond what the model saw during training.

## Observed Results

### With cumulative APCP (wrong)
```
APCP input correlation: r=0.843 (Herbie vs GIK differ significantly)
  Herbie APCP max: 30.66 mm    (per-bucket, correct)
  GIK APCP max:   129.15 mm   (cumulative, ~4.2x inflated)

cGAN output:
  Herbie path: max=3.22 mm/h, mean=0.088 mm/h
  GIK path:    max=6.90 mm/h, mean=0.314 mm/h  (2x higher due to inflated input)
  Output correlation: r=0.838 (paths disagree)
```

### With per-bucket APCP (correct)
```
APCP input correlation: r=1.000 (Herbie and GIK identical)
  Herbie APCP max: 30.66 mm
  GIK APCP max:    30.66 mm   (same source data, no conversion)

cGAN output:
  Herbie path: max=3.22 mm/h, mean=0.088 mm/h
  GIK path:    max=3.28 mm/h, mean=0.087 mm/h  (nearly identical)
  Output correlation: r=0.908 (paths agree, small GAN noise)
```

## How to Ensure Correct APCP for cGAN

### Option 1: Do NOT use `--cumulative_apcp` (recommended for cGAN)

When running the GIK pipeline for cGAN inference, use `--cgan_steps_only`
without `--cumulative_apcp`:

```bash
uv run run_gefs_gik_cgan_pipeline.py --date 20240520 --stages 1,2,3,4 \
    --cgan_steps_only --max_members 30
```

This streams only the 5 cGAN-needed steps (positions 10,12,14,16,18 →
hours 30,36,42,48,54) and writes the raw bucket-incremental values directly.

### Option 2: Herbie fetch (already correct)

```bash
uv run fetch_gefs_herbie_for_cgan.py --date 20240520 --max-members 30
```

Herbie fetches per-step GRIB messages which contain the bucket-incremental values
by default. No conversion needed.

### Option 3: Modify inference to handle cumulative (not recommended)

If you must use cumulative APCP input, you could modify the inference script to
convert back to per-bucket before normalization:

```python
# In run_gefs_inference_raw.py, after loading apcp data:
if field == "apcp":
    # Convert cumulative back to per-bucket (difference between consecutive steps)
    # data shape: (member, step, lat, lon) after transpose
    # step 0 = cumulative at hour_1, step 1 = cumulative at hour_2
    # Per-bucket = cumulative[step] - cumulative[step-1]
    # But we only have 2 steps, so: bucket = cumul[1] - cumul[0]
    # This is fragile — better to just feed raw bucket values
```

This is fragile and error-prone because:
- You'd need the cumulative value at the prior bucket boundary
- The two steps in each valid-time window may span bucket boundaries differently
- The training data used raw bucket values, so matching that format is safest

## When IS Cumulative APCP Useful?

The `--cumulative_apcp` flag is useful for **non-cGAN analysis** such as:
- Computing total rainfall over a forecast period (e.g., 24h or 48h totals)
- Exceedance probability plots (P(total rain > threshold))
- Comparing against rain gauge accumulations

It should NOT be used as cGAN input because the model was trained on per-bucket values.
