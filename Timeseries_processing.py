# -*- coding: utf-8 -*-
"""
Temporal post-processing - batch version (multi-year x multi-tile).

WHAT IT DOES
------------
Expects an input directory structured as:
    ROOT_DIR/
        2015/ SDC30_EBD_V001_<TILE>_2015.tif
        2016/ SDC30_EBD_V001_<TILE>_2016.tif
        ...
        2024/ SDC30_EBD_V001_<TILE>_2024.tif

For every TILE present in all year subfolders, its 10-year time series is
read and smoothed pixel-by-pixel:
  1. Forest threshold (FOREST_MIN = 3 m): only years above this value are
     smoothed; years at/below it pass through unchanged and naturally mark
     segment boundaries from planting/harvest.
  2. Disturbance detection (drop >= DIST_DROP_M for PERSIST_YEARS in a row),
     including clear-cuts that fall below the forest threshold.
  3. Per-segment Mann-Kendall trend test (exact critical value, adaptive to
     valid sample size) -> 3-year SMA if trend is significant, else 5-year SMA.
  4. Segment boundaries are extrapolated with a segment-specific Theil-Sen
     slope before the SMA is applied.
  5. Output is clipped to non-negative values.

Each tile is written as one 10-band UInt16 GeoTIFF (value = smoothed x 100,
NoData = 65535) named CAVH_V001_<TILE>.tif in OUTPUT_DIR. This batch version
does not output the auxiliary disturbance-year layer.

All I/O uses GDAL. Each tile is processed in spatial blocks (BLOCK) to limit
memory use; tiles are independent, so one tile failing does not stop the batch.

Dependencies: pip install numpy gdal  (or: conda install -c conda-forge gdal)

HOW TO USE
----------
1. Edit ROOT_DIR, OUTPUT_DIR, and YEARS in the CONFIG section below.
2. Adjust FOREST_MIN / DIST_DROP_M / PERSIST_YEARS / WIN_TREND / WIN_NOTREND
   if your data or use case needs different thresholds.
3. Run: python timeseries_postprocess.py
   Progress prints every LOG_EVERY tiles; failures are logged and skipped
   rather than stopping the run.
"""

from __future__ import annotations

import os
import re
import glob
import time
import traceback

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()


# ==============================================================================
#                                   CONFIG
# ==============================================================================

ROOT_DIR = r"D:\China_veg\all"
YEARS = list(range(2015, 2025))

IN_NAME_RE = r"^SDC30_EBD_V001_(?P<tile>.+)_{year}\.tif$"
IN_GLOB_TMPL = "SDC30_EBD_V001_*_{year}.tif"

OUTPUT_DIR = r"D:\China_veg\CAVH_V001"
OUT_NAME_TMPL = "CAVH_V001_{tile}.tif"

SKIP_EXISTING = True          # skip tiles whose output already exists (resumable)
EXPECTED_TILE_COUNT = 1216    # informational only, does not affect processing

FOREST_MIN = 3.0

DIST_DROP_M = 7.0
PERSIST_YEARS = 2
ALLOW_END_DISTURBANCE = True

WIN_TREND = 3
WIN_NOTREND = 5
ALPHA = 0.05

FILL_GAPS = False

SCALE = 100.0
OUT_NODATA = 65535
OUT_VMAX = 65534
COMPRESS = "DEFLATE"

BLOCK = 512
LOG_EVERY = 20


# ==============================================================================
#                       Mann-Kendall exact critical value
# ==============================================================================

def mk_exact_critical_S(n: int, alpha: float = 0.05):
    """Exact minimum significant |S| for a two-sided Mann-Kendall test at sample size n."""
    if n < 3:
        return None
    poly = np.array([1.0])
    for k in range(1, n + 1):
        poly = np.convolve(poly, np.ones(k))
    total = poly.sum()
    cum = np.cumsum(poly)
    N = n * (n - 1) // 2
    for s in range(0, N + 1):
        if (N - s) % 2 != 0:
            continue
        d = (N - s) // 2
        p_two_sided = 2.0 * cum[d] / total
        if p_two_sided <= alpha:
            return s
    return None


def build_crit_lut(n_max: int, alpha: float) -> np.ndarray:
    lut = np.full(n_max + 1, 10 ** 6, dtype=np.int32)
    for n in range(3, n_max + 1):
        c = mk_exact_critical_S(n, alpha)
        if c is not None:
            lut[n] = c
    return lut


# ==============================================================================
#                              CORE ALGORITHM
# ==============================================================================

def compute_mk_S(arr: np.ndarray, forest: np.ndarray):
    n = arr.shape[0]
    S = np.zeros(arr.shape[1:], dtype=np.int32)
    with np.errstate(invalid="ignore"):
        for i in range(n - 1):
            for j in range(i + 1, n):
                m = forest[i] & forest[j]
                d = arr[j] - arr[i]
                S += np.where(m, np.sign(d), 0).astype(np.int32)
    n_valid = forest.sum(axis=0).astype(np.int32)
    return S, n_valid


def detect_disturbance(arr, finite, forest, drop_thr, persist, allow_end):
    """Disturbance year needs a forested prior year; only validity is required
    for the drop year and the persistence window (so clear-cuts count)."""
    n = arr.shape[0]
    dist = np.zeros(arr.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        for t in range(1, n):
            base = arr[t - 1] - drop_thr
            cond = forest[t - 1] & finite[t] & (arr[t] < base)
            checked = 0
            for k in range(t, t + persist):
                if k < n:
                    cond &= finite[k] & (arr[k] < base)
                    checked += 1
            if checked < persist and not allow_end:
                continue
            dist[t] = cond
    return dist


def segment_bounds(dist: np.ndarray, forest: np.ndarray):
    """Split into contiguous forest segments at disturbance years / non-forest
    years; returns start/end/seg_id/break_start."""
    n = dist.shape[0]
    idx = np.broadcast_to(
        np.arange(n, dtype=np.int32).reshape(n, 1, 1), dist.shape)

    prev_forest = np.zeros_like(forest)
    prev_forest[1:] = forest[:-1]

    break_start = dist | (forest & ~prev_forest)
    break_start[0] = False

    start = np.maximum.accumulate(
        np.where(break_start, idx, 0), axis=0).astype(np.int32)

    boundary = dist | (~forest)
    nxt = np.where(boundary, idx, n).astype(np.int32)
    end = np.empty_like(nxt)
    end[n - 1] = n
    for t in range(n - 2, -1, -1):
        end[t] = np.minimum(end[t + 1], nxt[t + 1])
    end = (end - 1).astype(np.int32)

    seg_id = np.cumsum(break_start, axis=0).astype(np.int32)
    return start, end, seg_id, break_start


def theilsen_per_segment(arr, forest, seg_id):
    """Median Theil-Sen slope computed independently per segment, broadcast
    back to every time step."""
    n, H, W = arr.shape
    pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]

    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.empty((len(pairs), H, W), dtype=np.float32)
        for p, (i, j) in enumerate(pairs):
            raw[p] = (arr[j] - arr[i]) / float(j - i)

    n_seg = int(seg_id.max()) + 1
    slope_by_g = np.zeros((max(n_seg, n), H, W), dtype=np.float32)

    buf = np.empty_like(raw)
    for g in range(n_seg):
        memb = (seg_id == g) & forest
        if not memb.any():
            continue
        for p, (i, j) in enumerate(pairs):
            ok = memb[i] & memb[j]
            buf[p] = np.where(ok, raw[p], np.nan)
        with np.errstate(invalid="ignore"):
            s = np.nanmedian(buf, axis=0)
        slope_by_g[g] = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    return np.take_along_axis(slope_by_g, seg_id.astype(np.intp), axis=0)


def adaptive_sma(arr, forest, start, end, slope, half_w):
    """Symmetric moving average within each segment; boundaries borrow one
    Theil-Sen-extrapolated virtual point."""
    n, H, W = arr.shape

    h_start = np.take_along_axis(arr, np.clip(start, 0, n - 1).astype(np.intp), axis=0)
    h_end = np.take_along_axis(arr, np.clip(end, 0, n - 1).astype(np.intp), axis=0)

    out = np.full((n, H, W), np.nan, dtype=np.float32)

    for t in range(n):
        st, en = start[t], end[t]
        hs, he, sl = h_start[t], h_end[t], slope[t]

        k = np.minimum(half_w, np.minimum(t - st + 1, en - t + 1))

        acc = np.zeros((H, W), dtype=np.float64)
        cnt = np.zeros((H, W), dtype=np.float64)

        for d in range(-2, 3):
            usable = (abs(d) <= k)
            if not usable.any():
                continue
            u = t + d

            if 0 <= u < n:
                m_in = usable & (u >= st) & (u <= en) & forest[u]
                if m_in.any():
                    acc += np.where(m_in, arr[u], 0.0)
                    cnt += m_in

            if d < 0:
                vleft = hs - sl
                m_l = usable & (u == st - 1) & np.isfinite(vleft)
                if m_l.any():
                    acc += np.where(m_l, vleft, 0.0)
                    cnt += m_l

            if d > 0:
                vright = he + sl
                m_r = usable & (u == en + 1) & np.isfinite(vright)
                if m_r.any():
                    acc += np.where(m_r, vright, 0.0)
                    cnt += m_r

        with np.errstate(invalid="ignore", divide="ignore"):
            sm = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
        out[t] = sm.astype(np.float32)

    return out


def process_block(arr, crit_lut):
    """Full post-processing for one block. arr: (n,H,W) float32, NoData already NaN."""
    n = arr.shape[0]
    finite = np.isfinite(arr)
    forest = finite & (arr > FOREST_MIN)

    S, n_valid = compute_mk_S(arr, forest)
    crit = crit_lut[np.clip(n_valid, 0, len(crit_lut) - 1)]
    has_trend = np.abs(S) >= crit

    dist = detect_disturbance(arr, finite, forest, DIST_DROP_M, PERSIST_YEARS,
                              ALLOW_END_DISTURBANCE)
    start, end, seg_id, break_start = segment_bounds(dist, forest)
    any_break = break_start.any(axis=0)

    slope = theilsen_per_segment(arr, forest, seg_id)

    half_w = np.where(has_trend | any_break,
                      WIN_TREND // 2, WIN_NOTREND // 2).astype(np.int32)

    out = adaptive_sma(arr, forest, start, end, slope, half_w)
    out = np.where(forest, out, arr)

    if not FILL_GAPS:
        out = np.where(finite, out, np.nan)

    with np.errstate(invalid="ignore"):
        out = np.where(np.isfinite(out), np.maximum(out, 0.0), np.nan)

    return out.astype(np.float32)


# ==============================================================================
#                    TILE DISCOVERY (scan year subfolders)
# ==============================================================================

def discover_tiles(root_dir, years):
    """
    Scan root_dir/<year>/ for files matching SDC30_EBD_V001_<TILE>_<year>.tif.
    Returns {tile_id: {year: filepath}} plus a list of tiles missing some years.
    """
    tiles: dict[str, dict[int, str]] = {}
    for y in years:
        year_dir = os.path.join(root_dir, str(y))
        if not os.path.isdir(year_dir):
            print(f"[warning] year directory not found, skipped: {year_dir}")
            continue
        pattern = os.path.join(year_dir, IN_GLOB_TMPL.format(year=y))
        name_re = re.compile(IN_NAME_RE.format(year=y))
        for p in glob.glob(pattern):
            m = name_re.match(os.path.basename(p))
            if not m:
                continue
            tile_id = m.group("tile")
            tiles.setdefault(tile_id, {})[y] = p

    complete = {tid: d for tid, d in tiles.items() if all(y in d for y in years)}
    incomplete = {tid: d for tid, d in tiles.items() if tid not in complete}
    return complete, incomplete


# ==============================================================================
#                          GDAL READ/WRITE HELPERS
# ==============================================================================

def open_and_check(ordered_paths):
    """Open all input files for one tile in year order; verify grid/projection match."""
    ds_list = []
    for p in ordered_paths:
        ds = gdal.Open(p, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"GDAL could not open: {p}")
        ds_list.append(ds)

    ref = ds_list[0]
    W, H = ref.RasterXSize, ref.RasterYSize
    gt_ref = ref.GetGeoTransform()
    proj_ref = ref.GetProjection()

    for ds, p in zip(ds_list, ordered_paths):
        if ds.RasterXSize != W or ds.RasterYSize != H:
            raise ValueError(f"Raster size mismatch: {p}")
        gt = ds.GetGeoTransform()
        if any(abs(a - b) > 1e-9 for a, b in zip(gt, gt_ref)):
            raise ValueError(f"Geotransform mismatch: {p}")
        if ds.GetProjection() != proj_ref:
            raise ValueError(f"Projection mismatch: {p}")

    nodatas = [ds.GetRasterBand(1).GetNoDataValue() for ds in ds_list]
    return ds_list, W, H, gt_ref, proj_ref, nodatas


def create_output_dataset(path, W, H, n_bands, gt, proj, nodata,
                          gdal_dtype=gdal.GDT_UInt16,
                          compress=COMPRESS, block=256):
    driver = gdal.GetDriverByName("GTiff")
    creation_opts = [
        f"COMPRESS={compress}",
        "PREDICTOR=2",
        "TILED=YES",
        f"BLOCKXSIZE={block}",
        f"BLOCKYSIZE={block}",
        "BIGTIFF=IF_SAFER",
    ]
    ds = driver.Create(path, W, H, n_bands, gdal_dtype, options=creation_opts)
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    for b in range(1, n_bands + 1):
        ds.GetRasterBand(b).SetNoDataValue(nodata)
    return ds


# ==============================================================================
#                            SINGLE-TILE PROCESSING
# ==============================================================================

def process_one_tile(tile_id, files_by_year, years, crit_lut, block=BLOCK):
    """
    Process one tile's full time series and write a 10-band UInt16 GeoTIFF.
    Returns (status, message) where status is one of "ok", "skipped", "failed".
    """
    out_path = os.path.join(OUTPUT_DIR, OUT_NAME_TMPL.format(tile=tile_id))
    if SKIP_EXISTING and os.path.exists(out_path):
        return "skipped", out_path

    ordered_paths = [files_by_year[y] for y in years]
    n = len(years)

    ds_list, W, H, gt, proj, in_nodatas = open_and_check(ordered_paths)

    dst = create_output_dataset(out_path, W, H, n, gt, proj, OUT_NODATA,
                                gdal_dtype=gdal.GDT_UInt16,
                                compress=COMPRESS, block=256)

    in_bands = [ds.GetRasterBand(1) for ds in ds_list]

    n_bx = int(np.ceil(W / block))
    n_by = int(np.ceil(H / block))

    for by in range(n_by):
        for bx in range(n_bx):
            col_off, row_off = bx * block, by * block
            w = min(block, W - col_off)
            h = min(block, H - row_off)

            arr = np.empty((n, h, w), dtype=np.float32)
            for i, band in enumerate(in_bands):
                blk = band.ReadAsArray(col_off, row_off, w, h).astype(np.float32)
                nd = in_nodatas[i]
                if nd is not None:
                    blk[blk == nd] = np.nan
                arr[i] = blk

            if np.isfinite(arr).any() and (arr > FOREST_MIN).any():
                out = process_block(arr, crit_lut)
            else:
                # Entirely non-forest / no valid data: pass through, clip to non-negative.
                with np.errstate(invalid="ignore"):
                    out = np.where(np.isfinite(arr), np.maximum(arr, 0.0), np.nan)
                out = out.astype(np.float32)

            with np.errstate(invalid="ignore"):
                scaled = np.rint(out * SCALE)
                scaled = np.clip(scaled, 0, OUT_VMAX)
            out_u16 = np.where(np.isfinite(scaled), scaled,
                               OUT_NODATA).astype(np.uint16)

            for b in range(n):
                dst.GetRasterBand(b + 1).WriteArray(
                    out_u16[b], xoff=col_off, yoff=row_off)

    for i, y in enumerate(years, start=1):
        band = dst.GetRasterBand(i)
        band.SetDescription(str(y))
        band.SetMetadataItem("scale_factor", str(SCALE))
        band.FlushCache()

    dst.SetMetadataItem("SCALE_FACTOR", str(SCALE))
    dst.SetMetadataItem(
        "METHOD", "FORMS-T style temporal post-processing (Schwartz et al., 2025 RSE)")
    dst.SetMetadataItem("FOREST_MIN", str(FOREST_MIN))
    dst.SetMetadataItem("MK_CRITICAL_S", str(int(crit_lut[n])))
    dst.SetMetadataItem("DISTURBANCE_DROP", str(DIST_DROP_M))
    dst.SetMetadataItem("WINDOW_TREND", str(WIN_TREND))
    dst.SetMetadataItem("WINDOW_NOTREND", str(WIN_NOTREND))
    dst.SetMetadataItem("DATA_TYPE", "UInt16 (non-negative)")
    dst.SetMetadataItem("YEARS", ",".join(str(y) for y in years))
    dst.SetMetadataItem("TILE_ID", tile_id)

    dst.FlushCache()
    dst = None
    for ds in ds_list:
        ds = None  # noqa: F841

    return "ok", out_path


# ==============================================================================
#                                    MAIN
# ==============================================================================

def main():
    t0 = time.time()
    print("=" * 72)
    print("FORMS-T style temporal post-processing - batch version (multi-tile)")
    print("=" * 72)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[scan] root directory: {ROOT_DIR}")
    complete, incomplete = discover_tiles(ROOT_DIR, YEARS)
    tile_ids = sorted(complete.keys())

    print(f"[scan] complete tiles (all {len(YEARS)} years present): {len(tile_ids)}  "
          f"(expected ~{EXPECTED_TILE_COUNT})")
    if incomplete:
        print(f"[warning] {len(incomplete)} tiles are missing some years, skipped, e.g.:")
        for tid in list(incomplete.keys())[:5]:
            missing = [y for y in YEARS if y not in incomplete[tid]]
            print(f"        {tid}: missing years {missing}")

    if not tile_ids:
        raise RuntimeError("No complete tiles found; check ROOT_DIR / file naming.")

    n_years = len(YEARS)
    crit_lut = build_crit_lut(n_years, ALPHA)
    print(f"\n[MK  ] n={n_years}, alpha={ALPHA}, two-sided critical value |S| >= {crit_lut[n_years]}")
    print(f"[forest] threshold > {FOREST_MIN}; disturbance drop {DIST_DROP_M} m / persist {PERSIST_YEARS} yr")
    print(f"[output] UInt16, value = max(smoothed,0) x {int(SCALE)}, NoData = {OUT_NODATA}, "
          f"skip existing = {SKIP_EXISTING}")
    print(f"[output dir] {OUTPUT_DIR}\n")

    n_ok = n_skip = n_fail = 0
    fail_log = []

    for idx, tile_id in enumerate(tile_ids, start=1):
        try:
            status, info = process_one_tile(tile_id, complete[tile_id], YEARS, crit_lut)
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            fail_log.append((tile_id, str(e)))
            print(f"       [failed] tile={tile_id}: {e}")
            traceback.print_exc(limit=1)

        if idx % LOG_EVERY == 0 or idx == len(tile_ids):
            el = time.time() - t0
            eta = el / idx * (len(tile_ids) - idx)
            print(f"       {idx:>5}/{len(tile_ids)}  ok={n_ok} skipped={n_skip} "
                  f"failed={n_fail}  elapsed {el:7.1f}s  ETA {eta:7.1f}s")

    print("\n" + "=" * 72)
    print(f"[done] ok={n_ok}  skipped={n_skip}  failed={n_fail}  total={len(tile_ids)} tiles")
    print(f"[output dir] {OUTPUT_DIR}")
    if fail_log:
        print(f"[failures] {len(fail_log)} total:")
        for tid, msg in fail_log[:20]:
            print(f"        {tid}: {msg}")
        if len(fail_log) > 20:
            print(f"        ... {len(fail_log) - 20} more omitted, check the log")
    print(f"[total time] {time.time() - t0:.1f} s")
    print("=" * 72)


if __name__ == "__main__":
    main()