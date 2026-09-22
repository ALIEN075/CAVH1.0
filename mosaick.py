"""
Patch merging tool for GeoTIFF image tiles.

WHAT IT DOES
------------
Reconstructs large GeoTIFF mosaics from directories of overlapping small
patches (e.g. model inference output split into tiles). Overlapping regions
are blended with a Gaussian weight (center-heavy, edges tapered) so seams
between adjacent patches are smooth.

Patch filenames must end in "..._<row>_<col>.tif", where row/col identify
the patch's position in the original grid.

HOW TO USE
----------
1. Set INPUT_ROOT to a directory containing one subfolder per image, each
   subfolder holding that image's patches.
2. Set OUTPUT_ROOT to where the merged GeoTIFFs should be written. Each
   subfolder becomes one output file named "<subfolder>.tif".
3. Adjust original_width / original_height / patch_size / overlap_ratio in
   batch_merge_patches() to match how the patches were generated.
4. Run: python predict.py
"""

import os
import numpy as np
from osgeo import gdal, gdal_array


def create_weight_matrix(size):
    """Gaussian weight matrix (high in center, low at edges) for seamless blending."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xv, yv = np.meshgrid(x, y)
    weight = np.exp(-(xv**2 + yv**2))
    return weight


def merge_patches(
        patch_folder,
        output_path,
        original_width=3600,
        original_height=3600,
        patch_size=256,
        overlap_ratio=0.1):
    """Merge all patches in one folder into a single large GeoTIFF."""
    files = [f for f in os.listdir(patch_folder) if f.endswith('.tif')]
    if len(files) == 0:
        print(f"Warning: no TIFF files in {patch_folder}, skipping")
        return

    sample_path = os.path.join(patch_folder, files[0])
    sample = gdal.Open(sample_path)
    if sample is None:
        print(f"Error: could not open {sample_path}")
        return

    bands = sample.RasterCount
    dtype = sample.GetRasterBand(1).DataType
    proj = sample.GetProjection()
    geo = sample.GetGeoTransform()

    nodata_list = [sample.GetRasterBand(b + 1).GetNoDataValue() for b in range(bands)]
    sample = None

    stride = int(patch_size * (1 - overlap_ratio))

    # Derive the origin patch's row/col from its filename to compute the
    # merged image's geotransform origin.
    parts = files[0].replace(".tif", "").split("_")
    row0 = int(parts[-2])
    col0 = int(parts[-1])

    origin_x = geo[0] - col0 * stride * geo[1]
    origin_y = geo[3] - row0 * stride * geo[5]
    new_geo = list(geo)
    new_geo[0] = origin_x
    new_geo[3] = origin_y

    result = np.zeros((bands, original_height, original_width), dtype=np.float32)
    weight_sum = np.zeros((original_height, original_width), dtype=np.float32)
    weight = create_weight_matrix(patch_size)

    for file in files:
        path = os.path.join(patch_folder, file)
        parts = file.replace(".tif", "").split("_")
        row = int(parts[-2])
        col = int(parts[-1])

        y = row * stride
        x = col * stride

        # Clamp so the last row/col of patches stays inside the canvas.
        if y + patch_size > original_height:
            y = original_height - patch_size
        if x + patch_size > original_width:
            x = original_width - patch_size

        ds = gdal.Open(path)
        data = ds.ReadAsArray().astype(np.float32)
        ds = None

        if bands == 1:
            data = data[np.newaxis, :, :]

        for b in range(bands):
            result[b, y:y+patch_size, x:x+patch_size] += data[b] * weight
        weight_sum[y:y+patch_size, x:x+patch_size] += weight

    weight_sum[weight_sum == 0] = 1
    for b in range(bands):
        result[b] /= weight_sum

    result = result.astype(gdal_array.GDALTypeCodeToNumericTypeCode(dtype))

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_path,
        original_width,
        original_height,
        bands,
        dtype,
        options=['COMPRESS=DEFLATE']
    )
    out_ds.SetGeoTransform(new_geo)
    out_ds.SetProjection(proj)

    for b in range(bands):
        out_band = out_ds.GetRasterBand(b + 1)
        out_band.WriteArray(result[b])
        if nodata_list[b] is not None:
            out_band.SetNoDataValue(nodata_list[b])

    out_ds.FlushCache()
    out_ds = None
    print(f"Merge complete: {output_path}")


def batch_merge_patches(
        root_folder,
        output_root,
        original_width=3600,
        original_height=3600,
        patch_size=256,
        overlap_ratio=0.1):
    """Merge every subfolder of root_folder into its own output GeoTIFF."""
    os.makedirs(output_root, exist_ok=True)

    sub_folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
    if len(sub_folders) == 0:
        print("No subfolders found under the root directory.")
        return

    print(f"Found {len(sub_folders)} subfolders, starting batch merge...\n")

    for idx, sub_folder in enumerate(sub_folders, 1):
        patch_folder = os.path.join(root_folder, sub_folder)
        output_tif = os.path.join(output_root, f"{sub_folder}.tif")

        print(f"\n===== [{idx}/{len(sub_folders)}] {sub_folder} =====")
        merge_patches(
            patch_folder=patch_folder,
            output_path=output_tif,
            original_width=original_width,
            original_height=original_height,
            patch_size=patch_size,
            overlap_ratio=overlap_ratio
        )

    print("\nAll subfolders merged.")


# ====================== EDIT THESE PATHS ======================
INPUT_ROOT = r'D:\China_veg\2015tile'
OUTPUT_ROOT = r'D:\China_veg\2015tile_big'
# ================================================================

if __name__ == '__main__':
    gdal.UseExceptions()

    batch_merge_patches(
        root_folder=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        original_width=3600,
        original_height=3600,
        patch_size=256,
        overlap_ratio=0.1
    )