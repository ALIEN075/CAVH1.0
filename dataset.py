from torch.utils.data import Dataset
from osgeo import gdal
import torch
import numpy as np
import logging
import random
    
class GeoTIFFDataset(Dataset):
    def __init__(self, input_txt, label_txt, transform=None):
        self.input_paths = self._load_paths(input_txt)
        self.label_paths = self._load_paths(label_txt)
        self.transform = transform

    def _load_paths(self, txt_file):
        with open(txt_file, 'r') as f:
            paths = f.read().splitlines()
        return paths

    def _load_geotiff(self, path, bands):
        dataset = gdal.Open(path)
            # 检查文件是否成功打开
        if dataset is None:
            raise FileNotFoundError(f"GDAL 无法打开文件，请检查路径或文件完整性：{path}")

        data = np.zeros((bands, dataset.RasterYSize, dataset.RasterXSize), dtype=np.float32)

        for i in range(bands):
            band = dataset.GetRasterBand(i + 1)
            band_data = band.ReadAsArray()
        
            # 判断数据类型：uint16 则归一化到 [0, 1]，float32 则保持不变
            if band.DataType == gdal.GDT_UInt16:
                band_data = band_data.astype(np.float32) / 65536.0
            else:
                band_data = band_data.astype(np.float32)
            
            data[i, :, :] = band_data
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        return data

    def _load_label_geotiff(self, path):
        dataset = gdal.Open(path)
        label_data = dataset.GetRasterBand(1).ReadAsArray()
        return label_data

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        # 加载数据
        input = self._load_geotiff(self.input_paths[idx], bands=12)
        label = self._load_label_geotiff(self.label_paths[idx])
        
        # 应用数据增强
        if self.transform:
            input, label = self.transform(input, label)
        
        # 转换为张量
        input = torch.FloatTensor(input)
        label = torch.FloatTensor(label)
        
        return input, label
    