import os
from osgeo import gdal
import numpy as np
import torch
from U_swin_Net import UNet_Swin
import logging

def read_tif(filename):
    dataset = gdal.Open(filename)  

    im_width = dataset.RasterXSize  
    im_height = dataset.RasterYSize  

    im_geotrans = dataset.GetGeoTransform()  
    im_proj = dataset.GetProjection()  
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  

    del dataset
    return im_proj, im_geotrans, im_data

def write_img(filename, im_proj, im_geotrans, im_data, im_height):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    im_bands = 1  
    im_width = im_height

    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise Exception("Could not create GDAL driver")

    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    if dataset is None:
        raise Exception(f"Could not create file {filename}")

    try:
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
        data_to_write = im_data[0, 0, :, :]  # 提取出2D数组
        dataset.GetRasterBand(1).WriteArray(data_to_write)
        dataset.FlushCache()

    except Exception as e:
        print(f"Error writing file: {str(e)}")
        raise
    
    finally:
        dataset = None

def load_UnetSwin_model(model_path,nums = 12):
    net = UNet_Swin(channels = nums)
    net = net.eval()
    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict['model_state_dict'])
    net = net.cuda()
    return net

def load_geotiff(path, bands):
    dataset = gdal.Open(path)
    data = np.zeros((bands, dataset.RasterYSize, dataset.RasterXSize), dtype=np.float32)
    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        band_data = band.ReadAsArray()
    
        if band.DataType == gdal.GDT_UInt16:
            band_data = band_data.astype(np.float32) / 65536.0
        else:
            band_data = band_data.astype(np.float32)
        data[i, :, :] = band_data

        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logging.warning(f"Found NaN or Inf in {path}, band {i+1}")
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    return data

def predict_images(model, nums, img_path,predict_path):
    landsat_all = os.listdir(img_path)
    
    os.makedirs(predict_path, exist_ok=True)
    
    for file in landsat_all:
        print(f"Processing: {file}")
        
        landsat_file = os.path.join(img_path, file)

        landsat_data = load_geotiff(landsat_file, bands=nums)
             
        landsat_tensor = torch.FloatTensor(landsat_data).unsqueeze(0).cuda()
        
        with torch.no_grad():
            pr = model(landsat_tensor)
            pr = pr.cpu().numpy()
        
        proj, geotrans, _ = read_tif(landsat_file)
        output_path = os.path.join(predict_path, file)
        write_img(output_path, proj, geotrans, pr, pr.shape[-1])
        
        print(f"Saved prediction to: {output_path}")

def predict_images_from_txt(model, nums, txt_path, predict_path):
    """
    Read the image paths from the txt file, perform model predictions for each image, and save the results.
    :param model: The trained prediction model
    :param nums: The number of bands used / a list of bands
    :param txt_path:A txt file storing the image paths (one complete file path per line)
    :param predict_path: The root directory for outputting prediction results
    """
    os.makedirs(predict_path, exist_ok=True)
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        file_paths = [line.strip() for line in f.readlines() if line.strip()]
    
    for landsat_file in file_paths:
        if not os.path.exists(landsat_file):
            print(f"⚠️  文件不存在，跳过：{landsat_file}")
            continue
            
        print(f"Processing: {landsat_file}")
        
        landsat_data = load_geotiff(landsat_file, bands=nums)
             
        landsat_tensor = torch.FloatTensor(landsat_data).unsqueeze(0).cuda()
        
        with torch.no_grad():
            pr = model(landsat_tensor)
            pr = pr.cpu().numpy()
        
        proj, geotrans, _ = read_tif(landsat_file)
        
        file_name = os.path.basename(landsat_file)
        output_path = os.path.join(predict_path, file_name)
        
        write_img(output_path, proj, geotrans, pr, pr.shape[-1])
        
        print(f"✅ Saved prediction to: {output_path}")

if __name__ == "__main__":
    model_path = r'E:\LQH\EBD\Models_all\wdcy_veg.pth'

    txt_path = r"E:\LQH\EBD\subregion\2015\wdcy.txt"

    predict_path = r'D:\China_veg\2015\wdcy'
    model = load_UnetSwin_model(model_path,nums = 12)
    predict_images_from_txt(model, 12 ,txt_path, predict_path)