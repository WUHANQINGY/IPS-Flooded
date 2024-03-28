#	coding:utf-8


import time
import numpy as np
import os
import pandas as pd
#import gdal
import sys
from osgeo.gdalconst import *
from osgeo import gdal
from osgeo import osr
# from sklearn import preprocessing
# from math import radians, cos, sin, asin, sqrt
# from numba import jit


def Raster_Reader(RasterFile):
    raster_df = gdal.Open(RasterFile, gdal.GA_ReadOnly)
    if raster_df is None:
        print('Cannot open ', RasterFile)
        sys.exit(1)
    return raster_df


def Raster_Write(save_path, cols, rows, projection, geotransform, noDataValue, raster_values):
    # 新建一个tiff格式图像，并将代表像元值的数组写入图像，最后根据指定的路径存储图像。需要给定的参数包括：
    # rows、cols——图像行列大小，projection——投影坐标系，geotransform——地理空间坐标系
    # noDatavalue——空像元赋值，raster_value——代表像元值的数组，大小为图像行列大小
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    ds = driver.Create(save_path, cols, rows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)
    ds.GetRasterBand(1).SetNoDataValue(noDataValue)
    # raster_values是结果影像的像元值数组
    ds.GetRasterBand(1).WriteArray(raster_values)
    ds = None



if __name__ == '__main__':
    # 尝试先将所有PIDs微博点位生成的概率图进行简单的求平均合成
    # 方法是把所有的栅格图像像元值读入数组，直接在数组级别进行平均值计算，而不是采用循环
    DEM_path = r'E:\HNrainstorm\newtest3\xxfactors\dem.tif'
    rootdir = r'E:\HNrainstorm\newtest3\result'
    PID_save_path_1 = r'E:\HNrainstorm\newtest3\result_ALL.tif'
    #PID_save_path_2 = r'E:\HNrainstorm\newtest2\result\result.tif'
    raster_df = Raster_Reader(DEM_path)#读取DEM数据
    cols = raster_df.RasterXSize#存行、列号
    rows = raster_df.RasterYSize
    projection = raster_df.GetProjection()#存投影、地理坐标系
    geotransform = raster_df.GetGeoTransform()
    raster_mean = np.array([[0.0] * cols] * rows)#初始化一个行列数与DEN数据行列数相同的列表数组
    num = 0
# 多幅栅格求均值
    for site_data_name in os.listdir(rootdir):
        raster_path = os.path.join(rootdir, site_data_name)  # 获取完整路径
        if os.path.isdir(raster_path):
            for file in os.listdir(raster_path):
                file = os.path.join(raster_path, file)
                print(file)
                raster_df = Raster_Reader(file)
                raster_values = raster_df.GetRasterBand(1).ReadAsArray(0, 0, cols, rows)
        raster_df = Raster_Reader(raster_path)
        raster_values = raster_df.GetRasterBand(1).ReadAsArray(0, 0, cols, rows)
        raster_mean = raster_mean + raster_values
        num = num + 1
    print('num: ', num)
    raster_mean = raster_mean / num
    # 尝试再次标准化到0-1
    # 结果显示，标准化之后只是范围变成了0-1，像元之间的变化分布没变
    normalized_raster_mean = (raster_mean - raster_mean.min()) / (raster_mean.max() - raster_mean.min())
    # Raster_Write(PID_save_path_1, cols, rows, projection, geotransform, -999, raster_mean)
    Raster_Write(PID_save_path_1, cols, rows, projection, geotransform, -999, normalized_raster_mean)