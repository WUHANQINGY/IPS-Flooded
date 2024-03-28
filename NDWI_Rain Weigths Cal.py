# -*- coding: utf-8 -*
import os
import os.path
import sys
from osgeo import gdal,osr
import numpy as np
#from sympy import *
import pandas as pd
from scipy.stats import multivariate_normal
from pyproj import Proj

def Raster_Reader(RasterFile):
    raster_df = gdal.Open(RasterFile, gdal.GA_ReadOnly)
    if raster_df is None:
        print('Cannot open ', RasterFile)
        sys.exit(1)
    return raster_df


def get_value_by_rowcol(raster_df, point_x, point_y):
    geotransform = raster_df.GetGeoTransform()
    a = np.array([[geotransform[1], geotransform[2]], [geotransform[4], geotransform[5]]])
    b = np.array([point_x - geotransform[0], point_y - geotransform[3]])
    row_col = np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解
    row = int(np.floor(row_col[1]))
    col = int(np.floor(row_col[0]))
    rows = raster_df.RasterYSize
    cols = raster_df.RasterXSize
    '''if (row > rows) or (col > cols):
        print('行列号超出图像范围 ')
        sys.exit(1)
    # img为栅格图像像元值填充的数组'''
    img = raster_df.GetRasterBand(1).ReadAsArray(0, 0, cols, rows)
    raster_value = img[row, col]
    return row, col,raster_value


def get_value_by_coordinates(raster_df, GPSlng, GPSlat):
    # 需要注意图像的坐标系，投影坐标系可能会转换出错
    pcs = osr.SpatialReference()
    pcs.ImportFromWkt(raster_df.GetProjection())
    gcs = pcs.CloneGeogCS()
    geotransform = raster_df.GetGeoTransform()
    cols = raster_df.RasterXSize
    rows = raster_df.RasterYSize
    img = raster_df.GetRasterBand(1).ReadAsArray(0, 0, cols, rows)
    raster_value = None

    '''将原始的经纬度坐标转换为投影坐标
    coordinates_geo是传入的参数，代表原始的经纬度坐标
    coordinates_geo[0]——经度，coordinates_geo[1]——纬度 '''
    ct = osr.CoordinateTransformation(gcs, pcs)
    coordinates_xy = ct.TransformPoint(GPSlng, GPSlat)
    x = coordinates_xy[0]
    y = coordinates_xy[1]
    z = coordinates_xy[2]

    '''根据GDAL的六参数模型geotransform[0-5]，将投影坐标转为图像上坐标（行列号）
        (x, y)——(row, col)
        :coordinates_xy[0]——投影坐标x
        :coordinates_xy[1]——投影坐标y
    '''
    a = np.array([[geotransform[1], geotransform[2]], [geotransform[4], geotransform[5]]])
    b = np.array([x - geotransform[0], y - geotransform[3]])
    row_col = np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解
    row = int(np.floor(row_col[1]))
    col = int(np.floor(row_col[0]))
    # 取行列号代表的像元的像元值
    RasterValue_byCoordinate = get_value_by_rowcol(raster_df, row, col)
    return row, col, RasterValue_byCoordinate


def Gaussian_value_Cal(N, M, mean, sigma, radius):
    """
       根据输入参数，生成高斯正态分布，并根据输入的坐标，计算对应的概率密度函数值
       :param N：正态分布的维度，这里是二维
       :param M：
       :param mean, sigma：均值和标准差
       :param radius: 以i为中选择的缓冲半径，为纳入计算范围的像元区域，则矩阵大小为[2*radius+1, 2*radius+1]
       return Z
       正态分布概率密度函数，根据输入的x，y坐标值，计算得到的概率密度,type为M*M维的数组
       """
    '''
    mean = np.zeros(N) + mean  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * sigma  # eye(N)生成N维对角矩阵，协方差矩阵，每个维度的方差都为 sigma
    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)

    # 生成二维网格平面
    # np.meshgrid()从坐标向量中返回坐标矩阵，X轴可取m个值 Y轴可取n个值则可以获得m*n个坐标
    # np.linspace(start, stop, M)返回的是 [start, stop]之间M个的均匀分布等差数列
    radius_value = radius*30
    X, Y = np.meshgrid(np.linspace(-radius_value, radius_value, M), np.linspace(-radius_value, radius_value, M))
    # 二维坐标数据
    d = np.dstack([X, Y])
    # 计算Z轴数据（高度数据），即为计算二维联合高斯概率
    Z = Gaussian.pdf(d)'''

    radius_value = radius * 30
    X, Y = np.meshgrid(np.linspace(-radius_value, radius_value, M), np.linspace(-radius_value, radius_value, M))
    h = 15000
    # 二维坐标数据
    # 计算Z轴数据（高度数据），即为计算二维联合高斯概率
    Z = np.exp(-((X - mean) ** 2 + (Y - mean) ** 2) / (2 * h * (sigma ** 2))) / (2 * np.pi * (sigma ** 2) * h)
    return Z


def PID_weight_Cal(MNDWI_df, row_i, col_i, radius):
    """
    计算每张PID图像对应的权重
    根据微博淹没点位i，取缓冲半径为R=360m，n=12个像元
    构建以i为中心的25*25(12+12+1)大小的像元矩阵，依次计算每个像元的MNDWI_values和高斯曲面值Gaussian_values
    :param row_i: 中心点微博点位i在MNDWI栅格图像上的行
    :param col_i: 中心点微博点位i在MNDWI栅格图像上的列
    :param radius: 以i为中选择的缓冲半径，为纳入计算范围的像元区域，则矩阵大小为[2*radius+1, 2*radius+1]
    :return: 微博点位i处的NDWI值加权值，是一个常数
    """
    rows = MNDWI_df.RasterYSize
    cols = MNDWI_df.RasterXSize
    img = MNDWI_df.GetRasterBand(1).ReadAsArray(0, 0, cols, rows)
    # result_cols, result_cols是缓冲区大小，也就是结果数组的行列大小
    result_rows = 2*radius + 1
    result_cols = 2*radius + 1
    MNDWI_values_ResultArea = np.array([[0.0] * result_cols] * result_rows)
    Gaussian_values = np.array([[0.0] * result_cols] * result_rows)
    N = 2
    M = 2 * radius + 1
    mean = 0
    sigma = 1
    Gaussian_values = Gaussian_value_Cal(N, M, mean, sigma, radius)*30
    # 函数返回根据输入的x，y坐标值，计算得到的概率密度,type为M*M维的数组
    for m in range(12):
        for n in range(12):
            row = row_i + m - 12
            col = col_i + n - 12
            if (row > rows) or (col > cols):
                print('行列号超出图像范围 ')
                sys.exit(1)
            # img为栅格图像像元值填充的数组
            MNDWI_values_ResultArea[m, n] = img[row, col]
    #MNDWI_values_ResultArea = 1-abs((MNDWI_values_ResultArea/np.maximum(MNDWI_values_ResultArea,-MNDWI_values_ResultArea).max()))

    weighted_MNDWI_i = (MNDWI_values_ResultArea.max() - MNDWI_values_ResultArea) / (MNDWI_values_ResultArea.max() - MNDWI_values_ResultArea.min())#逆向型指标
    weighted_MNDWI_i = (Gaussian_values * MNDWI_values_ResultArea).sum()
    # 这里计算第i张PID的权重，按照公式应该是PID点位周围的MNDWI和高斯概率求双重定积分，但是具体运算有点不太清楚(格网积分直接相乘，除非是利用坐标积分)
    return weighted_MNDWI_i


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


def PIDs_integrate(point_df, MNDWI_df,  raster_df, radius, PID_raster_rootdir, PID_result_path_1, PID_result_path_2):
    """
       将每张PIDS栅格图像进行综合，首先计算每张PID的权重
       读入PID点位的经纬度，以计算在MNDWI图像上的行列坐标
       调用权重计算函数，得到第n张PID图像的权重
       依次读入PIDs图像，进行加权综合
       加权综合生成最终的一张淹没概率图像，输入到路径
       :param row_i: 中心点微博点位i在MNDWI栅格图像上的行
       :param col_i: 中心点微博点位i在MNDWI栅格图像上的列
       :param radius: 以i为中选择的缓冲半径，为纳入计算范围的像元区域，则矩阵大小为[2*radius+1, 2*radius+1]
       :return: 微博点位i处的NDWI值加权值，是一个常数
       """
    point_num = point_df.shape[0]
    PID_weight_MNDWI = np.array([[0.0] * 1] * point_num)
    PID_weight_Rain = np.array([[0.0] * 1] * point_num)
    for i in range(point_num):
        # 依次读入微博数据点位，读取经纬度、水深等信息
        point_id = point_df.iloc[i, 0]
        point_lng = point_df.iloc[i, 1]
        point_lat = point_df.iloc[i, 2]
        #  [i, 1]是点位ID，[i,2]是GPSlng为经度，[i,3]是GPSlat为纬度，[i,4]是Depth为洪涝淹没水深
        # 函数get_value_by_coordinates()根据PID点位的经纬度，获取对应像元的行列数 row_i, col_i
        proj1 = Proj("epsg:32649")
        point_x, point_y = proj1(point_lng, point_lat, inverse=False)
        row_i, col_i, MNDWI_i= get_value_by_rowcol(MNDWI_df, point_x, point_y)
        PID_weight_MNDWI[i] = PID_weight_Cal(MNDWI_df, row_i, col_i, radius)
        #row_i_rain, col_i_rain, PID_weight_Rain[i] = get_value_by_rowcol(Rain_df, point_x, point_y)
    # normalized_weight_MNDWI = (PID_weight_MNDWI - PID_weight_MNDWI.min()) / (PID_weight_MNDWI.max() - PID_weight_MNDWI.min())
    # normalized_weight_Rain = (PID_weight_Rain - PID_weight_Rain.min()) / (PID_weight_Rain.max() - PID_weight_Rain.min())
    print('normalized_weight_MNDWI: ', PID_weight_MNDWI)
    #print('normalized_weight_Rain: ', PID_weight_Rain)
    num = 0
    # num用来计数，判断是哪一张PID图像，以便取对应的权重
    # raster_df为任意读取的一张PID图像，以获取图像参数，从而写入最终加权栅格图像
    cols = raster_df.RasterXSize
    rows = raster_df.RasterYSize
    projection = raster_df.GetProjection()
    geotransform = raster_df.GetGeoTransform()
    PIDs_raster_weighted = np.array([[0.0] * cols] * rows)
    path_list = os.listdir(PID_raster_rootdir)
    path_list.sort(key=lambda x: int(x.split('.tif')[0]))
    for site_data_name in path_list:
        print(site_data_name)
        PID_raster_path = os.path.join(PID_raster_rootdir, site_data_name)  # 获取完整路径
        PID_raster_df = Raster_Reader(PID_raster_path)
        PID_raster_values = PID_raster_df.GetRasterBand(1).ReadAsArray(0, 0, cols, rows)
        PIDs_raster_weighted = PIDs_raster_weighted + PID_raster_values * PID_weight_MNDWI[num]
        num = num + 1
    np.seterr(divide='ignore', invalid='ignore')
    normalized_raster_weighted = (PIDs_raster_weighted - PIDs_raster_weighted.min()) / (PIDs_raster_weighted.max() - PIDs_raster_weighted.min())
    # Raster_Write(PID_result_path_1, cols, rows, projection, geotransform, -999, PIDs_raster_weighted)
    Raster_Write(PID_result_path_2, cols, rows, projection, geotransform, -999, normalized_raster_weighted)


if __name__ == '__main__':
    PID_raster_path = r'E:\HNrainstorm\newtest\zzfactors\dem_ZZ.tif'
    PID_Point_Path = r'E:\HNrainstorm\newtest\extra-info\zz.xls'
    MNDWI_raster_path = r'E:\HNrainstorm\newtest\zzfactors\distance_zz.tif'
    # Rain_raster_path = r'E:\HNrainstorm\newtest\xxfactors\7.20rain_xx.tif'
    PID_raster_rootdir = r'E:\HNrainstorm\newtest\PID\PID_ONEPOINT_ALL\zz'
    # 注意文件读取顺序是否是正确的，比如10会在2前面
    PID_result_path_1 = r'E:\HNrainstorm\newtest\PID\ZZTEST\distance\distance_weighted.tif'
    PID_result_path_2 = r'E:\HNrainstorm\newtest\PID\ZZTEST\distance\distance_weighted_Normalized.tif'

    point_df = pd.read_excel(PID_Point_Path)
    MNDWI_df = Raster_Reader(MNDWI_raster_path)
    #Rain_df = Raster_Reader(Rain_raster_path)
    print('rows and cols of MNDWI: ', MNDWI_df.RasterYSize, MNDWI_df.RasterXSize)
    # raster_df为任意读取的一张PID图像，以获取图像参数，从而写入最终加权栅格图像
    raster_df = Raster_Reader(PID_raster_path)

    PIDs_integrate(point_df, MNDWI_df,  raster_df, 12, PID_raster_rootdir, PID_result_path_1, PID_result_path_2)
    # 这里row和cols是PID栅格图像的大小，也是最后生成加权图像的大小

