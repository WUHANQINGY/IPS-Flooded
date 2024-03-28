#	coding:utf-8
import time
import numpy as np
import math
import os
import pandas as pd
import sys
from osgeo import gdal
from osgeo import osr
from math import radians, cos, sin, asin, sqrt


# 目标：
# 已知一个微博淹没点的经纬度，以这个点位为中心，生成研究区范围（固定的范围）的PID图
# 所需数据：研究区DEM栅格图像，微博点位经纬度和淹没水深
# 步骤：依次读入微博点位数据
# 对每个点位i，读入研究区范围影像（可以以DEM栅格图为底图），遍历栅格像元
# 对每个像元j(x,y)，根据DEM数据和微博点位数据，工具公式计算PID(j)，存入数组，之后直接改写DEM图的像元值即可
# 最终形成对应于某个点位i的PID栅格图


def Raster_Reader(RasterFile):
    raster_df = gdal.Open(RasterFile, gdal.GA_ReadOnly)
    if raster_df is None:
        print('Cannot open ', RasterFile)
        sys.exit(1)
    return raster_df


'''def get_value_byimagexy(raster_df, point_x, point_y):
    geotransform = raster_df.GetGeoTransform()
    a = np.array([[geotransform[1], geotransform[2]], [geotransform[4], geotransform[5]]])
    b = np.array([point_x - geotransform[0], point_y - geotransform[3]])
    row_col = np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解
    row = int(np.floor(row_col[1]))
    col = int(np.floor(row_col[0]))
    rows = raster_df.RasterYSize
    cols = raster_df.RasterXSize
    if (row > rows) or (col > cols):
        print('行列号超出图像范围 ')
        sys.exit(1)
    # img为栅格图像像元值填充的数组
    img = raster_df.GetRasterBand(1).ReadAsArray(0, 0, cols, rows)
    raster_value = img[row, col]
    return raster_value'''
def get_value_by_rowcol(raster_df, row, col):
    rows = raster_df.RasterYSize
    cols = raster_df.RasterXSize
    if (row > rows) or (col > cols):
        print('行列号超出图像范围 ')
        sys.exit(1)
    # img为栅格图像像元值填充的数组
    img = raster_df.GetRasterBand(1).ReadAsArray(0, 0, cols, rows)
    raster_value = img[row, col]
    return raster_value

def get_value_by_coordinates(raster_df, GPSlng, GPSlat):
    # 直接根据图像坐标，或者依据GDAL的六参数模型将给定的投影、地理坐标转为影像图上坐标后，返回对应像元的像元值
    # :return:指定坐标的像元值'''

    # 传入的参数raster_df表示gdal数据集，读取数据集中的信息：
    # gcs——地理空间坐标系，pcs——投影坐标系
    # cols、rows——栅格影像的大小相关信息
    # img——将栅格图像一个波段的像元值读取为数组形式
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
    coordinates_xy = ct.TransformPoint(GPSlat, GPSlng)#注意TransformPoint（纬度，经度）
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
    print('row and col of PID: ', row, col)
    RasterValue_byCoordinate = get_value_by_rowcol(raster_df, row, col)
    return RasterValue_byCoordinate


'''def imagexy2geo(raster_df, row, col):
    根据GDAL的六参数模型将影像图上坐标（行列号）转为地理坐标（经纬度）
    先转换为投影坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y) 
    trans = raster_df.GetGeoTransform()
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]

    # 将投影坐标转换为经纬度坐标
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(raster_df.GetProjection())
    geosrs = prosrs.CloneGeogCS()

    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(px, py)
    return coords[0], coords[1]
'''

'''def flood_depth_DHij(dem_i, dem_j, depth_i):
    # DEM栅格图像默认的nodata是-999（若设置过nodata值，替换-999即可），所以当dem_j为-999时，该像元为空，直接计算会出错，而是应该给depth_j赋值为0
    if dem_j == -999:
        depth_j = 0.0
    elif (dem_i-dem_j) >= 0.0:
        #depth_j = depth_i + dem_i - dem_j
        depth_j =  dem_i - dem_j
    else:
        depth_j = 0.0
    return depth_j'''

# @jit(nopython=True)
def PID_Cal(cols, rows, point_lng, point_lat,depth_i, dem_i, DEM_GeoCoordinates, DEM_raster_values):
    # 读取DEM栅格图像的参数，包括行列数、投影、坐标信息等，由于PIDs和DEM是同一个研究区，所以新建PIDs图像采用DEM的参数
    # 依次读入微博数据点位，读取经纬度、水深等信息
    # 新建数组，用来存放大小和研究图像像元数量一致
    new_raster_values = np.array([[0.0] * cols] * rows)
    normalized_new_raster = np.array([[0.0] * cols] * rows)
    depth_js = np.array([[0.0] * cols] * rows)
    # 先把DEM栅格图像中的nodata值置为8888，这样方便后续的计算，因为如果是-999（此处设置过nodata值为0，替换-999即可），会和判断条件搞混
    #DEM_raster_values = np.where(DEM_raster_values == 0, 8888, DEM_raster_values)
    # 计算depth_j，并存入数组
    depth_js = dem_i + depth_i - DEM_raster_values
    #depth_js = dem_i  - DEM_raster_values
    depth_js = np.where(depth_js > 0, depth_js, 0)

    # 下面计算球面距离，由于距离计算比较复杂，没办法像上面一样对整个数组进行计算了，只有循环了
    EucDistance_ijs = np.array([[1.0] * cols] * rows)
    for j in range(rows):
        for k in range(cols):
            # 计算球面距离，两个点分别为PID微博点位，第[j,k]个栅格像元中心点位，形式都是经纬度坐标
            lon1 = point_lng
            lat1 = point_lat
            lon2 = DEM_GeoCoordinates[j][ 2 * k+1]
            lat2 = DEM_GeoCoordinates[j][ 2 * k]
            # 将十进制度数转化为弧度
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            r = 6371  # 地球平均半径，单位为公里
            #EucDistance_ijs[j, k] = c * r * 1000  # euc_distance是最后的距离，单位是m
            EucDistance_ijs[j, k] = math.sqrt(c * r * 1000)  # euc_distance是最后的距离，单位是m
            # 这里尝试将球面距离EucDistance开根号，降低逆距离加权的影响，不然算出来远距离的淹没范围太小

    # 计算出 depth_j 和 EucDistance_ij两个数组之后，可以直接对数组进行运算，然后单位都是m，所以直接相除可以消除量纲
    # 此外，最后的像元值需要统一标准化到[0,1]之间
    new_raster_values = depth_js / EucDistance_ijs
    normalized_new_raster = (new_raster_values - new_raster_values.min()) / (new_raster_values.max() - new_raster_values.min())
    # normalized_new_raster = preprocessing.MinMaxScaler().fit_transform(new_raster_values)
    print('shape of normalized_new_raster: ', normalized_new_raster.shape)

    return new_raster_values, normalized_new_raster
    # DEM_raster_values表示输入的DEM的值数组, new_raster_values表示PID计算的原始值数组,
    # normalized_new_raster表示PID原始值标准化到０－１区间的值数组

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
def DEM_Coordinates_Cal(raster_df):
    # 计算DEM栅格每个像元中心点的经纬度数据，存入数组DEM_GeoCoordinates，大小为[rows, 2*cols]
    cols = raster_df.RasterXSize
    rows = raster_df.RasterYSize
    DEM_GeoCoordinates = np.array([[0.0] * (2 * cols)] * rows)
    print('shape of DEM_GeoCoordinates: ', DEM_GeoCoordinates.shape)# 输出几行几列
    for j in range(rows):
        for k in range(cols):
            # 根据GDAL的六参数模型将影像图上坐标（行列号）先转换为投影坐标
            trans = raster_df.GetGeoTransform()
            px = trans[0] + k * trans[1] + j * trans[2]
            py = trans[3] + k * trans[4] + j * trans[5]
            # 将投影坐标转换为经纬度坐标
            prosrs = osr.SpatialReference()
            prosrs.ImportFromWkt(raster_df.GetProjection())
            geosrs = prosrs.CloneGeogCS()
            ct = osr.CoordinateTransformation(prosrs, geosrs)
            coords = ct.TransformPoint(px,py)
            DEM_GeoCoordinates[j, 2 * k] =coords[0]
            DEM_GeoCoordinates[j, 2 * k + 1]=coords[1]
    return DEM_GeoCoordinates

def PIDs_Processing(raster_df, point_df, DEM_GeoCoordinates, PIDs_path):
    # 读取DEM栅格图像的参数，包括行列数、投影、坐标信息等，由于PIDs和DEM是同一个研究区，所以新建PIDs图像采用DEM的参数

    point_num = point_df.shape[0]
    print('shape of point_df: ', point_num)
    cols = raster_df.RasterXSize
    rows = raster_df.RasterYSize
    projection = raster_df.GetProjection()
    geotransform = raster_df.GetGeoTransform()
    normalized_PID_values = np.array([[0.0] * cols] * rows)
    DEM_raster_values = raster_df.GetRasterBand(1).ReadAsArray(0, 0, cols, rows)
    print('DEM shape: '+'normalized_PID_values shape:', DEM_raster_values.shape,normalized_PID_values.shape)
    print('rows and cols: ', rows, cols)

    for i in range(point_num):
        # 依次读入微博数据点位，读取经纬度、水深等信息
        point_id = point_df.iloc[i, 0]
        point_lng = point_df.iloc[i, 1]
        point_lat = point_df.iloc[i, 2]
        depth_i = point_df.iloc[i, 3]
        print(point_df.iloc[i, 0], point_df.iloc[i, 1], point_df.iloc[i, 2], point_df.iloc[i, 3])
        #  [i, 1]是点位ID，[i,2]是GPSlng为经度，[i,3]是GPSlat为纬度，[i,4]是Depth为洪涝淹没水深
        # 函数get_value_by_coordinates()根据PID点位的经纬度，获取对应像元的DEM值
        dem_i = get_value_by_coordinates(raster_df,point_lng,point_lat)
        print('dem_i: ', dem_i)
        # 调用PID_Cal()函数，计算每个微博点位对应的PID值，并写入新的栅格图像
        PID_values, normalized_PID_values = PID_Cal(cols, rows, point_lng, point_lat, depth_i, dem_i,
                                                    DEM_GeoCoordinates, DEM_raster_values)

        # j,k循环结束，则对应一个微博点位point的PIDs图像计算完成，下面需要按照point的索引，新建对应的栅格文件，并写入栅格图像
        PID_save_path = os.path.join(PIDs_path, (str(point_id) + '.tif'))
        Raster_Write(PID_save_path, cols, rows, projection, geotransform, -999, normalized_PID_values)
        print('This is the ', i + 1, 'of all PIDs points !')

# 将DEM栅格值、PID栅格值、标准化之后的PID栅格值，写入记录表格
        '''data_1 = pd.DataFrame(DEM_raster_values)
        data_2 = pd.DataFrame(PID_values)
        data_3 = pd.DataFrame(normalized_PID_values)
        PID_excel_path = os.path.join(PIDs_path, (str(point_id) + '.xlsx'))
        writer = pd.ExcelWriter(PID_excel_path)  # 写入Excel文件
        data_1.to_excel(writer, 'DEM_raster_values', float_format='%.5f')
        data_2.to_excel(writer, 'PID_values', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
        data_3.to_excel(writer, 'normalized_PID_values', float_format='%.5f')
        writer.save()'''

if __name__ == '__main__':
    Weibo_Point_Path = r'E:\HNrainstorm\newtest3\testpoint\xx-1102.xlsx'
    DEM_path = r'E:\HNrainstorm\newtest3\xxfactors\dem.tif'
    PIDs_path = r'E:\HNrainstorm\newtest3\PID'
    # Weibo_Point_Path = r'F:\RSData-Processing\Data Fusion\Weibo Data\for test_1.csv'
    # DEM_path = r'F:\RSData-Processing\Data Fusion\Related Maps\ForTest_DEM.tif'
    # PIDs_path = r'F:\RSData-Processing\Data Fusion\PIDs\For test'
    # DEM_GeoCoordinates_path = r'F:\RSData-Processing\Data Fusion\PIDs\DEM_ForTest_GeoCoordinates.xlsx'

    start = time.time()

    raster_df = Raster_Reader(DEM_path)
    point_df = pd.read_excel(Weibo_Point_Path,sheet_name='Sheet3')
    # read_excel()函数参数，header = None，index_col = None表示读取excel不读取行、列索引
    # header默认为0，index_col默认为None，具体应该根据excel数据为准
    DEM_GeoCoordinates = DEM_Coordinates_Cal(raster_df)
    print('type of DEM_GeoCoordinates: ', type(DEM_GeoCoordinates))
    print('shape of DEM_GeoCoordinates: ', DEM_GeoCoordinates.shape)

    PIDs_Processing(raster_df, point_df, DEM_GeoCoordinates, PIDs_path)

    end = time.time()
    run_time = end - start
    print('Average time={}'.format(run_time))

