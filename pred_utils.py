from utils.test import TTAFrame
from utils.networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from glob import glob
import os
import numpy as np
import cv2
import torch
from osgeo import osr,ogr,gdal
os.environ['PROJ_LIB'] = r'D:\QGIS\apps\Python39\Lib\site-packages\pyproj\proj_dir\share\proj'


def TwoPercentLinear(image, max_out=255, min_out=0):
    '''
    image:需要拉伸的影像
    max_out:输出影像的最大像素值，默认为255
    min_out:输出影像的最小像素值，默认为0
    '''
    h, w, _ = image.shape
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[img1==0]=[100,100,100]
    b, g, r = cv2.split(image)

    def gray_process(gray, maxout = max_out, minout = min_out):
        high_value = np.percentile(gray, 98)
        low_value = np.percentile(gray, 2)
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
        processed_gray = ((truncated_gray - low_value)/(high_value - low_value)) * (maxout - minout)
        return processed_gray

    r_p = gray_process(r)
    g_p = gray_process(g)
    b_p = gray_process(b)
    result = cv2.merge((b_p, g_p, r_p))
    result[img1==0]=[0,0,0]
    return np.uint8(result)


def cut_image(image,out_path):
    '''
    image:需要裁剪的影像
    out_path:裁剪后的影像保存路径
    '''
    m=0
    h1,w1,_=image.shape
    x=h1//1024
    y=w1//1024
    img=np.zeros(((x+1)*1024,(y+1)*1024,3))
    img[0:h1,0:w1]=image
    for i in range(x+1):
        for j in range(y+1):
            new_imge=img[i*1024:(i+1)*1024,j*1024:(j+1)*1024,:]
            out_dir=out_path +f'/{m}.png'
            cv2.imwrite(out_dir ,new_imge )
            m+=1


def merge_image(img_path,out_path,x,y,h,w):
    '''
    img_path:需要拼接的影像所在的文件夹
    out_path:拼接完的影像的输出路径
    x:完整影像的高度整除512得到的数字
    y：完整影像的宽度整除512得到的数字
    h:完整影像的高度
    w:完整影像的宽度
    '''
    m=0
    img=np.zeros(((x+1)*1024,(y+1)*1024,3))
    for i in range(x+1):
        for j in range(y+1):
            image1=cv2.imread(img_path +f'/{m}.png')
            img[i*1024:(i+1)*1024,j*1024:(j+1)*1024,:]=image1
            m+=1
    img1=img[0:h,0:w,:]
    img1=img1.astype(np.int16)
    cv2.imwrite(out_path ,img1)

def predict(image_path,output_path,pth_path):
    clip_path=output_path +'/clip'
    pred_path=output_path +'/pred'

    #创建中间保存文件夹
    if os.path.exists (clip_path ) is False :
        os.makedirs(clip_path )
    if os.path.exists(pred_path ) is False:
        os.makedirs(pred_path )

    #模型加载
    solver = TTAFrame(DinkNet34)
    print(pth_path )
    solver.load(pth_path)

    image_paths=glob(image_path +'\*.tif')

    for dir in image_paths :
        print(dir )
        out_dir=os.path.join(output_path ,os.path.basename(dir ) )
        image=cv2.imread(dir )
        image=TwoPercentLinear(image)
        cut_image(image,clip_path )

        img_paths=glob(clip_path +'/*.png')
        h, w, _ = image.shape
        x = h //1024
        y = w //1024

        #预测裁剪后影像并存储到临时文件夹中
        for path in img_paths :
            basename=os.path.basename(path )
            out_path=os.path.join(pred_path ,basename )
            mask=solver.test_one_img_from_path(path)
            mask[mask > 4.0] = 255
            mask[mask <= 4.0] = 0
            mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
            cv2.imwrite(out_path  , mask.astype(np.uint8))

        #合并预测影像
        merge_image(pred_path ,out_dir ,x,y,h,w)

        #为影像添加地理参考
        sorce=gdal.Open(dir,gdal.GA_ReadOnly  )
        sr=osr.SpatialReference()
        geo_transform=sorce.GetGeoTransform()
        sr.ImportFromWkt(sorce.GetProjectionRef())
        target=gdal.Open(out_dir ,gdal.GARIO_UPDATE )
        target.SetProjection(sr.ExportToWkt() )
        target.SetGeoTransform(geo_transform)

        sorce =None
        target =None

    #删除临时文件夹
    cut_out = clip_path.replace("/", "\\")
    pred_out = pred_path.replace('/', "\\")

    cmd1 = f'rd/s/q {cut_out}'
    cmd2 = f'rd/s/q {pred_out}'
    os.system(cmd1)
    os.system(cmd2)
