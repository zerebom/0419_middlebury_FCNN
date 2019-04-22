# 1024^2の画像を分割するうえにディレクトリを分けて、通し番号を付けるコード
from PIL import Image
import os
import sys
import glob
from datetime import datetime


def ImgSplit(im):
    # 読み込んだ画像を200*200のサイズで54枚に分割する
    height = 256
    width = 256

    # 縦の分割枚数
    for h1 in range(4):
        # 横の分割枚数
        for w1 in range(4):
            w2 = w1 * height
            h2 = h1 * width
            yield im.crop((w2, h2, width + w2, height + h2))


if __name__ == '__main__':
    # 画像の読み込み
    modify = r'C:\Users\k-higuchi\Desktop\LAB2019\2019_04_10\0419_middlebury_FCNN\dataset\2006_stereo\Modify'
    dir_names = ['Left_disparity', 'Left_RGB', 'Right_disparity', 'Right_RGB']

    # 写真の種類ごと
    for dir_name in dir_names:
        global_count = 0
        picture_paths = glob.glob(modify + r'\\' + dir_name+r'\\*.png')
        print(picture_paths)
        splits_dir = modify + r'\\' + dir_name + r'\\splits\\'
        try:
            os.mkdir(splits_dir)
        except:
            pass

        # 写真ごと
        for pic_path in picture_paths:
            local_count = 0
            print(pic_path)

            pic_file = os.path.basename(pic_path)
            pic = Image.open(pic_path)
            root, ext = os.path.splitext(pic_file)

            # 分割ごと
            for ig in ImgSplit(pic):
                local_count += 1
                global_count += 1
                # 保存先フォルダの指定
                ig.save(splits_dir+str(global_count) +
                        "_"+root+"_"+str(local_count)+ext)
