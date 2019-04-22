import glob
import os
import re
from PIL import Image

def path_changes(picture_name,path_dict,init_size=(1024,1024)):
    for dic_name, path_end in path_dict.items():
        #画像の種類ごとにpathを取得する(左目disp->右目disp->...)
        paths = glob.glob(original+path_end)
        for path, pic_name in zip(paths,picture_name):
            image = Image.open(path)
            
            if init_size:
                image = image.resize(init_size)
            #親ディレクトリ/画像種類別ディレクトリ/画像名.png
            image.save(modify+r'\\'+dic_name+r'\\'+pic_name+'.png')


if __name__ == '__main__':

    original = r'C:\Users\k-higuchi\Desktop\LAB2019\2019_04_10\0419_middlebury_FCNN\dataset\2006_stereo\Original'
    modify = r'C:\Users\k-higuchi\Desktop\LAB2019\2019_04_10\0419_middlebury_FCNN\dataset\2006_stereo\Modify'
    
    #ディレクトリ名が画像名になっているので、それを抽出(ex:baby1等)
    picture_paths = glob.glob(original+r'\*')
    picture_name = [re.sub(r'.+\\', '', x) for x in picture_paths]

    path_dict = {
        'Left_disparity': r'\*\disp1.png',
        'Left_RGB': r'\*\view1.png',
        'Right_disparity': r'\*\disp5.png',
        'Right_RGB': r'\*\view5.png'}
    
    path_changes(picture_name, path_dict)
