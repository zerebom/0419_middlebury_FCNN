import tensorflow as tf
import glob

DIRECTORY_PATH = r'C:\Users\k-higuchi\Desktop\LAB2019\2019_04_10\0419_middlebury_FCNN\dataset\2006_stereo\Modify'
PATH_ENDS=r'\splits\*.png'
TRAIN_DIRS = [r'\Left_RGB', r'\Left_disparity', r'\Right_disparity']
TEST_DIR=r'\Right_RGB'

# def load_data(train_rate, directory_path):

R_rgb_list = glob.glob(DIRECTORY_PATH + TEST_DIR+PATH_ENDS)

L_rgb_list = glob.glob(DIRECTORY_PATH + TRAIN_DIRS[0]+PATH_ENDS)
L_disp_list = glob.glob(DIRECTORY_PATH + TRAIN_DIRS[1]+PATH_ENDS)
R_disp_list = glob.glob(DIRECTORY_PATH + TRAIN_DIRS[2]+PATH_ENDS)


with tf.python_io.TFRecordWriter('input.tfrecord') as w:
    for im0, im1, im2 in zip(L_rgb_list, L_disp_list, R_disp_list):
        print(im0,im1,im2)

        #ファイルをバイナリとして読み込み
        with tf.gfile.FastGFile(im0, 'rb') as f0, \
             tf.gfile.FastGFile(im1, 'rb') as f1, \
             tf.gfile.FastGFile(im2, 'rb') as f2:
            
            data0 = f0.read()
            data1 = f1.read()
            data2 = f2.read()
            
            #取得したbyte列をkey,valueに登録
            features = tf.train.Features(feature={
                'Left_RGB': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data0])),
                'Left_disparity': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data1])),
                'Right_disparity': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data2]))
            })

            #Exampleクラスにkey, valueを登録して書き込み
            example = tf.train.Example(features=features)
            w.write(example.SerializeToString())







with tf.python_io.TFRecordWriter('teacher.tfrecord') as w:

    for img in R_rgb_list:

        #ファイルをバイナリとして読み込み
        with tf.gfile.FastGFile(img, 'rb') as f:
            data = f.read()

        #取得したbyte列をkey,valueに登録
        features = tf.train.Features(feature={
            'Right_RGB': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))
        })

        #Exampleクラスにkey, valueを登録して書き込み
        example = tf.train.Example(features=features)
        w.write(example.SerializeToString())

# def image2TFrecord_from_glob(glob_list,TFR_name,data_name):
    
