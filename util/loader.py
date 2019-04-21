import numpy as np
import glob
from PIL import Image
import os
directory_path = r'C:/Users/k-higuchi\Desktop\LAB2019\2019_04_10\Middle_Data\quarter_resolution\MiddEval3-data-Q\MiddEval3\trainingQ'


class Loader(object):
    def __init__(self, directory_path, init_size=(128, 128), one_hot=True):
        self._data = Loader.import_data(directory_path, init_size, one_hot)

        def load_train_test(self, train_rate=0.85, shuffle=True, transpose_by_color=False):
            """
            `Load datasets splited into training set and test set.
            訓練とテストに分けられたデータセットをロードします．
            Args:
                train_rate (float): Training rate.
                shuffle (bool): If true, shuffle dataset.
                transpose_by_color (bool): If True, transpose images for chainer. [channel][width][height]
            Returns:
                Training Set (Dataset), Test Set (Dataset)
            """
            if train_rate < 0.0 or train_rate > 1.0:
                raise ValueError("train_rate must be from 0.0 to 1.0.")
            if transpose_by_color:
                self._data.transpose_by_color()
            if shuffle:
                self._data.shuffle()

            train_size = int(self._data.images_left.shape[0] * train_rate)
            data_size = int(len(self._data.images_left))
            train_set = self._data.perm(0, train_size)
            test_set = self._data.perm(train_size, data_size)

            return train_set, test_set

    @staticmethod
    def import_data(directory_path, init_size=None, one_hot=True):
        # Generate paths of images to load
        # 読み込むファイルのパスリストを作成
        paths_right, paths_left = Loader.generate_paths(directory_path)

        # Extract images to ndarray using paths
        # 画像データをndarrayに展開
        images_left, images_right = Loader.extract_images(
            paths_right, paths_left, init_size, one_hot)

        # Get a color palette
        # カラーパレットを取得
        image_sample_palette = Image.open(paths_left[0])
        palette = image_sample_palette.getpalette()

        return DataSet(images_left, images_right, palette)

    @staticmethod
    def generate_paths(directory_path):
        left_path = r'\*\im0.png'
        right_path = r'\*\im1.png'

        paths_left = glob.glob(directory_path+left_path)
        paths_right = glob.glob(directory_path+right_path)

        if len(paths_right) == 0 or len(paths_left) == 0:
            raise FileNotFoundError("Could not load images.")

        return paths_right, paths_left

    @staticmethod
    def extract_images(paths_left, paths_right, init_size, one_hot):
        images_left, images_right = [], []

        # Load images from directory_path using generator
        print("Loading left images", end="", flush=True)
        for image in Loader.image_generator(paths_left, init_size, antialias=True):
            images_left.append(image)
            if len(images_left) % 20 == 0:
                print(".", end="", flush=True)
        print(" Completed", flush=True)

        print("Loading right images", end="", flush=True)
        for image in Loader.image_generator(paths_right, init_size, normalization=False):
            images_right.append(image)
            if len(images_right) % 20 == 0:
                print(".", end="", flush=True)
        print(" Completed")

        assert len(images_left) == len(images_right)

    @staticmethod
    def crop_to_square(image):
        size = min(image.size)
        left, upper = (image.width - size) // 2, (image.height - size) // 2
        right, bottom = (image.width + size) // 2, (image.height + size) // 2
        return image.crop((left, upper, right, bottom))

    @staticmethod
    def image_generator(file_paths, init_size=(300, 300), normalization=False, antialias=False):
        for file_path in file_paths:
            if file_path.endswith('.png') or file_path.endswith('.png'):
                image = Image.open(file_path)
                image = Loader.crop_to_square(image)
            if init_size is not None and init_size != image.size:
                image = image.resize(init_size)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image = np.asarray(image)
            if normalization:
                image = image / 255.0
            yield image


class DataSet(object):
    def __init__(self, images_left, images_right, image_palette, augmenter=None):
        assert len(images_left) == len(
            images_right), "images and labels must have same length."
        self._images_left = images_left
        self._images_right = images_right
        self._image_palette = image_palette
        self._augmenter = augmenter

    @property
    def images_left(self):
        return self._images_left

    @property
    def images_right(self):
        return self._images_right

    @property
    def palette(self):
        return self._image_palette

    @property
    def length(self):
        return len(self._images_left)

    def print_information(self):
        print("****** Dataset Information ******")
        print("[Number of Images]", len(self._images_left))

    def __add__(self, other):
        images_left = np.concatenate([self.images_left, other.images_left])
        images_right = np.concatenate(
            [self.images_right, other.images_right])
        return DataSet(images_left, images_right, self._image_palette, self._augmenter)

    def shuffle(self):
        idx = np.arange(self._images_left.shape[0])
        np.random.shuffle(idx)
        self._images_left, self._images_right = self._images_left[idx], self._images_right[idx]

    def transpose_by_color(self):
        self._images_left = self._images_left.transpose(0, 3, 1, 2)
        self._images_right = self._images_right.transpose(0, 3, 1, 2)

    def perm(self, start, end):
        end = min(end, len(self._images_left))
        return DataSet(self._images_left[start: end], self._images_right[start: end], self._image_palette, self._augmenter)

    def __call__(self, batch_size=20, shuffle=True, augment=True):
        """
        `A generator which yields a batch. The batch is shuffled as default.
        バッチを返すジェネレータです。 デフォルトでバッチはシャッフルされます。
        Args:
            batch_size (int): batch size.
            shuffle (bool): If True, randomize batch datas.
        Yields:
            batch (ndarray[][][]): A batch data.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be more than 1.")
        if shuffle:
            self.shuffle()

        for start in range(0, self.length, batch_size):
            batch = self.perm(start, start+batch_size)
        if augment:
            assert self._augmenter is not None, "you have to set an augmenter."
            yield self._augmenter.augment_dataset(batch, method=[ia.ImageAugmenter.NONE, ia.ImageAugmenter.FLIP])
        else:
            yield batch
