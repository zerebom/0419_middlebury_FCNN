import glob
directory_path = r'C:/Users/k-higuchi\Desktop\LAB2019\2019_04_10\Middle_Data\quarter_resolution\MiddEval3-data-Q\MiddEval3\trainingQ'


def generate_paths(directory_path):
    left_path = r'\*\im0.png'
    right_path = r'\*\im1.png'

    left_paths = glob.glob(directory_path+left_path)
    right_paths = glob.glob(directory_path+right_path)

    return left_paths, right_paths


def crop_to_square(image):
    size = min(image.size)
    left, upper = (
        image.width - size) // 2, (image.height - size) // 2
    right, bottom = (
        image.width + size) // 2, (image.height + size) // 2
    return image.crop((left, upper, right, bottom))


def image_generatar(file_paths, init_size=(300, 300), normalization=False):
    for file_path in file_paths:
        if file_path.endswith('.png') or file_path.endswith('.png'):
            image = Image.open(file_path)
            image = crop_to_square(image)
        if init_size is not None and init_size != image.size:
            image = image.resize(init_size)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = np.asarray(image)
        if normalization:
            image = image / 255.0
        yield image


output_size = (300, 300)
left_paths, right_paths = generate_paths(directory_path)
for i in image_generatar(left_paths, output_size, normalization=False):
    print(i.shape)
