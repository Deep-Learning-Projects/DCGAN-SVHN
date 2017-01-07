import numpy as np
import math
from scipy import io
from PIL import Image

def _load_images():
	train_data = io.loadmat('data/train_32x32.mat')
	images = train_data['X']
	images = np.transpose(images, (3,2,0,1))
	images = images / 127.5 - 1.0
	return images

# num_images = 73200
num_images = 200
images = _load_images()[:num_images]
batch_size = 100
num_batches = num_images / batch_size

def get_batch(i):
	start_index = i * batch_size
	end_index = (i+1) * batch_size
	batch = images[start_index:end_index]
	return batch

def combine_and_save(generated_images, save_path):
	num = generated_images.shape[0]
	width = int(math.sqrt(num))
	height = int(math.ceil(float(num)/width))
	shape = generated_images.shape[2:]
	image = np.zeros((height*shape[0], width*shape[1], 3), dtype=generated_images.dtype)
	for index, img in enumerate(generated_images):
		i = int(index/width)
		j = index % width
		image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = np.transpose(img, (1,2,0))
	image = (image + 1) * 127.5
	Image.fromarray(image.astype(np.uint8), mode='RGB').save(save_path)



