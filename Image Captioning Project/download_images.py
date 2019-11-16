import pandas as pd


# %%
import os
from PIL import Image
import requests
from io import BytesIO
import natsort
import timeout_decorator
timeout_decorator.timeout(45, use_signals=False)
from validate_images import validate_images


def resize_image(image, size):
	"""Resize an image to the given size."""
	return image.resize(size, Image.ANTIALIAS)

def get_max_image():
	path = 'Images/'
	files = natsort.natsorted(os.listdir(path))
	res = [int(i) for i in files[-1] if i.isdigit()]
	s = ''
	for i in res:
		s += str(i)
	return int(s)


def download_and_resize_images(output_dir, size, n=100):
	df = pd.read_csv('remaining_images.tsv', sep='\t')
	df.columns = ['Caption', 'URL']
	df2 = pd.read_csv('downloaded_images.csv')
	max_image = get_max_image()
	"""Resize the images in 'image_dir' and save into 'output_dir'."""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	num_images = len(df.URL)
	filenames = [f'Image_{a+max_image}.jpg' for a in range(1,n+1)]
	i = 0
	counter = 0
	for url in df.URL:
		i += 1
		if i == n:
			break
		image = f'Image_{i+max_image}.jpg'
		try:
			response = requests.get(url, timeout=8)
			img = Image.open(BytesIO(response.content))
			img = resize_image(img, size)
			img.save(os.path.join(output_dir, image))
			if (i + 1) % 50 == 0:
				print("[{}/{}] Resized the images and saved into '{}'."
					  .format(i + 1, num_images, output_dir))
		except (TimeoutError, OSError) as e:
			print(f'an {e} exception occured')
			counter += 1
			continue
	df4 = df.copy()
	df4['filename'] = filenames + ['a.jpg' for _ in range(len(df4) - len(filenames))]
	df['filename'] = filenames + ['a.jpg' for _ in range(len(df) - len(filenames))]
	df4 = df4[df4['filename'].isin(os.listdir('./Images'))]
	df2 = pd.concat([df2, df4])
	df = df[~df['filename'].isin(os.listdir('./Images'))]
	df = df.loc[counter:]
	df = df.drop('filename', axis=1)
	df.to_csv('remaining_images_test.csv', index=False)
	df2.to_csv('downloaded_images_test.csv', index=False)
	validate_images()

def download_images(num_batches, batch_size=500):
	for _ in range(num_batches):
		download_and_resize_images('./Images/', (256, 256), batch_size)

download_images(6000, 500)

