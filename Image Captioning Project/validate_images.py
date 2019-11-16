import os
from PIL import Image
def validate_images(path='./Images/'):
	for filename in os.listdir(path):
		if filename.endswith('.jpg'):
			try:
				img = Image.open(path + filename)  # open the image file
				img.verify()  # verify that it is, in fact an image
			except (IOError, SyntaxError) as e:
				os.remove(path + filename)
				print('Bad file:', filename, '\n...Removed!')
