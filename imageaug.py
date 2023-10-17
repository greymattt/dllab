# Algorithm:
# 1. Import necessary packages.
# 2. Define the image data generator with augmentation settings.
# 3. Load and preprocess a sample image.
# 4. Then add batch dimension.
# 5. Generate augmented images.


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('nike.png')
img = np.array(img)

HEIGHT = img.shape[0]
WIDTH = img.shape[1]


# Flip Image
def flip(img):
	img = np.fliplr(img)
	plt.imshow(img)
	plt.show()

# img[:, :-pixel] - right
# img[:, pixel:] - left

# Left Shift
def left(img, pixel):
    img[:, :-pixel] = img[:, pixel:]
    img[:, WIDTH-pixel:] = 0
    plt.imshow(img)
    plt.show()

# Right Shift
def right(img, pixel):
    img[:, pixel:] = img[:, :-pixel]
    img[:, :pixel] = 0
    plt.imshow(img)
    plt.show()


# img[:-pixel] - down
# img[pixel:] - up


# Up Shift
def up(img, pixel):
    img[:-pixel] = img[pixel:]
    img[HEIGHT-pixel:] = 0
    plt.imshow(img)
    plt.show()

# Down Shift
def down(img, pixel):
    img[pixel:] = img[:-pixel]
    plt.imshow(img)
    plt.show()

# Adding Noise
def noiseimg(img, level=5):
    noise = np.random.randint(level, size=img.shape, dtype='uint8')
    noisy_img = np.clip(img + noise, 0, 255)
    plt.imshow(noisy_img)
    plt.show()


left(img, 100)