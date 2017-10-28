import math 
import numpy as np
from numpy import *
from PIL import Image

def clipping(Img, alpha, B, beta):
	imgArr = np.array(Img)
	(H, W) = imgArr.shape
	for i in range(H):
		for j in range(W):
			if 0 <= imgArr[i, j] < alpha:
				imgArr[i, j] = 0
			elif alpha <= imgArr[i, j] < B:
				imgArr[i, j] = beta *(imgArr[i, j] - alpha)
			else:
				imgArr[i, j] = beta *(B - alpha)
	return imgArr
def main(file):
	I = Image.open(file)
	alpha = 50
	B = 150
	beta = 2
	img = clipping(I, alpha, B, beta)
	Image.fromarray(img).show()
	Image.fromarray(img).convert("RGB").save("../SavingImage/2_bw_clipping.jpg","PNG")

main("../ImageData/2_bw.jpg")