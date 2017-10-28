import math 
import numpy as np
from numpy import *
from PIL import Image

def rangeCompression(Img, c):
	imgArr = np.array(Img)
	(H, W) = imgArr.shape
	IrcArr = np.zeros((H, W))
	for i in range(H):
		for j in range(W):
			IrcArr[i, j] = c * math.log10(1 + imgArr[i, j])
	return IrcArr

def main(file):
	I = Image.open(file).convert('L')

	for i in range(4):
		c = 10**i
		img = rangeCompression(I, c)
		Image.fromarray(img).show()
		fileName = "../SavingImage/2_bw_clipping_c=" + str(10**i) + ".jpg"
		Image.fromarray(img).convert("RGB").save(fileName,"PNG")


main("../ImageData/2_bw.jpg")