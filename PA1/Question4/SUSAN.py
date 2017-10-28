import math 
import numpy as np
from numpy import *
from sympy import E
from PIL import Image
from PIL import ImageDraw

def saveImg(Img, fileName):
    Img.convert("RGB").save(fileName,"PNG")

def drawCircleOnImg(Img, pointArr, threshold):
    print("Drawing Circle")
    Img = Img.convert('RGB')
    draw = ImageDraw.Draw(Img)
    count = 0
    for i in range(pointArr.shape[0]):
        for j in range(pointArr.shape[1]):
            if pointArr[i, j] > threshold:
                draw.ellipse( (j - 2, i - 2, j + 2, i + 2), outline ='red')
                count = count + 1
    print( "Total circle drawing : " + str(count) )
    Img.show()
    saveImg(Img, "../SavingImage/test.png")

def NMS(R):
    (H, W) = R.shape
    nmsR = np.zeros(R.shape)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            partR = R[i - 1: i + 2, j - 1: j + 2]
            if argmax(partR) == 4:
                nmsR[i, j] = R[i, j]
    return nmsR

def susan(imgArr ,mask, t):
    (H, W) = imgArr.shape
    sumCArr = np.zeros(imgArr.shape)
    R = np.zeros(imgArr.shape)
    for i in range(3, H - 3):
        for j in range(3, W - 3):
            partImg = imgArr[i - 3 : i + 4, j - 3 : j + 4]
            centerPixVal = imgArr[i, j]
            cArr = exp( -1 * ( ( partImg - centerPixVal ) / t )**6 )
            cMulMask = multiply(cArr, mask)
            sumCArr[i, j] = sum(cMulMask) - 1
    threshold = sumCArr[ (int)(argmax(sumCArr) / W), argmax(sumCArr) % H ] * 0.5
    low = where(sumCArr <= threshold)
    R[low] = threshold - sumCArr[low]
    return R
"""
def imageNormalization(imgArr, newMax, newMin):
    imgArrX, imgArrY = imgArr.shape
    oldMax = imgArr[ (int)(argmax(imgArr) / imgArrX), argmax(imgArr)%imgArrY ]
    oldMin = imgArr[ (int)(argmin(imgArr) / imgArrX), argmin(imgArr)%imgArrY ]
    print( (oldMax, oldMin) )
"""
def medianFilter(imgArr, filterSize):
    imgArrX, imgArrY = imgArr.shape
    newImgArr = np.zeros(imgArr.shape)
    for x in range((int)(filterSize/2), imgArrX - (int)(filterSize/2)):
        for y in range((int)(filterSize/2), imgArrY - (int)(filterSize/2)):
            partImgArr = imgArr[x - (int)(filterSize/2): x + (int)(filterSize/2) + 1, 
                                y - (int)(filterSize/2): y + (int)(filterSize/2) + 1]
            oneDArr = np.zeros(filterSize**2)
            for i in range(filterSize):
                for j in range(filterSize):
                    oneDArr[i*filterSize+j] = partImgArr[i, j]
            oneDArr.sort()
            newImgArr[x, y] = oneDArr[(int)(filterSize**2 / 2)]
    return newImgArr

def convolve(imgArr, mask):
    maskSize = mask.shape[0]
    imgArrX, imgArrY = imgArr.shape
    newImgArr = np.zeros(imgArr.shape)
    for x in range((int)(maskSize/2), imgArrX - (int)(maskSize/2)):
        for y in range((int)(maskSize/2), imgArrY - (int)(maskSize/2)):
            partImgArr = imgArr[x - (int)(maskSize/2): x + (int)(maskSize/2) + 1, 
                                y - (int)(maskSize/2): y + (int)(maskSize/2) + 1]
            newImgArr[x, y] = sum(np.multiply(partImgArr, mask))
    return newImgArr

def gaussian(sigma, kSize):
    (centerX, centerY)= ( (kSize - 1) / 2, (kSize - 1) / 2)
    s2 = 2.0 * (sigma**2)
    g = np.zeros( (kSize, kSize) )
    for y in range(kSize):
        for x in range(kSize):
            x2 = (x - centerX)**2
            y2 = (y - centerY)**2
            g[x, y] = 1 / ( 2 * math.pi * sigma**2 ) * exp( -(x2 + y2) / s2 )
    return g / sum(g)

def main(file, G, threshold):
    I = Image.open(file).convert('L')
    imgArr = np.array(I)

    mask = np.array([
                [0,0,1,1,1,0,0],
                [0,1,1,1,1,1,0],
                [1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1],
                [0,1,1,1,1,1,0],
                [0,0,1,1,1,0,0]])
    
    #Imedi = medianFilter(imgArr, 3)
    #imageNormalization(Imedi, 255, 0)
    Iblur = convolve(imgArr, G)
    #imageNormalization(Iblur, 255, 0)
    Image.fromarray(Iblur).show()
    R = susan(Iblur, mask, 20)
    nmsR = NMS(R)
    drawCircleOnImg(I, nmsR, threshold)
    

sigma = 0.2
maskSize = 11
G = gaussian(sigma, maskSize)
main("../ImageData/susan_input2.png", G, 0)
#main("../ImageData/susan_input2.png", G, 10)
