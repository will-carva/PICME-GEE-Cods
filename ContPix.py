import cv2
import numpy

for i in range(0,16):
    img = cv2.imread("images/img%i.1.png"%i)
    print("As dimensões dessa imagem são: " + str(img.shape))
    altura, largura, cores = img.shape
    qntd = 0
    for y in range(0, altura):
            for x in range(0, largura):
                if all( i == 255 for i in img[y][x]):
                    qntd += 1
    preto = 656593 - qntd
    print("A quantidade de pixeis pretos é " + str(preto))
