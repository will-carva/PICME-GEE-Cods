import cv2
import numpy as np
import sys
import time

i=0
j=0
x=0
print(" ")
print("ATENÇÃO: AS IMAGENS DEVEM CONTER O MESMO TAMANHO.")
print("EXEMPLO: 600x600 & 600x600")
print(" ")
print("Digite o nome e a extensão da primeira imagem: ")
print("Exemplo: imagem-1.jpg ")

read= input("Nome da imagem: ")
img = cv2.imread('images/'+read)
saida = cv2.imread('images/'+read) #seta tipo de váriavel (a imagem não é utilizada)

print(" ")
print("Digite o nome e a extensão da segunda imagem: ")
print("Exemplo: imagem-2.jpg ")
read2= input("Nome da imagem: ")
fundo = cv2.imread('images/'+read2)

altura=fundo.shape[0]
largura=fundo.shape[1]

v=float(input("Digite o valor de Intensidade: "))
print("""
				+--------------------------------------+
				|          Selecione o Método          |
				+--------------------------------------+
				+--------------------------------------+
				| [1] Média simples                    |
				+--------------------------------------+
				| [2] Soma das raizes quadráticas      |
				+--------------------------------------+
				| [3] Média quadrática                 |
				+--------------------------------------+
				| [4] Básico (preto e branco)          |
				+--------------------------------------+
					""")
opcao2 = int(input("Digite a opção escolhida: "))

for i in range(altura):
	for j in range(largura):
		red = float(img[i,j,2]) - float(fundo[i,j,2])
		green = float(img[i,j,1]) - float(fundo[i,j,1])
		blue = float(img[i,j,0]) - float(fundo[i,j,0])

		if (red < 0):
			red = 0
		if (green < 0):
			green = 0
		if (blue < 0):
			blue = 0


		if (opcao2 == 1):

			valor = float((red + green + blue) / 3)
			if (red > v or green >v or blue >v):
				saida[i,j,2] = 0
				saida[i,j,1] = 0
				saida[i,j,0] = 0
			else:
				saida[i,j,2] = valor
				saida[i,j,1] = valor
				saida[i,j,0] = valor


		elif (opcao2 ==2):

			valor = float((red**2 + green**2 + blue**2)**0.5)
			if (red > v or green >v or blue >v):
				saida[i,j,2] = 0
				saida[i,j,1] = 0
				saida[i,j,0] = 0
			else:
				saida[i,j,2] = valor
				saida[i,j,1] = valor
				saida[i,j,0] = valor

		elif (opcao2 ==3):
			valor = float((red**2 + green**2 + blue**2)/3)
			if (red > v or green >v or blue >v):
				saida[i,j,2] = 0
				saida[i,j,1] = 0
				saida[i,j,0] = 0
			else:
				saida[i,j,2] = valor
				saida[i,j,1] = valor
				saida[i,j,0] = valor

		elif (opcao2 ==4):
			if (red > v or green >v or blue >v):
				saida[i,j,2] = 0
				saida[i,j,1] = 0
				saida[i,j,0] = 0
			else:
				saida[i,j,2] = 255
				saida[i,j,1] = 255
				saida[i,j,0] = 255



resposta=input("Deseja visualizar a imagem agora? [Y] [N] : ")
if (resposta == "Y" or resposta=="y"):
	print("Digite qualquer tecla para fechar a visualização (não clique no X).")
	cv2.imshow("Imagem Sobreposta",saida)
	cv2.waitKey()
	cv2.destroyAllWindows()

resposta=input("Deseja salvar a imagem? [Y] [N] : ")
if (resposta == "Y" or resposta=="y"):
	nome=input("Digite o nome para a imagem a ser salva: (não necessita extensão) ")
	cv2.imwrite(nome+".jpg",saida)
	print("Imagem gerada e salva com sucesso!")
