import ee # importação
import PIL
import requests
from PIL import Image
from io import BytesIO
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import io
# importação do sklearn
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
ee.Initialize() # inicialização

# Função para aplicar à imagem vinda da coleção a máscara de água
def mascara_agua(imagem):
    qa = imagem.select('pixel_qa')
    return qa.bitwiseAnd(1 << 2).eq(0)

# Função para aplicar à imagem vinda da coléção a máscara de nuvem/sombra de nuvem
def mascara_nuvem(imagem):
    qa = imagem.select('pixel_qa')
    return qa.bitwiseAnd(1 << 3).eq(0) and (qa.bitwiseAnd(1 << 5).eq(0)) and (qa.bitwiseAnd(1 << 6).eq(0)) and (qa.bitwiseAnd(1 << 7).eq(0))

# função para aplicar as máscaras
def aplicar_mascaras(imagem):

    # criar uma imagem em branco/vazio para evitar problemas no fundo ao gerar um PNG
    # usamos valores dummies (neste caso, branco)
    vazio = ee.Image(99999)

    # máscara de água
    agua = vazio.updateMask(mascara_agua(imagem).Not()).rename('agua')

    # máscara de nuvem (criará uma imagem com apenas nuvens)
    # caso a imagem não tenha nuvens, ela ficará toda branca
    nuvem = vazio.updateMask(mascara_nuvem(imagem).Not()).rename('nuvem')

    # podemos ainda, ao contrário da linha anterior, REMOVER as nuvens
    # notem que retiramos a função .Not (negação)
    sem_nuvem = vazio.updateMask(mascara_nuvem(imagem)).rename('sem_nuvem')

    # aplicar o indice NDVI
    ndvi = imagem.expression('(nir - red) / (nir + red)',{'nir':imagem.select('B5'),'red':imagem.select('B4')}).rename('ndvi')

    # assim como fizemos para o NDVI, retornamos uma imagem com as novas bandas
    return imagem.addBands([ndvi,agua,nuvem,sem_nuvem])

# função para aplicar uma máscara em uma banda específica
# A mascará a ser aplicada
def aplicar_mascara_banda(imagem, banda_mascara, banda_origem, band_destino):

    # Primeiramente, temos que aplicar a máscara desejada na banda de origem, que será nomeada para a banda de destino
    # Podemos, inclusive, sobscrever a banda de origem, sem problemas
    imagem_mascara = imagem.select(banda_origem).updateMask(imagem.select(banda_mascara)).rename(band_destino)

    # Depois, temos que criar uma imagem em branco que receberá a máscara, renomeando também para banda de destino
    imagem_mascara = ee.Image(99999).blend(imagem_mascara).rename(band_destino)

    # Retornar a imagem com a nova banda nomeada com a string da banda_destino
    return imagem.addBands([imagem_mascara])

def aplicar_mascara_banda_modificada(imagem, imagem_mascara, banda_origem, band_destino):

    # Primeiramente, temos que aplicar a máscara desejada na banda de origem, que será nomeada para a banda de destino
    # Podemos, inclusive, sobscrever a banda de origem, sem problemas
    imagem_mascara = imagem.select(banda_origem).updateMask(imagem_mascara).rename(band_destino)

    # Depois, temos que criar uma imagem em branco que receberá a máscara, renomeando também para banda de destino
    imagem_mascara = ee.Image(99999).blend(imagem_mascara).rename(band_destino)

    # Retornar a imagem com a nova banda nomeada com a string da banda_destino
    return imagem.addBands([imagem_mascara])

    # Inicialmente, devemos extrair as coordenadas por pixel da imagem
    # O GEE faz essa operação adicoinando uma banda com essas novas informações,
    # que extrairemos abaixo
    imagem = imagem.addBands(ee.Image.pixelLonLat())

    # Extraindo efetivament as coordenadas nas bandas recém criadas (latitude e longitude)
    # Nesta parte, é utilizado o que citamos como reducer (verifiar na documentação do GEE), mas ele permite que sejam feitas operações com uma imagem como: reduzi-la, modificar sua escala, etc.
    # ainda, os atributos utilizados são: geometry (geometria, mesma utilizada em outros exemplos), scale (escala do sensor, 30 metros no caso do Landsat), bestEffort (garante que a imagem terá a melhor escala possível, caso a definida seja muito grande para processamento)
    coordenadas = imagem.select(['longitude', 'latitude']+bandas).reduceRegion(reducer=ee.Reducer.toList(),geometry=geometria,scale=30,bestEffort=True)

    # ponteiro para incluir os valores dos pixeis de cada banda, já criando uma Numpy Array
    # o FOR abaixo irá percorrer cada banda que foi definida no parâmetro da função para extrair seus valores, um a um
    # As funções ee.List e getInfo() permitem transformar os pixeis em lista e depois extraí-los, respectivamente
    bandas_valores = []
    for banda in bandas:

        # adiciona pixel por pixel, em cada uma das bandas desejadas
        # transforma o valor do pixel em float para evitar erros de processamento futuros
        bandas_valores.append(np.array(ee.List(coordenadas.get(banda)).getInfo()).astype(float))


    # Retorna no forma de Numpy Array os dados separados pelas colunas [0,1,2..N BANDAS] sendo LATITUDE, LONGITUDE e VALOR DO PIXEL (POR BANDA...N BANDAS)
    return np.array(ee.List(coordenadas.get('latitude')).getInfo()).astype(float), np.array(ee.List(coordenadas.get('longitude')).getInfo()).astype(float), bandas_valores

# Notem que foi criada uma coordenada (Latitude e Longitude) através de uma string, posteriormente repartida pelas virgulas
# Essa abordagem é importante para quando utilizarmos a linha da comando -46.3555863,-23.166012,-46.3555863,-23.166012
coordenadas = "-44.730543,-22.464540,-44.470186,-22.647335"

# Aqui, usamos uma ferramenta do Python chamada de unpacking
x1,y1,x2,y2 = coordenadas.split(",")

# Criamos a geometria com base nas coordenadas 'quebradas' acima
geometria = geometry = ee.Geometry.Polygon(
        [[[float(x1),float(y2)],
          [float(x2),float(y2)],
          [float(x2),float(y1)],
          [float(x1),float(y1)],
          [float(x1),float(y2)]]])

# Podemos, também, extrair as coordenadas centrais da área de estudo
latitude_central = (float(x1)+float(x2))/2
longitude_central = (float(y1)+float(y2))/2
# string de datas (poderia vir, por exemplo, da linha de comando como um argumento)
datas = "2015-01-19,2015-01-20"
datas1 = "2015-04-09,2015-04-10"
datas2 = "2015-06-28,2015-06-29"
datas3 = "2015-07-14,2015-07-15"
datas4 = "2015-08-15,2015-08-16"
datas5 = "2015-09-16,2015-09-17"
datas6 = "2015-12-05,2015-12-06"
datas7 = "2016-01-06,2016-01-07"
datas8 = "2016-04-11,2016-04-12"
datas9 = "2016-06-14,2016-06-15"
datas10 = "2016-08-17,2016-08-18"
datas11 = "2016-09-18,2016-09-19"
datas12 = "2016-12-07,2016-12-08"
datas13 = "2017-01-08,2017-01-09"
datas14 = "2017-02-09,2017-02-10"
datas15 = "2017-03-29,2017-03-30"
datas16 = "2017-06-01,2017-06-02"


# Divisão das duas datas pela vírgula, novamente usando a técnica de unpacking
inicio,fim = datas.split(",")
inicio1,fim1 = datas1.split(",")
inicio2,fim2 = datas2.split(",")
inicio3,fim3 = datas3.split(",")
inicio4,fim4 = datas4.split(",")
inicio5,fim5 = datas5.split(",")
inicio6,fim6 = datas6.split(",")
inicio7,fim7 = datas7.split(",")
inicio8,fim8 = datas8.split(",")
inicio9,fim9 = datas9.split(",")
inicio10,fim10 = datas10.split(",")
inicio11,fim11 = datas11.split(",")
inicio12,fim12 = datas12.split(",")
inicio13,fim13 = datas13.split(",")
inicio14,fim14 = datas14.split(",")
inicio15,fim15 = datas15.split(",")
inicio16,fim16 = datas16.split(",")


# Consultando a coleção com base na área de estudo e datas selecionadas
# Imagem para extração do corpo d'água
# Aqui, para gerar uma série temporal bem extensa, extendemos o filtro da data
colecao_mascara_agua = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate('2018-01-01','2020-01-01').filterMetadata('CLOUD_COVER','less_than', 30)
colecao_mascara_agua = colecao_mascara_agua.map(aplicar_mascaras)
imagem_mascara_agua = colecao_mascara_agua.median()
# Consultando a coleção com base na área de estudo e datas selecioandas
# Notem que utilizamos o filtro 'CLOUD_COVER'. Vários datasets possuem esses 'metadados' que podem ser utilizados para, neste caso, pegar as imagens com menos nuvem possível
colecao = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio,fim).filterMetadata('CLOUD_COVER','not_less_than', 1)

colecao1 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio1,fim1).filterMetadata('CLOUD_COVER','less_than', 30)
colecao2 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio2,fim2).filterMetadata('CLOUD_COVER','less_than', 10 )
colecao3 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio3,fim3).filterMetadata('CLOUD_COVER','less_than', 1 )
colecao4 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio4,fim4).filterMetadata('CLOUD_COVER','not_less_than', 7 )
colecao5 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio5,fim5).filterMetadata('CLOUD_COVER','not_less_than', 5)
colecao6 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio6,fim6).filterMetadata('CLOUD_COVER','less_than', 40).filterMetadata('CLOUD_COVER','not_less_than', 30)
colecao7 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio7,fim7).filterMetadata('CLOUD_COVER','not_less_than', 10).filterMetadata('CLOUD_COVER','less_than', 20)
colecao8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio8,fim8).filterMetadata('CLOUD_COVER','not_less_than', 8).filterMetadata('CLOUD_COVER','less_than', 10)
colecao9 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio9,fim9).filterMetadata('CLOUD_COVER','not_less_than', 15).filterMetadata('CLOUD_COVER','less_than', 20)
colecao10 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio10,fim10).filterMetadata('CLOUD_COVER','less_than', 2)
colecao11 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio11,fim11).filterMetadata('CLOUD_COVER','not_less_than', 10).filterMetadata('CLOUD_COVER','less_than', 20)
colecao12 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio12,fim12).filterMetadata('CLOUD_COVER','not_less_than', 1).filterMetadata('CLOUD_COVER','less_than', 2)
colecao13 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio13,fim13).filterMetadata('CLOUD_COVER','less_than', 30)
colecao14 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio14,fim14).filterMetadata('CLOUD_COVER','less_than', 30)
colecao15 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio15,fim15).filterMetadata('CLOUD_COVER','not_less_than', 15).filterMetadata('CLOUD_COVER','less_than', 30)
colecao16 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio16,fim16).filterMetadata('CLOUD_COVER','less_than', 10).filterMetadata('CLOUD_COVER','not_less_than', 9)

print(str(type(colecao1)))

# Mostrar o total de imagens encontradas
print("Total de imagens encontradas: "+str(colecao.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao1.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao2.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao3.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao4.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao5.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao6.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao7.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao8.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao9.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao10.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao11.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao12.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao13.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao14.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao15.size().getInfo()))
print("Total de imagens encontradas: "+str(colecao16.size().getInfo()))

colecao = colecao.map(aplicar_mascaras)

colecao1 = colecao1.map(aplicar_mascaras)
colecao2 = colecao2.map(aplicar_mascaras)
colecao3 = colecao3.map(aplicar_mascaras)
colecao4 = colecao4.map(aplicar_mascaras)
colecao5 = colecao5.map(aplicar_mascaras)
colecao6 = colecao6.map(aplicar_mascaras)
colecao7 = colecao7.map(aplicar_mascaras)
colecao8 = colecao8.map(aplicar_mascaras)
colecao9 = colecao9.map(aplicar_mascaras)
colecao10 = colecao10.map(aplicar_mascaras)
colecao11 = colecao11.map(aplicar_mascaras)
colecao12 = colecao12.map(aplicar_mascaras)
colecao13 = colecao13.map(aplicar_mascaras)
colecao14 = colecao14.map(aplicar_mascaras)
colecao15 = colecao15.map(aplicar_mascaras)
colecao16 = colecao16.map(aplicar_mascaras)

imagem = colecao.median()

imagem1 = colecao1.median()
imagem2 = colecao2.median()
imagem3 = colecao3.median()
imagem4 = colecao4.median()
imagem5 = colecao5.median()
imagem6 = colecao6.median()
imagem7 = colecao7.median()
imagem8 = colecao8.median()
imagem9 = colecao9.median()
imagem10 = colecao10.median()
imagem11 = colecao11.median()
imagem12 = colecao12.median()
imagem13 = colecao13.median()
imagem14 = colecao14.median()
imagem15 = colecao15.median()
imagem16 = colecao16.median()

imagem = aplicar_mascara_banda(imagem, 'agua', 'ndvi', 'ndvi_agua')
imagem = aplicar_mascara_banda(imagem, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem = aplicar_mascara_banda(imagem, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem = aplicar_mascara_banda(imagem, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem1 = aplicar_mascara_banda(imagem1, 'agua', 'ndvi', 'ndvi_agua')
imagem1 = aplicar_mascara_banda(imagem1, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem1 = aplicar_mascara_banda(imagem1, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem1 = aplicar_mascara_banda(imagem1, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem2 = aplicar_mascara_banda(imagem2, 'agua', 'ndvi', 'ndvi_agua')
imagem2 = aplicar_mascara_banda(imagem2, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem2 = aplicar_mascara_banda(imagem2, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem2 = aplicar_mascara_banda(imagem2, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem3 = aplicar_mascara_banda(imagem3, 'agua', 'ndvi', 'ndvi_agua')
imagem3 = aplicar_mascara_banda(imagem3, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem3 = aplicar_mascara_banda(imagem3, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem3 = aplicar_mascara_banda(imagem3, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem4 = aplicar_mascara_banda(imagem4, 'agua', 'ndvi', 'ndvi_agua')
imagem4 = aplicar_mascara_banda(imagem4, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem4 = aplicar_mascara_banda(imagem4, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem4 = aplicar_mascara_banda(imagem4, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem5 = aplicar_mascara_banda(imagem5, 'agua', 'ndvi', 'ndvi_agua')
imagem5 = aplicar_mascara_banda(imagem5, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem5 = aplicar_mascara_banda(imagem5, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem5 = aplicar_mascara_banda(imagem5, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem6 = aplicar_mascara_banda(imagem6, 'agua', 'ndvi', 'ndvi_agua')
imagem6 = aplicar_mascara_banda(imagem6, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem6 = aplicar_mascara_banda(imagem6, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem6 = aplicar_mascara_banda(imagem6, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem7 = aplicar_mascara_banda(imagem7, 'agua', 'ndvi', 'ndvi_agua')
imagem7 = aplicar_mascara_banda(imagem7, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem7 = aplicar_mascara_banda(imagem7, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem7 = aplicar_mascara_banda(imagem7, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem8 = aplicar_mascara_banda(imagem8, 'agua', 'ndvi', 'ndvi_agua')
imagem8 = aplicar_mascara_banda(imagem8, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem8 = aplicar_mascara_banda(imagem8, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem8 = aplicar_mascara_banda(imagem8, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem9 = aplicar_mascara_banda(imagem9, 'agua', 'ndvi', 'ndvi_agua')
imagem9 = aplicar_mascara_banda(imagem9, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem9 = aplicar_mascara_banda(imagem9, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem9 = aplicar_mascara_banda(imagem9, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem10 = aplicar_mascara_banda(imagem10, 'agua', 'ndvi', 'ndvi_agua')
imagem10 = aplicar_mascara_banda(imagem10, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem10 = aplicar_mascara_banda(imagem10, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem10 = aplicar_mascara_banda(imagem10, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem11 = aplicar_mascara_banda(imagem11, 'agua', 'ndvi', 'ndvi_agua')
imagem11 = aplicar_mascara_banda(imagem11, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem11 = aplicar_mascara_banda(imagem11, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem11 = aplicar_mascara_banda(imagem11, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem12 = aplicar_mascara_banda(imagem12, 'agua', 'ndvi', 'ndvi_agua')
imagem12 = aplicar_mascara_banda(imagem12, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem12 = aplicar_mascara_banda(imagem12, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem12 = aplicar_mascara_banda(imagem12, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem13 = aplicar_mascara_banda(imagem13, 'agua', 'ndvi', 'ndvi_agua')
imagem13 = aplicar_mascara_banda(imagem13, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem13 = aplicar_mascara_banda(imagem13, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem13 = aplicar_mascara_banda(imagem13, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem14 = aplicar_mascara_banda(imagem14, 'agua', 'ndvi', 'ndvi_agua')
imagem14 = aplicar_mascara_banda(imagem14, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem14 = aplicar_mascara_banda(imagem14, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem14 = aplicar_mascara_banda(imagem14, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem15 = aplicar_mascara_banda(imagem15, 'agua', 'ndvi', 'ndvi_agua')
imagem15 = aplicar_mascara_banda(imagem15, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem15 = aplicar_mascara_banda(imagem15, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem15 = aplicar_mascara_banda(imagem15, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

imagem16 = aplicar_mascara_banda(imagem16, 'agua', 'ndvi', 'ndvi_agua')
imagem16 = aplicar_mascara_banda(imagem16, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem16 = aplicar_mascara_banda(imagem16, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem16 = aplicar_mascara_banda(imagem16, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')
"""
mascara_agua = ee.Image(0).blend(ee.Image(99999).updateMask(imagem_mascara_agua.select('agua').gt(0))).eq(99999)
imagem = aplicar_mascara_banda_modificada(imagem, mascara_agua, 'ndvi', 'ndvi_agua')

imagem1 = aplicar_mascara_banda_modificada(imagem1, mascara_agua, 'ndvi', 'ndvi_agua')
imagem2 = aplicar_mascara_banda_modificada(imagem2, mascara_agua, 'ndvi', 'ndvi_agua')
imagem3 = aplicar_mascara_banda_modificada(imagem3, mascara_agua, 'ndvi', 'ndvi_agua')
imagem4 = aplicar_mascara_banda_modificada(imagem4, mascara_agua, 'ndvi', 'ndvi_agua')
imagem5 = aplicar_mascara_banda_modificada(imagem5, mascara_agua, 'ndvi', 'ndvi_agua')
imagem6 = aplicar_mascara_banda_modificada(imagem6, mascara_agua, 'ndvi', 'ndvi_agua')
imagem7 = aplicar_mascara_banda_modificada(imagem7, mascara_agua, 'ndvi', 'ndvi_agua')
imagem8 = aplicar_mascara_banda_modificada(imagem8, mascara_agua, 'ndvi', 'ndvi_agua')
imagem9 = aplicar_mascara_banda_modificada(imagem9, mascara_agua, 'ndvi', 'ndvi_agua')
imagem10 = aplicar_mascara_banda_modificada(imagem10, mascara_agua, 'ndvi', 'ndvi_agua')
imagem11 = aplicar_mascara_banda_modificada(imagem11, mascara_agua, 'ndvi', 'ndvi_agua')
imagem12 = aplicar_mascara_banda_modificada(imagem12, mascara_agua, 'ndvi', 'ndvi_agua')
imagem13 = aplicar_mascara_banda_modificada(imagem13, mascara_agua, 'ndvi', 'ndvi_agua')
imagem14 = aplicar_mascara_banda_modificada(imagem14, mascara_agua, 'ndvi', 'ndvi_agua')
imagem15 = aplicar_mascara_banda_modificada(imagem15, mascara_agua, 'ndvi', 'ndvi_agua')
imagem16 = aplicar_mascara_banda_modificada(imagem16, mascara_agua, 'ndvi', 'ndvi_agua')
"""
# Depois, cortamos a imagem
# scale = escala do sensor. No caso do Landsat-8/OLI são 30 metros

imagem_corte = imagem.clipToBoundsAndScale(geometry=geometria,scale=30)

imagem_corte1 = imagem1.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte2 = imagem2.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte3 = imagem3.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte4 = imagem4.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte5 = imagem5.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte6 = imagem6.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte7 = imagem7.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte8 = imagem8.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte9 = imagem9.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte10 = imagem10.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte11 = imagem11.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte12 = imagem12.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte13 = imagem13.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte14 = imagem14.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte15 = imagem15.clipToBoundsAndScale(geometry=geometria,scale=30)
imagem_corte16 = imagem16.clipToBoundsAndScale(geometry=geometria,scale=30)

imagem_pillow = PIL.Image.open(BytesIO(requests.get(imagem_corte.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow.save('images/img0.png')

imagem_pillow1 = PIL.Image.open(BytesIO(requests.get(imagem_corte1.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow1.save('images/img1.png')

imagem_pillow2 = PIL.Image.open(BytesIO(requests.get(imagem_corte2.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow2.save('images/img2.png')

imagem_pillow3 = PIL.Image.open(BytesIO(requests.get(imagem_corte3.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow3.save('images/img3.png')

imagem_pillow4 = PIL.Image.open(BytesIO(requests.get(imagem_corte4.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow4.save('images/img4.png')

imagem_pillow5 = PIL.Image.open(BytesIO(requests.get(imagem_corte5.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow5.save('images/img5.png')

imagem_pillow6 = PIL.Image.open(BytesIO(requests.get(imagem_corte6.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow6.save('images/img6.png')

imagem_pillow7 = PIL.Image.open(BytesIO(requests.get(imagem_corte7.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow7.save('images/img7.png')

imagem_pillow8 = PIL.Image.open(BytesIO(requests.get(imagem_corte8.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow8.save('images/img8.png')

imagem_pillow9 = PIL.Image.open(BytesIO(requests.get(imagem_corte9.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow9.save('images/img9.png')

imagem_pillow10 = PIL.Image.open(BytesIO(requests.get(imagem_corte10.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow10.save('images/img10.png')

imagem_pillow11 = PIL.Image.open(BytesIO(requests.get(imagem_corte11.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow11.save('images/img11.png')

imagem_pillow12 = PIL.Image.open(BytesIO(requests.get(imagem_corte12.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow12.save('images/img12.png')

imagem_pillow13 = PIL.Image.open(BytesIO(requests.get(imagem_corte13.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow13.save('images/img13.png')

imagem_pillow14 = PIL.Image.open(BytesIO(requests.get(imagem_corte14.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow14.save('images/img14.png')

imagem_pillow15 = PIL.Image.open(BytesIO(requests.get(imagem_corte15.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow15.save('images/img15.png')

imagem_pillow16 = PIL.Image.open(BytesIO(requests.get(imagem_corte16.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow16.save('images/img16.png')



imagem_pillow = PIL.Image.open(BytesIO(requests.get(imagem_corte.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow.save('images/img0.1.png')

imagem_pillow1 = PIL.Image.open(BytesIO(requests.get(imagem_corte1.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow1.save('images/img1.1.png')

imagem_pillow2 = PIL.Image.open(BytesIO(requests.get(imagem_corte2.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow2.save('images/img2.1.png')

imagem_pillow3 = PIL.Image.open(BytesIO(requests.get(imagem_corte3.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow3.save('images/img3.1.png')

imagem_pillow4 = PIL.Image.open(BytesIO(requests.get(imagem_corte4.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow4.save('images/img4.1.png')

imagem_pillow5 = PIL.Image.open(BytesIO(requests.get(imagem_corte5.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow5.save('images/img5.1.png')

imagem_pillow6 = PIL.Image.open(BytesIO(requests.get(imagem_corte6.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow6.save('images/img6.1.png')

imagem_pillow7 = PIL.Image.open(BytesIO(requests.get(imagem_corte7.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow7.save('images/img7.1.png')

imagem_pillow8 = PIL.Image.open(BytesIO(requests.get(imagem_corte8.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow8.save('images/img8.1.png')

imagem_pillow9 = PIL.Image.open(BytesIO(requests.get(imagem_corte9.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow9.save('images/img9.1.png')

imagem_pillow10 = PIL.Image.open(BytesIO(requests.get(imagem_corte10.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow10.save('images/img10.1.png')

imagem_pillow11 = PIL.Image.open(BytesIO(requests.get(imagem_corte11.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow11.save('images/img11.1.png')

imagem_pillow12 = PIL.Image.open(BytesIO(requests.get(imagem_corte12.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow12.save('images/img12.1.png')

imagem_pillow13 = PIL.Image.open(BytesIO(requests.get(imagem_corte13.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow13.save('images/img13.1.png')

imagem_pillow14 = PIL.Image.open(BytesIO(requests.get(imagem_corte14.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow14.save('images/img14.1.png')

imagem_pillow15 = PIL.Image.open(BytesIO(requests.get(imagem_corte15.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow15.save('images/img15.1.png')

imagem_pillow16 = PIL.Image.open(BytesIO(requests.get(imagem_corte16.select(['ndvi_agua_sem_nuvem']).getThumbUrl({'min':0, 'max':3000})).content))
imagem_pillow16.save('images/img16.1.png')
