# -*- coding: latin-1 -*- 
#AA - script para ler e visualizar imagens
import matplotlib.pyplot as plt
fName="lena.tif"   #necessário estar na mesma dir. que código
I=plt.imread(fName)#ler imagens (I-> numpy array uint8)
plt.subplot(1,2,1) #1x2 figuras (3º valor = índice da figura 1 ou 2)
plt.imshow(I) #origem no canto inferior esquerdo
#tirar eixos - plt.axis('off') tb dá
plt.xticks([]),plt.yticks([]),plt.box(True) 
#atenção que pixeis estão em  uint8 
#só é possível representar valores entre 0-255
plt.subplot(1,2,2)
plt.imshow(I*2)
plt.xticks([]),plt.yticks([]),plt.box(True)
#guarda figura 
plt.savefig('../figs/L0AAex005.png', transparent=True, bbox_inches='tight', pad_inches=0)
plt.show() #não é necessário em     #guardar  em ficheiro ".png"
           #iPython usar show()     #(na directoria "../figs/")
                                    #(dá erro se não existir)
