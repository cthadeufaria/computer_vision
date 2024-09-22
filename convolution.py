import os
import requests
from scipy import signal as sg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

conv_types = ['full', 'same', 'valid']

g = [[1, 0, -1],
     [1, 0, -1],
     [1, 0, -1]]
I = [[1, 2, 0, 4, 5],
     [5, 6, 0, 8, 6],
     [4, 4, 0, 4, 7],
     [1, 2, 0, 4, 8]]

# Cross correlated image
for type in conv_types:
    print('With zero padding ({0}):'.format(type))
    cross_correlated = sg.convolve( I, g, type)
    print('{0} \n'.format(cross_correlated))

# Convoluted image
for type in conv_types:
    print('Without zero padding ({0}):'.format(type))
    convoluted = sg.convolve( I, g, type)
    print('{0} \n'.format(convoluted))

urls = {'./images/bird.jpg': 'https://ibm.box.com/shared/static/cn7yt7z10j8rx6um1v9seagpgmzzxnlz.jpg',
        './images/number.jpg': 'https://ibm.box.com/shared/static/vvm1b63uvuxq88vbw9znpwu5ol380mco.jpg'}

for path, url in urls.items():
    if not os.path.isfile(path):
        response = requests.get(url)
        with open(path, 'wb') as file:
            file.write(response.content)

for img in urls.keys():
    im = Image.open(img)  # type here your image's name
    image_gr = im.convert("L")   # convert("L") translate color images into black and white
    print("\n Original type: %r \n\n" % image_gr)
    arr = np.asarray(image_gr)
    print("After conversion to numerical representation: \n\n %r" % arr)

    ### Plot image
    imgplot = plt.imshow(arr)
    imgplot.set_cmap('gray')  #you can experiment different colormaps (Grey,winter,autumn)
    print("\n Input image converted to gray scale: \n")
    plt.show()

    # Edge detection:
    kernel = np.array([[ 0, 1, 0],
                    [ 1,-4, 1],
                    [ 0, 1, 0],])
    grad = sg.convolve2d(arr, kernel, mode='same', boundary='symm')
    print('GRADIENT MAGNITUDE - Feature map')
    fig, aux = plt.subplots(figsize=(10, 10))
    aux.imshow(np.absolute(grad), cmap='gray')
    plt.show()

    grad_biases = np.absolute(grad) + 100
    grad_biases[grad_biases > 255] = 255

    print('GRADIENT MAGNITUDE - Feature map')
    fig, aux = plt.subplots(figsize=(10, 10))
    aux.imshow(np.absolute(grad_biases), cmap='gray')
    plt.show()

