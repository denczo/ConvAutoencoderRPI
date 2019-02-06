from ConvAutoencoder.ConvLayer import ConvLayer
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

img=mpimg.imread('images/XWing32x32.png')
img=np.append(img[:,:,:1].flatten(),np.append(img[:,:,1:2],img[:,:,2:3]))
fig = plt.figure(figsize=(10, 10))

#filterSize, filterAmount, stride, learnRate
CAE = ConvLayer(9, 9, 1, 0.000001)
CAE.setInput(img,3)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

for i in range(100):
    if i%25 == 0:
        print(i)
        fig.clear()
        for i in range(CAE.filterAmount):
            learned = CAE.featureMaps.T
            learned = learned[i*900:(i+1)*900]
            learned = learned.reshape(30, 30)
            fig.add_subplot(1, 9, i + 1)
            plt.axis('off')
            plt.imshow(learned, interpolation='None')

        #input = CAE.input[i*CAE.inputSize:(i+CAE.channels)*CAE.inputSize]

        for i in range(CAE.channels):
            fig.add_subplot(3, 3, i+1)
            real = CAE.input[i*CAE.inputSize:(i+1)*CAE.inputSize]
            real = real.reshape(32, 32)
            plt.axis('off')
            plt.imshow(real, interpolation='None')

        for i in range(CAE.channels):
            fig.add_subplot(3, 3, i+7)
            real = CAE.reconstr[i*CAE.inputSize:(i+1)*CAE.inputSize]
            real = real.reshape(32, 32)
            plt.axis('off')
            plt.imshow(real, interpolation='None')
        plt.draw()
        plt.pause(0.0001)

    CAE.forwardActivation()
    CAE.backwardsActivation()
    CAE.contrastiveDivergence()
plt.show()
