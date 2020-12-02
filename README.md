# ConvolutionalAutoEncoder
This is an alternative architecure for convolutional autoencoder for image classification. It was part of a bachelor thesis.
The goal of the bachelor thesis was to create and validate an architecure which is more efficient than classical solutions and which can be run on a Raspberry Pi Zero.
Classical solutions need to do the training on external more powerful hardware. They then transfer the trained weights to the plattform which doesn't have the ressources to do the training in a fair amount of time. This architecure is meant to be run completely on a Raspberry Pi Zero (Training & Classification).
To test it's performance it was trained with the MNIST Dataset. No Library like tensorflow or keras was used.

### Test with MNIST

first convolutional layer:  
<img src="https://i.imgur.com/yzW8UqT.png" height="200" > 

second convolutional layer:  
<img src="https://i.imgur.com/kjdb4N6.png" height="200" > 

confusion matrix:  
<img src="https://i.imgur.com/VSOsEce.png" height="400" > 

accuracy: 86%


Please keep in mind, this is a prototype!
