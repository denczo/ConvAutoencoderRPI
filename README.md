# Alternative architecture for Convolutional Autoencoder
This is the prototype of an alternative architecure for convolutional autoencoder for image classification. It was part of a bachelor thesis.
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


| Training set  | Test set| Training duration  | Accuracy |
| ------------- |:-------:|:------------------:|:--------:|
| 60k           | 10k     | 177.97 s           | 85.12 %  |
| 40k           | 10k     | 130.96 s           | 86.26 %  |
| 20k           | 10k     | 61.49 s            | 85.81 %  |
| 10k           | 10k     | 30.22 s            | 84.23 %  |
| 5k            | 10k     | 15.4 s             | 83.8 %   |       
| 1k            | 10k     | 3.95 s             | 80.39 %  |
| 500           | 10k     | 2.3 s              | 74.77 %  |

_Platform: Core i5 2500k, 12GB DDR3 RAM, Windows 10, 128GB SSD_  
_Training duration includes training of the convolutional layers (0.75 s)_


| Training set  | Test set| Training duration  | Accuracy |
| ------------- |:-------:|:------------------:|:--------:|
| 1k            | 10k     | 178.64 s           | 80.6 %  |
| 500           | 10k     | 104.25 s           | 74.79 %  |

_Platform: Rasperry Zero_   
_Training duration includes training of the convolutional layers (31.61 s)_
