# neuralstyle

This is an implemenation of the paper *A Neural Algorithm of Artistic Style* by L. Gatsy, A. Ecker and M. Bethge ([http://arxiv.org/abs/1508.06576](http://arxiv.org/abs/1508.06576)).

For this program to work, you must [download](https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl) the weights for the VGG-19 deep net, and place the file in the models directory.

Here I've taken the content of this image:
![alt text](https://raw.githubusercontent.com/lhannest/neuralstyle/master/images/big_photo.jpg)

And I've used the style of this image:
![alt text](https://raw.githubusercontent.com/lhannest/neuralstyle/master/images/big_art.jpg)

To generate this image:
![alt text](https://raw.githubusercontent.com/lhannest/neuralstyle/master/images/results/result2.png)
