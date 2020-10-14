# cnn-colab
Convolutional nets for colab 
<br>

Hey there! :cat:

* the "nn-images.zip" uncompresses into a "nn-images" directory that contains 100 images ("base-images")

* the "VGG-self-contained.py" containts a VGG-simil network that trains itself over 100k images generated from the base-images and returns a small-report. 

* there's more info about the directories' structure and the output in the script's index.

<br><br><br>
More info:<br>
<b> what we are trying to do here is to train a network for learning the rotation angle AND the scaling factor</b><br>
<i>(yes, both at the same time. Simone's paper splitted the task between two networks, so we are aiming for a benchmark.)</i>
