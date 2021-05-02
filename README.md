<h1> Interactive Digit Recogniser (96% accuracy) </h1>

This project lets users input a digit between 0 and 9 into a [Tkinter](https://docs.python.org/3/library/tkinter.html) canvas.
A trained neural network predicts the identity of the digit with associated probabilities.

Notes on training the neural network:
<ul>
  <li> I used the <a href="http://yann.lecun.com/exdb/mnist/">MNIST</a> dataset to train a <a href="https://www.tensorflow.org/guide/keras/sequential_model">Keras</a> neural network.  </li>
  <li> It was found that the <a href="http://yann.lecun.com/exdb/mnist/">MNIST</a> images were too regular, so did not generalise well to interactively drawn digits. To counter this, I used image augmentation to enhance my dataset: use of scaling, rotations and transformations. </li>
  <li> Normalising the inputs in range [0,1] gave stronger neural network performance. </li>
</ul>

To run:

<ol> 
  <li> Open neural_network.py and run. This saves the neural network. </li>
  <li> Open input_grid.py and press run. A GUI will appear with an input canvas. </li>
</ol>  

Enjoy!

![Interactive Digit Recogniser](https://github.com/RobertFielding/Interactive-Digit-Recogniser/blob/master/Interactive_Digit_Recogniser.gif)
