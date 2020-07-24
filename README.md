# CNN_visualization
Implementation to visualize the internal representation of images inside convolutional neural networks as part of my bachelors thesis. The goal was a framework that could handle all steps from dataset (raw images) to interpretable visualisations: pre-processiong, training and visualization.
I worked with Keras with TensorFlow backend using available visualization solutions for that part and combined everything to a working test-environment.
In the current state it's a working prototype that might be improved or extended in the future.

Some examples of what the output looks like. Note that there are always two outputs, one for each used vis-library
keras-vis (https://github.com/raghakot/keras-vis) and
iNNvestigate (https://github.com/albermax/innvestigate)
the output for innvestigate is always the same as only the class with the highest score is shown but vis is different depending on how many classes we are dealing with as this library can visualize results per class.

artificial data from the thesis:

![Alt text](images/gray_vis.png?raw=true "example of artificial data, vis library")
![Alt text](images/gray_innvestigate.png?raw=true "example of artificial data, innvestigate library")

also from the thesis but relevant areas have been annotated:

![Alt text](images/ano_vis.png?raw=true "example of annotated artificial data, vis library")
![Alt text](images/ano_innvestigate.png?raw=true "example of annotated artificial data, innvestigate library")

again with a different annotation-technique:

![Alt text](images/shift_vis.png?raw=true "example of shift-annotated artificial data, vis library")
![Alt text](images/shift_innvestigate.png?raw=true "example of shift-annotated artificial data, innvestigate library")

an example with a dataset of cats and dogs to prove it working with color images as well:

![Alt text](images/cat_vis.png?raw=true "example of cat+dog dataset, vis library")
![Alt text](images/cat_innvestigate.png?raw=true "example of cat+dog dataset, innvestigate library")

and an example of it tackling the mnist dataset of handwritten digits to see the behaviour of more than two classes:

![Alt text](images/1_vis.png?raw=true "example of mnist dataset, vis library")
![Alt text](images/1_innvestigate.png?raw=true "example of mnist dataset, innvestigate library")
