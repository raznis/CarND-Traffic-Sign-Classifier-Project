#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram.png "Histogram"
[image2]: ./examples/grayscale.png "Grayscaled Image"
[image3]: ./examples/rgb.png "Original Image"
[image4]: ./examples/1x.png "Traffic Sign 1"
[image5]: ./examples/2x.png "Traffic Sign 2"
[image6]: ./examples/3x.png "Traffic Sign 3"
[image7]: ./examples/8x.png "Traffic Sign 4"
[image8]: ./examples/9x.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/raznis/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a histogram of depicting the number of training examples for each class of traffic signal. We can see there is a large variance in the dataset. Some classes have a very small amount of examples.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth and fifth code cell of the notebook.

As a preprocessing step, I decided to convert the images to grayscale because it makes it easier for the neural network to detect edges.

Here is an example of a traffic sign image before and after grayscaling.
![alt text][image3]
![alt text][image2]

After trying data normalization, I decided not to do it in the final version, since it seemed to have hurt prediction accuracy.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the sixth code cell of the notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by randomly allocating 10 percent of the training set to validation.

My final training set had 35288 number of images. My validation set and test set had 3921 and 12630 number of images.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x9 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x18 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x18 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 8x8x36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x36					|
| Flatten			    | output 576   									|
| Fully connected		| output 120        							|
| RELU					|												|
| Dropout				| Keep ratio 0.6								|
| Fully connected		| output 84	        							|
| RELU					|												|
| Dropout				| Keep ratio 0.6								|
| Fully connected		| output 43	        							|
| Softmax				| output is probabilities						|

 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. I ran a standard training with cross validation procedure. 

To train the model, I used an ADAM optimizer with batch size 256, running 75 Epochs with a learning rate of 0.001.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eighth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.995
* test set accuracy of 0.964

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first approach was vanilla LeNet with 3 input channels.
* What were some problems with the initial architecture?
The LeNet architecture did not do well on rgb input. Validation accuracy was below 92%.
* The architecture was adjusted by adding another 5x5 convolution layer with max pooling, and by adding dropout with keep ratio 0.6 to the first two fully connected layers. This was done for regularization purposes.
* Which parameters were tuned? How were they adjusted and why?
I experimented with several options for the dropout ratio ranging from 0.5 to 0.9. Empirically 0.6 showed the best results. The number of epoches was set relatively high at 75, since I did not notice validation accuracy dropping (probably due to regularization via dropout).
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Convolution layers detect aspects of the images hierarchically, so adding more layers gives the model more freedom to detects different levels of abstraction. Dropout prevents the model from overfitting by forcing it not to rely on any single neuron activation.


 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images are similar to what was found in the dataset, but each has a different background, and some are on an angle, i.e., not facing directly the camera.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way		 	| Right-of-way   								| 
| 30 km/h     			| 30 km/h 										|
| Priority Road			| Priority Road									|
| General caution  		| General caution				 				|
| Road work				| Road work		      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.5%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is absolutely sure that this is a right of way sign (probability of ~1.0), and the image does contain a right of way sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999        			| Right-of-way   								| 
| 1.0     				| 30 km/h 										|
| 1.0					| Priority Road									|
| 1.0	      			| General caution					 			|
| 0.999				    | Road work		      							|


Same goes for the other images. The model has very high predictions for the correct classes.