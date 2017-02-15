

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image8]: ./output_images/non_car_HOG_example.png
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.  

I started by reading in all the `vehicle` and `non-vehicle` images.  These example images come from a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself
I am using this data to train my SVM classifier for the project

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`


![alt text][image2]
![alt_text][image8]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters in colorspace, hog_chanel, spatial_size, etc and find out color space RGB has the least linear SVM accuracy.  I tried 'YCrcb' and 'HSV' colorspace and they have better SVM accuracy than RGB.
Also, use all hog_channel increasse the prediction accuracy.  

I settle with the following choice of HOG parameters

colorspace = 'YCrCb' 
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" 
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

Using these parameters, I was able to achieve ~99% accuracy using SVM classifier.

```59.52 Seconds to extract HOG features...
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6108
15.38 Seconds to train SVC...
Test Accuracy of SVC =  0.9896
My SVC predicts:  [ 1.  0.  1.  1.  1.  1.  0.  0.  1.  0.]
For these 10 labels:  [ 1.  0.  1.  1.  1.  1.  0.  0.  1.  0.]
0.00094 Seconds to predict 10 labels with SVC
####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
```

After the selection of the feature parameters, the feature vector are generated and scaled, and used to train the SVM classifier. We have 8792 car images and 8968 non car images, each is 64x64. I then split the data into randomzied training and test sets of 80/20. 
   
    
###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Since HOG is a computationally expensive operation, I have tried to optimize the sliding window operation. First, we run the classifier only on the lower half of the image where the road lies.  By doing such, I reduce false positive and reduce the time it takes for the pipeline to process the frame.  Next, I extract sections of region of interest and resize those sections to 64x64 so that the feature vector generated will be the same as those used to train the calssifer.  After sacling the generated feature vectors, they are passed to my SVM classifer to determine if there is a vehilce in this imag segment.  I used a sliding window approach different overlap percentages and window size.  The small window size took a lot of time to process and bigger window size didnot capture far vehicle too well.  So I in the end, I chose a single 96X96 window size and overlap of 50 percent.  I tried this sliding window mechansim on the test images shown below:

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?


Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

