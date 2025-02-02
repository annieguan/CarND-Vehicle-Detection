

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run ypipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image8]: ./output_images/non_car_HOG_example.png
[image3]: ./output_images/sliding_window.png
[image4]: ./output_images/heatmap_sliding_window.png
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
Also, use all hog_channels increasse the prediction accuracy.  

I settle with the following choice of HOG parameters

```
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
```

Using these parameters, I was able to achieve ~99% accuracy using SVM classifier.

```
59.52 Seconds to extract HOG features...
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6108
15.38 Seconds to train SVC...
Test Accuracy of SVC =  0.9896
My SVC predicts:  [ 1.  0.  1.  1.  1.  1.  0.  0.  1.  0.]
For these 10 labels:  [ 1.  0.  1.  1.  1.  1.  0.  0.  1.  0.]
0.00094 Seconds to predict 10 labels with SVC
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After the selection of the feature parameters, the feature vector are generated and scaled, and used to train the SVM classifier. We have 8792 car images and 8968 non car images, each is 64x64. I then split the data into randomzied training and test sets of 80/20. 
   
    
###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Since HOG is a computationally expensive operation, I have tried to optimize the sliding window operation. First, we run the classifier only on the lower half of the image where the road lies.  By doing such, I reduce false positive and reduce the time it takes for the pipeline to process the frame.  Next, I extract sections of region of interest and resize those sections to 64x64 so that the feature vector generated will be the same as those used to train the calssifer.  After scaling the generated feature vectors, they are passed to my SVM classifer to determine if there is a vehilce in this imag segment.  I used a sliding window approach different overlap percentages and window size.  The small window size took a lot of time to process and bigger window size didnot capture far vehicle too well.  

So I in the end, I chose a single 96X96 window size and overlap of 50 percent.  I tried this sliding window mechansim on the test images shown below:


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?


Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector,  a single 96X96 window size and overlap of 50 percent, which provided a nice result.  Here are some example images:

![alt text][image3]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap.  I then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

In order to remove the false positive, I integrate a heatmap over 15 frames of video, , such that areas of multiple detections get "hot", while transient false positives stay "cool". I then simply threshold the integrated heatmap to remove false positives.

Here's an example result showing the heatmap from all 6 test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes on the images.

### Here are six frames and their corresponding heatmaps:

![alt text][image4]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
  

1.  To speed things up, instead of extract HOG features from each individual windows as I searched across the image, I extract HOG features just once for the entire region of interest (just the lower half of each frame) and subsample that array for each sliding window.

2. The challenging part of this project for me is to remove the false positive and in the meantime, identify the right number of vehicle.  I integrate a heatmap over a series of frames. I tried the various combination for the buffer size to store the heatmap for a series of frames of video  as well as the threshold which makes sense.

3. My pipleline won't identity the vehicle as soon as it enters the scene as I integrate a heat map over 15 frames of video.  Also for the fast moving vehicle,  heat wont be get built at the same pixel and would be harder to detect. 

4. Things that I could make it more robust.  If there are overlap detection window or different scale, I can assign the position of the detection to the centroid of the overlapping windows.   THe other thing is to use multi-scale windows, use bigger window size for vehicle closer to camera, use the smaller window size for vehicle near horizon.
 
