# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[gray_image]: ./examples/gray_image.jpg "gray_image"
[blur_image]: ./examples/blur_image.jpg "blur_image"
[edges]: ./examples/edges.jpg "edges"
[masked_edges]: ./examples/masked_edges.jpg "masked_edges"
[line_img]: ./examples/line_img.jpg "line_img"
[lane_lines]: ./examples/lane_lines.jpg "lane_lines"
[roi]: ./examples/roi.jpg "roi"
[rio_with_dash]: ./examples/rio_with_dash.jpg "rio_with_dash."
[hough_line_on_origin]: ./examples/hough_line_on_origin.jpg "hough_line_on_origin."
[hough_lines]: ./examples/hough_lines.jpg "hough_lines."
[hough_line_slope]: ./examples/hough_line_slope.jpg "hough_line_slope."
[polyfit]: ./examples/polyfit.jpg "polyfit." 

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

### 1.1 My pipeline consisted of 5 steps. 

1. tansfrom color image to gray scale image then using gaussian blur to smooth the gray image
 ![][blur_image]
 
1. using canny to detect edges
![][edges]
1. set 4 vertices of our interest region and using this region as a mask to rule out the non-related lines.
![][rio_with_dash]
3. apply hough transform to detect the line segment 
![][hough_lines]
4. Draw hough lines on the original image.
![][hough_line_on_origin]

### 1.2 In order to draw a single line on the left and right lanes, I defined another function to draw the final lane lines called draw_lane_lines() function :

1. For each line in the set of hough line segments, calculate the slope and only keep the line segments whoes absolute value of slope is between 0.4~1(marked as blue and green) and drop those slope outside the threshold(marked as red) 
([figure 2-1]())
2. Divide those lines into two groups, the left group contains the line segments whoes slope is below zero (blue line) and the right group contains the line segments whoes slope is above zero (green line)
([figure 2-1]())
3. Apply linear regression to all the points in each group to get two lines(np.polyfit returns the slope and intercept of each line)
([figure 2-2]())
4. the farest points in the two lines is considering at the top of rio and the closest point is at the bottom of rio
5. Draw the lane lines by using the slope,intercept,bottomY and topY

 
![figure 2-1][hough_line_slope] 

![figure 2-2][polyfit] 

![figure 2-3][lane_lines] 


### 2. Identify potential shortcomings with your current pipeline

1. the algorithm that I use will fail when there are curved lines inside the region of interest .
2. the parameter of canny and hough transform may not be the best and it can not adjust depending on the different situations
3. the correctness of the result can be influenced by the light condition
4. when I apply the pipeline to the video , the lane lines has jittering


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to figure out how to detect the curved line
Another potential improvement could be to try to eliminate  influence of the light condition
I also want to try to apply low-pass filter to the lane lines to eliminate the jittering

