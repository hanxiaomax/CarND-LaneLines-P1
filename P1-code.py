
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[1]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read in an Image

# In[2]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)

plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')

print("origin image shape",image.shape)
print("gray image shape",gray.shape)


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[3]:


import math
import os
from collections import deque

images = os.listdir("test_images/")

fig_size = 30
COL = 2
        
# 转换为灰阶图像
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
   
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# canny 边缘检测
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

# 图像平滑：高斯模糊
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# 目标区域选取
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image,mask



def draw_lines(img, lines, color=[0, 255, 0], thickness=4):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    #fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if abs(slope) < 0.4 or abs(slope)>1:
                continue
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    draw_lane_lines(line_img, lines)
    return line_img,lines


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def draw_imgs(imgs,vertices=None):
    """
    show images as subplots
    """
    #print("==",imgs.keys())
    fig = plt.figure(figsize=(fig_size,fig_size),dpi=100)
    for index,img in enumerate(imgs):
        plt.subplot(len(imgs)/COL+1, COL, index+1)
        img_name = list(img.keys())[0]
        image = list(img.values())[0]
        plt.title(img_name)
        plt.imshow(image)
        print(img_name)
    #fig.savefig("./examples/test_image_after.jpg",bbox_inches='tight')

        
def save_image(filename,img,size=(288,162)):
    """
    resize and save image to local
    """
    img2save = cv2.resize(img,size,cv2.INTER_LINEAR)
    if len(size)>2:
        plt.imsave("./examples/{0}.jpg".format(filename),img2save)
    else:
        plt.imsave("./examples/{0}.jpg".format(filename),img2save,cmap="gray")

def draw_poly(filename,img,vertices):
    """
    draw the poly lines of roi and save it to local
    """
    fig = plt.figure(figsize=(3,2),dpi=100)
    x = []
    y = []
    for v in vertices:
        for xx,yy in v:
            x.append(xx)
            y.append(yy)
            plt.plot(xx,yy,'.', color='red',markersize=10)
            #note_point = "({0},{1})".format(xx,yy)
            #plt.annotate(note_point,xy=(xx,yy),color='red')
            
    x.append(x[0])
    y.append(y[0])
    #plt.xticks([])  
    #plt.yticks([])  
    #plt.axis('off')
    plt.plot(x, y, 'b--', lw=4)
    plt.imshow(img)
    
    fig.savefig("./examples/{0}_with_dash.jpg".format(filename),bbox_inches='tight')
    


def draw_lines_in_xyspace(filename,lines,size=(28,16)):
    """
    draw a plot to show all the line segments 
    red line : lines with illegal slope
    blue line : lines with negative slope which is considering as the left line
    green line : lines with postive slope which is considering as the right line
    """
    fig = plt.figure(figsize=size,dpi=100)
    ax = plt.gca()

    for line in lines:
        for x1,y1,x2,y2 in line:   
            x = [x1,x2]
            y = [y1,y2]
            slope = (y2-y1)/(x2-x1)
            if abs(slope) < 0.4 or abs(slope)>1:
                color = "r"
            elif slope < 0:
                color = "b"
            elif slope > 0:
                color = "g"
            plt.plot(x,y,color, lw=2)
            plt.plot(x1,y1,'x', markersize=12)
            plt.plot(x2,y2,'x', markersize=12)
            note_start = "({0},{1})".format(x1,y1)
            note_end = "({0},{1})".format(x2,y2)
            note_slope = "{0}".format(slope)
            plt.annotate(note_start,xy=(x1,y1))
            plt.annotate(note_end,xy=(x2,y2))
            plt.annotate(note_slope,xy=(x1,y1+10))
    ax.invert_yaxis() 
    plt.show()
    fig.savefig("./examples/{0}.jpg".format(filename),bbox_inches='tight')

    
def draw_polyfit_line(group,l,r,size=(28,16)):
    """
    draw all the end points and the line fit them
    """
    left_x,left_y,right_x,right_y = group
    fig = plt.figure(figsize=size,dpi=100)
    ax = plt.gca() 

    min_left_y = min(left_y) if len(left_y) > 0 else 99999
    max_left_y = max(left_y) if len(left_y) > 0 else -99999
    min_right_y = min(right_y) if len(right_y) > 0 else 99999
    max_right_y = max(right_y) if len(right_y) > 0 else -99999

    
    maxy = max(max_right_y,max_left_y)
    miny = min(min_left_y,min_right_y)
    y1 = maxy
    y2 = miny
    
    if len(left_x) > 0 and len(left_y) > 0:
        plt.plot(left_x,left_y,'xb',markersize="15")
        lx1 = l(maxy)
        lx2 = l(miny)
        plt.plot([lx1,lx2],[y1,y2],'*r',markersize="25")
        plt.plot((lx1,lx2),(y1,y2),'-g')
        
    if len(right_x) > 0 and len(right_y) > 0:  
        plt.plot(right_x,right_y,'xg',markersize="15")
        rx1 = r(maxy)
        rx2 = r(miny)
        plt.plot([rx1,rx2],[y1,y2],'*r',markersize="25")
        plt.plot((rx1,rx2),(y1,y2),'-y')
    
    ax.invert_yaxis() 
    fig.savefig("./examples/polyfit.jpg",bbox_inches='tight')
    
    
def draw_lane_lines(img, lines, color=[255,30, 0], thickness=12,size=(28,16)):
    """
    calculate the slope for each line segments and divide its end points into two groups
    points on lines with postive slope which is considering as the right line
    points on lines with negative slope which is considering as the left line
    also rule out the lines whose absulute value of slope is not betweent 0.4~1
    
    using np.polyfit to get a line most fit the points in left group or right group,and this function
    will return [m,b] for slope and intercept
    
    finally,using poly1d to get the polynomial and calculate the x by y. Then draw two lines on the image
    
    """
    i = 0
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    l = None
    r = None
    topY = int(330) # top of rio
    bottomY = int(img.shape[1]) #bottom of rio
    
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if abs(slope) > 0.4 and abs(slope)<1:
                if slope < 0: #left
                    left_x += [x1,x2]
                    left_y += [y1,y2]
                elif slope > 0: #right
                    right_x += [x1,x2]
                    right_y += [y1,y2]
            
    # polyfit for valid point
    
    # if we get a valid line , do polyfit and append it to the cache
    if len(left_y) > 0 and len(left_y) > 0:
        z1 = np.polyfit(left_y,left_x,1)
        left_line_queue.append(z1) 
        
    # if we get a valid line , do polyfit and append it to the cache
    if len(right_y) > 0 and len(right_y) > 0:
        z2 = np.polyfit(right_y,right_x,1)
        right_line_queue.append(z2)
        
    
    left_ave_z = get_ave_z(left_line_queue)
    right_ave_z = get_ave_z(right_line_queue)
    
    l = np.poly1d(left_ave_z)
    lx1 = int(l(bottomY))
    lx2 = int(l(topY))
    
    r = np.poly1d(right_ave_z)   
    rx1 = int(r(bottomY))
    rx2 = int(r(topY))
        
    cv2.line(img, (lx1,bottomY),(lx2,topY),[0,0,255],15)
    cv2.line(img, (rx1,bottomY),(rx2,topY),[0,0,255],15)
        
    #draw_polyfit_line((left_x,left_y,right_x,right_y),l,r)    
    
    


    
def get_ave_z(queue):
    m = 0
    b = 0
    for p in queue:
        m+=p[0]
        b+=p[1]
    ave_m = m/len(queue)
    ave_b = b/len(queue) 
    return ave_m,ave_b


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[4]:


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

MAX_FRAME_NUM = 6 # max priors line cache numbers 

left_line_queue = deque(maxlen=MAX_FRAME_NUM) # priors left line cache 
right_line_queue = deque(maxlen=MAX_FRAME_NUM) # priors right line cache 
    
    
def _pipeline(image): 
    """
    build a pipeline to draw lane lines on the image
    
    1. tansfrom color image to gray scale image then using gaussian blur to smooth the gray image
    2. using canny to detect edges
    3. set 4 vertices of our interest region and using this region as a mask to rule out the non-related lines.
    4. apply hough transform to detect the line segment 
    5. Draw hough lines on the original image.
    
    """
    # must make a copy otherwise when pipeline 
    # failed we cannot return the original image
    # beacuse the original image has been changed
    # by part of the pipeline
    
    image_copy = np.copy(image) 
    
    imshape = image_copy.shape
    
    kernel_size = 3       # kernel size for gaussian blur
    low_threshold = 100   # low threshold for canny
    high_threshold = 300  # high threshold for canny
    

    # vertices of the region of interest(roi)
    v1 = (0,imshape[0])
    v2 = (450, 290)
    v3 = (490, 290)
    v4 = (imshape[1],imshape[0])
    vertices = np.array([[v1,v2, v3, v4]], dtype=np.int32)
    
    # step1
    gray_image = grayscale(image_copy)
    
    blur_image = gaussian_blur(gray_image, kernel_size)
    #plt.imshow(blur_image)
    # step2
    edges = canny(blur_image, low_threshold, high_threshold)
    #plt.imshow(edges)
    # step3 
    masked_edges,rio= region_of_interest(edges,vertices)
    #plt.imshow(masked_edges)
    
    # step4: hough transform 
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 60    # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 80 #minimum number of pixels making up a line
    max_line_gap = 100    # maximum gap in pixels between connectable line segments
    line_img,lines = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    #plt.imshow(line_img)
    
    #draw_lines_in_xyspace("hough_line_slope",lines)
    
    # step5
    result = weighted_img(line_img, image, α=0.8, β=1., γ=0.)
    
    #save_image("gray_image",gray_image)
    #save_image("blur_image",blur_image)
    #save_image("edges",edges)
    #save_image("masked_edges",masked_edges)
    
    #save_image("hough_lines",line_img)
    #save_image("hough_line_on_origin",result)
    #save_image("lane_lines",result,size=(960,540))
    #save_image("rio",rio)
    #draw_poly("rio",edges,vertices)
    
    
    return result
    



#draw_lines(line_img, lines)

#cv2.line(test_image,v1,v2,color=[255, 0, 0], thickness=2)
#color_edges = np.dstack((edges, edges, edges)) 
#plt.imshow(result_exp)
# Draw the lines on the edge image
#lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 


ximages = os.listdir("test_images/")[1:]
def draw_test_images():
    check_images = []
    for img in ximages :
        filename = "test_images/"+img
        _img = mpimg.imread(filename)
        check_images.append({filename : _pipeline(_img)})
    draw_imgs(check_images)

import os
image_output_dir = 'test_image_output/'


def draw_bad_images():
    bad_images = []
    for img in os.listdir(image_output_dir)[1:]:        
        _img = mpimg.imread(image_output_dir+img)
        bad_images.append({img : _pipeline(_img)})
        
    draw_imgs(bad_images)

draw_test_images()
#draw_bad_images()


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[5]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[6]:



def process_image(image):
    global i
    i+=1
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    try :
        result = _pipeline(image)
    except Exception as e: #need to handle the error that pipeline failed to find any valid line
        result = image
        plt.imsave(image_output_dir + "frame_{0}.jpg".format(i),result)
        #error_img.append({"test" : image})
        
    return result


# Let's try the one with the solid white lane on the right first ...

# In[7]:


i = 0
white_output = 'test_videos_output/solidWhiteRight.mp4'

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')
white_clip = white_clip.resize(0.5)
white_clip.write_gif("examples/solidWhiteRight.gif",fps=10,fuzz=1)
#draw_imgs(error_img)


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[8]:


HTML("""
<video width="960" height="540" controls>

  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[9]:


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
#clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')

yellow_clip = clip2.fl_image(process_image)
get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')
yellow_clip = yellow_clip.resize(0.5)
yellow_clip.write_gif("examples/solidYellowLeft.gif",fps=10,fuzz=1)


# In[10]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[11]:


challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
#clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')
challenge_clip = challenge_clip.resize(0.5)
challenge_clip.write_gif("examples/challenge.gif",fps=10,fuzz=1)


# In[12]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

