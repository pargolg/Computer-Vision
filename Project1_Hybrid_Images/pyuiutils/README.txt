# This is a High Low pass, Gaussian blurred image code

The hybrid image example is of 2 famous clown. One from Batman Dark Horse, the other from IT.

functions include:

    cross_correlation_2d
    convolve_2d
    gaussian_blur_kernel_2d
    low_pass
    high_pass

Image Filtering. Image filtering (or convolution) is a fundamental image processing tool. See chapter 3.2 of Szeliski and the lecture materials to learn about image filtering (specifically linear filtering). Numpy has numerous built in and efficient functions to perform image filtering, but you will be writing your own such function from scratch for this assignment. More specifically, you will implement cross_correlation_2d, followed by convolve_2d which would use cross_correlation_2d.

Gaussian Blur. As you have seen in the lectures, there are a few different way to blur an image, for example taking an unweighted average of the neighboring pixels. Gaussian blur is a special kind of weighted averaging of neighboring pixels, and is described in the lecture slides. To implement Gaussian blur, you will implement a function gaussian_blur_kernel_2d that produces a kernel of a given height and width which can then be passed to convolve_2d from above, along with an image, to produce a blurred version of the image.

High and Low Pass Filters.Recall that a low pass filter is one that removed the fine details from an image (or, really, any signal), whereas a high pass filter only retails the fine details, and gets rid of the coarse details from an image. Thus, using Gaussian blurring as described above, implement high_pass and low_pass functions.

Hybrid Images. A hybrid image is the sum of a low-pass filtered version of the one image and a high-pass filtered version of a second image. There is a free parameter, which can be tuned for each image pair, which controls how much high frequency to remove from the first image and how much low frequency to leave in the second image. This is called the "cutoff-frequency". In the paper it is suggested to use two cutoff frequencies (one tuned for each image) and you are free to try that, as well. In the starter code, the cutoff frequency is controlled by changing the standard deviation (sigma) of the Gausian filter used in constructing the hybrid images. We provide you with the code for creating a hybrid image, using the functions described above.


To run the GUI to create your own hybrid image, run the following command:

python gui.py -t resources/sample-correspondance.json -c resources/sample-config.json
