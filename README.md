# Vision-Basics

This repository contains five computer vision exercises developed as part of the *Image Analysis and Computer Vision* course at ETH Zurich. The exercises cover a broad range of fundamental vision tasks, including image compression, segmentation, stereo reconstruction, and deep learning–based image classification. Together, they emphasize both algorithmic understanding and efficient implementation using classical methods and modern deep learning frameworks.

## Exercise 1
This exercise involves designing an image compression and reconstruction pipeline for noisy RGB images using a learned codebook from training data. The goal is to minimize reconstruction error while reducing storage and transmission cost, under explicit constraints on codebook size and compressed representation size.
## Exercise 2
This exercise requires implementing an interactive image segmentation algorithm that separates foreground and background using sparse user-provided scribbles. The solution combines k-means clustering to model foreground/background appearance and nearest-neighbor classification to assign a binary segmentation mask evaluated using IoU
## Exercise 3
This project involves implementing a stereo vision pipeline to estimate dense depth maps and per-pixel certainty scores from rectified image pairs. The solution performs camera calibration, correspondence matching, and triangulation using fully vectorized NumPy operations to ensure accurate and computationally efficient 3D reconstruction.
## Exercise 4
This exercise introduces PyTorch and convolutional neural networks through the design and training of a CNN for multi-class image classification.
## Exercise 5
This exercise focuses on image classification with limited training data by fine-tuning a pre-trained convolutional neural network or training a new model to recognize all ten classes. The objective is to leverage transfer learning and regularization techniques to maximize classification accuracy under data-scarce conditions.
