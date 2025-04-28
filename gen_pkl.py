'''
Python 3.6 
Pytorch 0.4
Written by Hongyu Wang in Beihang university
'''

import os
import sys
import pandas as pd
import pickle as pkl
import numpy
import imageio.v2 as imageio
#from scipy.misc.pilutil import imread, imresize, imsave
from PIL import Image

# Function to process images and save to pickle
def process_images(image_path, caption_file, output_file):
    oupFp_feature = open(output_file, 'wb')
    features = {}
    channels = 1
    sentNum = 0

    with open(caption_file) as scpFile:
        while True:
            line = scpFile.readline().strip()  # remove the '\r\n'
            if not line:
                break
            else:
                key = line.split('\t')[0]
                image_file = image_path + key + '_' + str(0) + '.bmp'
                if not os.path.exists(image_file):
                    print(f"File not found: {image_file}")
                    continue
                im = imageio.imread(image_file)
                mat = numpy.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')
                for channel in range(channels):
                    image_file = image_path + key + '_' + str(channel) + '.bmp'
                    if not os.path.exists(image_file):
                        print(f"File not found: {image_file}")
                        continue
                    im = imageio.imread(image_file)
                    mat[channel, :, :] = im
                sentNum += 1
                features[key] = mat
                if sentNum % 500 == 0:
                    print('Processed sentences', sentNum)

    print('Load images done. Sentence number', sentNum)
    pkl.dump(features, oupFp_feature)
    print('Save file done')
    oupFp_feature.close()


# Paths for test and training data
test_image_path = 'C:/Users/OP9020/Documents/Pytorch-Handwritten-Mathematical-Expression-Recognition/off_image_test/'
train_image_path = 'C:/Users/OP9020/Documents/Pytorch-Handwritten-Mathematical-Expression-Recognition/off_image_train/'
test_caption_file = 'test_caption.txt'
train_caption_file = 'train_caption.txt'
test_output_file = 'offline-test.pkl'
train_output_file = 'offline-train.pkl'

# Process test and training images
process_images(test_image_path, test_caption_file, test_output_file)
process_images(train_image_path, train_caption_file, train_output_file)