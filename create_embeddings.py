import os
from os import listdir
import shutil
import pandas as pd
import numpy as np
import time
import argparse

from skimage.io import imread
from skimage.transform import resize
from keras.utils import Sequence


from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras import Sequential
from keras.layers import Flatten
from keras.layers import Reshape

import cv2
import h5py

class imageBatchGenerator(Sequence):

    def __init__(self, dir_prefix, image_filenames, batch_size):
        self.image_filenames = image_filenames
        self.batch_size = batch_size
        self.numFiles = len(image_filenames)
        # If length of filenames are not an exact multiple, pad the list
        padLen = self.batch_size - (self.numFiles % self.batch_size)
        for _ in range(padLen):
            self.image_filenames.append(self.image_filenames[self.numFiles - 1])
            self.numFiles += 1
        self.dir_prefix = dir_prefix

    def image_preprocess_VGG16_SAN(self,img):
        # convert the image pixels to a numpy array
        img = image.img_to_array(img)
        #resize to fit the dimensions of the SAN paper
        #img = cv2.resize(img,dsize=(448,448))
        # reshape data for the model
        #img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        #img = img.reshape((1, 448, 448, 3 ))
        assert(img.shape[0] == 448)
        assert(img.shape[1] == 448)
        assert(img.shape[2] == 3)
        # prepare the image for the VGG model
        img = np.expand_dims(img,axis=0)
        img = preprocess_input(img)

        return img

    def __len__(self):
        return (len(self.image_filenames)//(self.batch_size) + 1)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        #imgArray = np.vstack([
        #           self.image_preprocess_VGG16_SAN(imread(self.dir_prefix+file_name))
        #           for file_name in batch_x])
        imgArray = np.vstack([
                   self.image_preprocess_VGG16_SAN(image.load_img(self.dir_prefix+file_name,target_size=(448,448)))
                   for file_name in batch_x])
        return imgArray


class createImageVectors():

    def __init__(self,rootDir,destDir,targetTask,batch_size,
                 datasetType="mscoco",modelName="VGG16",testDir="test2015/",trainDir="train2014/",valDir="val2014/"):
        self.rootDir = rootDir
        self.datasetType = datasetType
        self.testDir = self.rootDir + "/images/mscoco/" + testDir
        self.trainDir = self.rootDir + "/images/mscoco/" + trainDir 
        self.valDir = self.rootDir + "/images/mscoco/" + valDir
        self.modelName = modelName
        self.batch_size = batch_size
        assert(self.batch_size > 10)
        self.destDir = destDir
        self.targetTask = targetTask

    def getCocoFiles(self):
        if (self.targetTask == "test"):
            dirLoc = self.testDir
            id_start = len('COCO_test2015_')
        elif(self.targetTask == "train"):
            dirLoc = self.trainDir
            id_start = len('COCO_train2014_')
        else:
            dirLoc = self.valDir
            id_start = len('COCO_val2014_')
        #total number of numeric digits in the image id is 12.
        end_id =  id_start+12

        files = [(int(fileName[id_start:end_id]), fileName) for fileName in os.listdir(dirLoc)]
        self.origFileSize = len(files)
        self.filesDF = pd.DataFrame(files,columns=['imageId','fileName'])
        self.dirLoc = dirLoc

        return        

    def createVGG16Model(self):
        baseModel = VGG16(include_top = False,
                          weights = 'imagenet',
                          input_shape = (448,448,3))
        self.model = Sequential()
        self.model.add(baseModel)
        self.model.add(Reshape((196,512)))
        self.model.summary()

    def writeHd5Embedding(self):
        imageIds = np.array(self.filesDF['imageId'])
        fileName = self.destDir + "/" + self.targetTask + ".hdf5"
        print("Writing embeddings in file -> ", fileName)
        with h5py.File(fileName,"w") as f:
             f.create_dataset("imageIds",   data=imageIds)   
             f.create_dataset("embeddings", data=self.imageFeatures)   
        

    def processImages(self):

        if(self.datasetType == "mscoco"):
            self.getCocoFiles()
        else:
            print("Only MSCO is currently supported for datasetType")
            return

        if(self.modelName=="VGG16"):
            self.createVGG16Model()   
        else:
            print("Only VGG16 model is currently supported")

        #numSamples = self.filesDF.shape[0]
        numSamples = 1000

        batchGenerator = imageBatchGenerator(self.dirLoc,self.filesDF['fileName'].tolist(),self.batch_size)
        self.imageFeatures = self.model.predict_generator(batchGenerator,steps = (numSamples // self.batch_size) + 1, verbose=1)       
        print("Dimensions of Output ->",self.imageFeatures.shape)

def main(rootDir,destDir,targetTask,batch_size=15):
    ## TODO, need to validate the directories
    imgClass = createImageVectors(rootDir,destDir,targetTask,batch_size);     
    imgClass.processImages()
    imgClass.writeHd5Embedding()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Creates Image Embeddings from raw images')
    parser.add_argument("--rootDir",required=True,help="root directory for all the images. test/train/val directories should reside below this directory")
    parser.add_argument("--destDir",required=True,help="destination directory for all the embeddings  ")
    parser.add_argument("--batch_size",type=int,help="sizes of batches to use for running models")

    imageSetChoices = ['train','val','test']
    parser.add_argument('--targetTask',choices=imageSetChoices , required=True, help='specifies target image set for embeddings ')
    args = parser.parse_args()

    main(args.rootDir,args.destDir,args.targetTask,args.batch_size)
 
