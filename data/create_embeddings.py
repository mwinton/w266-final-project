import os
from os import listdir
import shutil
import pandas as pd
import numpy as np
import time
import argparse
import math
import sys

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

import h5py

class imageBatchGenerator(Sequence):
    """
       Generate batches of images for prediction with indices from batch_start to batch_start+batch_size
       The last batch  requested might not be an exact match with the remaining elements in the image file list  passed.
       In that case we pad the batch with duplicate entries. This way the prediction engine sees same elements per batch 
    """

    def __init__(self, dir_prefix, image_filenames, batch_size, loop_start,items_in_loop, max_samples):

        self.batch_size = batch_size
        self.loop_start = loop_start
        self.items_in_loop = items_in_loop

        print("Loop Start -> {}, items in Loop -> {}".format(loop_start,items_in_loop))

        assert (self.loop_start + self.items_in_loop <= max_samples)

        self.image_filenames = image_filenames[self.loop_start : self.loop_start + self.items_in_loop]
 
        if  (self.items_in_loop % self.batch_size != 0)  :
            # pad the list with duplicates
            lastfile = self.image_filenames[-1]
            padLen = self.batch_size  - (self.items_in_loop % self.batch_size)   
            for _ in range(padLen):
                self.image_filenames.append(lastfile)

        self.dir_prefix = dir_prefix
        print("Image batch gen: working on indices {} to {}".format(self.loop_start, self.loop_start + self.items_in_loop))
        print("Size of image files => {}".format(len(self.image_filenames)))


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
        #print("In __getitem__")
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        #imgArray = np.vstack([
        #           self.image_preprocess_VGG16_SAN(imread(self.dir_prefix+file_name))
        #           for file_name in batch_x])
        imgList = []
        for file_name in batch_x:
            #print("getitem -> loading image {}".format(self.dir_prefix+file_name))
            img = image.load_img(self.dir_prefix+file_name,target_size=(448,448))
            imgList.append(img)

        imgArray = np.vstack([
                   self.image_preprocess_VGG16_SAN(img)
                   for img in imgList ])

        #print("__getitem: Image Array shape : ", imgArray.shape)
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

    def writeHd5ImageIds(self):
        imageIds = np.array(self.filesDF['imageId'])
        fileName = self.destDir + "/" + self.targetTask + ".hdf5"
        if os.path.exists(fileName):
            os.remove(fileName)
            
        print("Writing imageIds  in file -> {} ".format(fileName))
        with h5py.File(fileName,"w") as f:
             f.create_dataset("imageIds",   data=imageIds[:self.numSamples])   

    def writeHd5Embedding(self,loop_num,dims):
        """
            Incrementally write numpy arrays to the hdf5 file format
            For the last batch, the imageFeatures might be larger in first dimension than dims passed
            This is because the array generated by predict_generator has to be an exact multiple of batch_size
        """
        fileName = self.destDir + "/" + self.targetTask + ".hdf5"
        items_in_batch = dims[0]
        print("Writing batch of  ({},{},{}) embeddings in file -> {} ".format(dims[0],dims[1],dims[2],fileName))
        with h5py.File(fileName,"a") as f:
             if loop_num == 0:
                 f.create_dataset("embeddings", data=self.imageFeatures[:items_in_batch],maxshape=(None,dims[1],dims[2]),
                                  dtype='float32',compression="gzip")
             else: 
                 f["embeddings"].resize((f["embeddings"].shape[0] + items_in_batch ), axis = 0)
                 f["embeddings"][-items_in_batch:] = self.imageFeatures[:items_in_batch,...]

             # f.create_dataset("embeddings", data=self.imageFeatures[:self.numSamples,...])   
             # >>> dset = myfile.create_dataset("MyDataset", (10, 1024), maxshape=(None, 1024))
             # >>> dset.resize(20, axis=0)   # or dset.resize((20,1024))



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
        self.numSamples = self.filesDF.shape[0]
        self.writeHd5ImageIds()

        ## Large sample sizes cause out of memory errors so chunk out the runs into manageable sizes
        ## and write them to disk before predicting the next chunk. 

        maxSampleSizePerLoop = min(10000,self.numSamples)
        assert(self.numSamples <= self.filesDF.shape[0] )
        fileList = self.filesDF['fileName'].tolist() 
 
        for loopNum in range (math.ceil(self.numSamples/ maxSampleSizePerLoop)):
            items_in_loop = min(maxSampleSizePerLoop,self.numSamples-loopNum*maxSampleSizePerLoop)

            batchGenerator = imageBatchGenerator(self.dirLoc,fileList,self.batch_size,
                             loopNum*maxSampleSizePerLoop,
                             items_in_loop, self.numSamples)
            print("In loop => steps {}".format(math.ceil(maxSampleSizePerLoop / self.batch_size)))
            self.imageFeatures = self.model.predict_generator(batchGenerator,steps = math.ceil(items_in_loop / self.batch_size) , verbose=1)       
            print("Writing Dimensions of Output ->",items_in_loop)
            self.writeHd5Embedding(loopNum,(items_in_loop,self.imageFeatures.shape[1],self.imageFeatures.shape[2]))

def main(rootDir,destDir,targetTask,batch_size=15):
    ## TODO, need to validate the directories
    imgClass = createImageVectors(rootDir,destDir,targetTask,batch_size);     
    imgClass.processImages()


if __name__ == "__main__":

    """
      Sample usage:
        python3 create_embeddings.py --rootDir /home/ram_iyer/vqa_data/  
                                     --destDir /home/ram_iyer/w266-final-project/data/ 
                                     --targetTask "train" 
                                     --batch_size 15
    """

    class writeableDir(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            prospective_dir=values
            if not os.path.isdir(prospective_dir):
                raise argparse.ArgumentTypeError(self,"writeableDir:{0} is not a valid path".format(prospective_dir))
            if os.access(prospective_dir, os.W_OK):
                setattr(namespace,self.dest,prospective_dir)
            else:
                raise argparse.ArgumentTypeError(self,"writeableDir:{0} is not a writable dir".format(prospective_dir))


    class readableDir(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            prospective_dir=values
            if not os.path.isdir(prospective_dir):
                raise argparse.ArgumentTypeError(self,"readableDir:{0} is not a valid path".format(prospective_dir))
            if os.access(prospective_dir, os.R_OK):
                setattr(namespace,self.dest,prospective_dir)
            else:
                raise argparse.ArgumentTypeError(self,"readableDir:{0} is not a readable dir".format(prospective_dir))

    parser = argparse.ArgumentParser(description='Creates Image Embeddings from raw images')
    parser.add_argument("--rootDir",required=True, action=readableDir,
                        help="root directory for all the images. test/train/val directories should reside below this directory")
    parser.add_argument("--destDir",required=True,action=writeableDir,
                        help="destination directory for all the embeddings  ")
    parser.add_argument("--batch_size",type=int,help="sizes of batches to use for running models")

    imageSetChoices = ['train','val','test']
    parser.add_argument('--targetTask',choices=imageSetChoices , required=True,
                         help='specifies target image set for embeddings ')
    args = parser.parse_args()

    main(args.rootDir,args.destDir,args.targetTask,args.batch_size)
 
