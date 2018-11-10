import argparse
import h5py
import math
import numpy as np
import os
import pandas as pd
import shutil
import sys
import time

from os import listdir
from skimage.io import imread
from skimage.transform import resize

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras import Sequential
from keras.layers import Flatten
from keras.layers import Reshape
from keras.utils import Sequence

from vgg16_options import VGG16Options

class ImageBatchGenerator(Sequence):
    """
       Generate batches of images for prediction with indices from batch_start to batch_start+batch_size
       The last batch  requested might not be an exact match with the remaining elements in the image file list  passed.
       In that case we pad the batch with duplicate entries. This way the prediction engine sees same elements per batch 
    """

    def __init__(self, options, dir_prefix, image_filenames, loop_start,items_in_loop, max_samples):

        self.vggnet_input_dim = options['vggnet_input_dim']  # 448
        self.image_depth = options['image_depth']  # 3
        self.n_image_embed = options['n_image_embed']  # 512
        self.n_image_regions = options['n_image_regions']  # 196
        self.batch_size = options['batch_size']  # 15 
        
        self.loop_start = loop_start
        self.items_in_loop = items_in_loop

        print("Loop Start -> {}, items in Loop -> {}".format(loop_start,items_in_loop))

        assert (self.loop_start + self.items_in_loop <= max_samples)

        self.image_filenames = image_filenames[self.loop_start : self.loop_start + self.items_in_loop]
 
        if  (self.items_in_loop % self.batch_size != 0)  :
            # pad the list with duplicates for the last batch  as Keras requires uniform size batches
            # These are only used to generate embeddings and will be dropped before embeddings are saved.
            lastfile = self.image_filenames[-1]
            padLen = self.batch_size  - (self.items_in_loop % self.batch_size)   
            for _ in range(padLen):
                self.image_filenames.append(lastfile)

        self.dir_prefix = dir_prefix
        print("Image batch gen: working on indices {} to {}".format(self.loop_start, self.loop_start + self.items_in_loop))
        print("Size of image files => {}".format(len(self.image_filenames)))


    def image_preprocess_vgg16_san(self,img):
        # convert the image pixels to a numpy array
        img = image.img_to_array(img)
        #resize to fit the dimensions of the SAN paper
        #img = cv2.resize(img,dsize=(self.vggnet_input_dim, self.vggnet_input_dim))
        # reshape data for the model
        #img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        #img = img.reshape((1, self.vggnet_input_dim, self.vggnet_input_dim, 3 ))
        assert(img.shape[0] == self.vggnet_input_dim)
        assert(img.shape[1] == self.vggnet_input_dim)
        assert(img.shape[2] == self.image_depth)
        # prepare the image for the VGG model
        img = np.expand_dims(img,axis=0)
        img = preprocess_input(img)

        return img

    def __len__(self):
        return (len(self.image_filenames)//(self.batch_size) + 1)

    def __getitem__(self, idx):
        """
            provide the images related to  batch (idx)
            will be called from predict_generator function 
            Also preprocess the images loaded from disk to confirm with the VGG16 model
        """
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        img_list = []
        for file_name in batch_x:
            #print("getitem -> loading image {}".format(self.dir_prefix+file_name))
            img = image.load_img(self.dir_prefix+file_name,
                                 target_size=(self.vggnet_input_dim, self.vggnet_input_dim))
            img_list.append(img)

        img_array = np.vstack([
                   self.image_preprocess_vgg16_san(img)
                   for img in img_list ])

        return img_array


class CreateImageVectors():

    def __init__(self, options, root_dir, dest_dir, target_task,
                 dataset_type="mscoco", model_name="VGG16",
                 test_dir="test2015/", train_dir="train2014/", val_dir="val2014/"):

        self.vggnet_input_dim = options['vggnet_input_dim']  # 448
        self.image_depth = options['image_depth']  # 3
        self.n_image_embed = options['n_image_embed']  # 512
        self.n_image_regions = options['n_image_regions']  # 196
        self.batch_size = options['batch_size']  # 15 
        assert(self.batch_size > 10)

        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.test_dir = self.root_dir + "/images/mscoco/" + test_dir
        self.train_dir = self.root_dir + "/images/mscoco/" + train_dir 
        self.val_dir = self.root_dir + "/images/mscoco/" + val_dir
        self.model_name = model_name
        self.dest_dir = dest_dir
        self.target_task = target_task

    def get_coco_files(self):
        if (self.target_task == "test"):
            dir_loc = self.test_dir
            id_start = len('COCO_test2015_')
        elif(self.target_task == "train"):
            dir_loc = self.train_dir
            id_start = len('COCO_train2014_')
        else:
            dir_loc = self.val_dir
            id_start = len('COCO_val2014_')
        #total number of numeric digits in the image id is 12.
        end_id =  id_start+12

        files = [(int(file_name[id_start:end_id]), file_name) for file_name in os.listdir(dir_loc)]
        self.orig_file_size = len(files)
        self.files_df = pd.DataFrame(files,columns=['imageId','file_name'])
        self.dir_loc = dir_loc

        return        

    def create_vgg16_model(self):
        base_model = VGG16(include_top = False,
                          weights = 'imagenet',
                          input_shape = (self.vggnet_input_dim, self.vggnet_input_dim, self.image_depth))
        self.model = Sequential()
        self.model.add(base_model)
        self.model.add(Reshape((self.n_image_regions, self.n_image_embed)))
        self.model.summary()

    def write_hd5_image_ids(self):
        img_ids = np.array(self.files_df['imageId'])
        file_name = self.dest_dir + "/" + self.target_task + ".hdf5"
        if os.path.exists(file_name):
            os.remove(file_name)
            
        print("Writing img_ids  in file -> {} ".format(file_name))
        with h5py.File(file_name,"w") as f:
             f.create_dataset("img_ids",   data=img_ids[:self.num_samples])   

    def write_hd5_embedding(self, loop_num,dims):
        """
            Incrementally write numpy arrays to the hdf5 file format
            For the last batch, the img_features might be larger in first dimension than dims passed
            This is because the array generated by predict_generator has to be an exact multiple of batch_size
        """
        file_name = self.dest_dir + "/" + self.target_task + ".hdf5"
        items_in_batch = dims[0]
        print("Writing batch of  ({},{},{}) embeddings in file -> {} ".format(dims[0],dims[1],dims[2],file_name))
        with h5py.File(file_name, "a") as f:
             if loop_num == 0:
                 f.create_dataset("embeddings",
                                  data=self.img_features[:items_in_batch],
                                  maxshape=(None,dims[1],dims[2]),
                                  dtype='float32',compression="gzip")
             else: 
                 f["embeddings"].resize((f["embeddings"].shape[0] + items_in_batch ), axis = 0)
                 f["embeddings"][-items_in_batch:] = self.img_features[:items_in_batch,...]

             # f.create_dataset("embeddings", data=self.img_features[:self.num_samples,...])   
             # >>> dset = myfile.create_dataset("MyDataset", (10, 1024), maxshape=(None, 1024))
             # >>> dset.resize(20, axis=0)   # or dset.resize((20,1024))


    def process_images(self):

        if(self.dataset_type == "mscoco"):
            self.get_coco_files()
        else:
            print("Only MSCOCO is currently supported for dataset_type")
            return

        if(self.model_name=="VGG16"):
            self.create_vgg16_model()   
        else:
            print("Only VGG16 model is currently supported")

        #num_samples = self.files_df.shape[0]
        self.num_samples = self.files_df.shape[0]
        self.write_hd5_image_ids()

        ## Large sample sizes cause out of memory errors so chunk out the runs into manageable sizes
        ## and write them to disk before predicting the next chunk. 

        max_sample_size_per_loop = min(10000,self.num_samples)
        assert(self.num_samples <= self.files_df.shape[0] )
        file_list = self.files_df['file_name'].tolist() 
 
        for loop_num in range (math.ceil(self.num_samples / max_sample_size_per_loop)):
            items_in_loop = min(max_sample_size_per_loop,self.num_samples-loop_num*max_sample_size_per_loop)

            batch_generator = ImageBatchGenerator(options, self.dir_loc, file_list,
                             loop_num * max_sample_size_per_loop,
                             items_in_loop, self.num_samples)
            print("In loop => steps {}".format(math.ceil(max_sample_size_per_loop / self.batch_size)))
            self.img_features = self.model.predict_generator(batch_generator,
                                                             steps = math.ceil(items_in_loop / self.batch_size),
                                                             verbose=1)       
            print("Writing Dimensions of Output ->", items_in_loop)
            self.write_hd5_embedding(loop_num,(items_in_loop,self.img_features.shape[1],self.img_features.shape[2]))

def main(options, root_dir, dest_dir, target_task):
    ## TODO, need to validate the directories
    img_class = CreateImageVectors(options, root_dir, dest_dir, target_task);     
    img_class.process_images()


if __name__ == "__main__":

    """
      Sample usage:
        python3 create_embeddings.py --root_dir /home/ram_iyer/vqa_data/  
                                     --dest_dir /home/ram_iyer/w266-final-project/data/ 
                                     --target_task "train" 
                                     --batch_size 15
    """

    class WriteableDir(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            prospective_dir=values
            if not os.path.isdir(prospective_dir):
                raise argparse.ArgumentTypeError(self,"WriteableDir:{0} is not a valid path".format(prospective_dir))
            if os.access(prospective_dir, os.W_OK):
                setattr(namespace,self.dest,prospective_dir)
            else:
                raise argparse.ArgumentTypeError(self,"WriteableDir:{0} is not a writable dir".format(prospective_dir))

    class ReadableDir(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            prospective_dir=values
            if not os.path.isdir(prospective_dir):
                raise argparse.ArgumentTypeError(self,"ReadableDir:{0} is not a valid path".format(prospective_dir))
            if os.access(prospective_dir, os.R_OK):
                setattr(namespace,self.dest,prospective_dir)
            else:
                raise argparse.ArgumentTypeError(self,"ReadableDir:{0} is not a readable dir".format(prospective_dir))

    parser = argparse.ArgumentParser(description='Creates Image Embeddings from raw images',
                        epilog='W266 Final Project (Fall 2018) by Rachel Ho, Ram Iyer, Mike Winton')
    parser.add_argument("--root_dir",required=True, action=ReadableDir,
                        help="root directory for all the images. test/train/val directories should reside below this directory")
    parser.add_argument("--dest_dir",required=True,action=WriteableDir,
                        help="destination directory for all the embeddings  ")
    parser.add_argument("--batch_size", type=int, default=15,
                        help="sizes of batches to use for running models")

    imageSetChoices = ['train','val','test']
    parser.add_argument('--target_task',choices=imageSetChoices , required=True,
                         help='specifies target image set for embeddings ')
    args = parser.parse_args()

    # load model options from config file
    options = VGG16Options().get_options()

    # override default for batch_size if specified
    if args.batch_size:
        options['batch_size'] = args.batch_size

    main(options, args.root_dir, args.dest_dir, args.target_task)
 
