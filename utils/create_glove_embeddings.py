import argparse
import h5py
import numpy as np
import os
import pickle

def main(glove_dir, dest_dir):
    
    embeddings_index = {}
    glove_path = os.path.join(glove_dir, 'glove.840B.300d.txt')
    print('Loading GloVe embeddings from ->', glove_path)
    
    bad_count = 0
    f = open(glove_path)
    for line in f:
        values = line.split()
        try:
            word = values[:-300][0]
            coefs = np.asarray(values[-300:], dtype='float32')
        except IndexError:
            bad_count += 1
            pass # throw away malformed samples
        embeddings_index[word] = coefs
    f.close()
    print('Found {} word vectors.  {} malformed vectors discarded.'.format(len(embeddings_index), bad_count))

    dest_path = os.path.join(dest_dir, 'glove.p')
    with open(dest_path, 'wb') as f:
        pickle.dump(embeddings_index, f)
    print('GloVe embeddings saved to {}'.format(dest_path))

if __name__ == "__main__":

    """
      Sample usage:
        python3 create_glove_embeddings.py --root_dir /home/mwinton/glove/ --dest_dir /home/mwinton/glove/ 
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

    parser = argparse.ArgumentParser(description='Creates GloVe embeddings from downloaded Stanford text file',
                        epilog='W266 Final Project (Fall 2018) by Rachel Ho, Ram Iyer, Mike Winton')
    parser.add_argument("--glove_dir",required=True, action=ReadableDir,
                        help="directory where the GloVe text file is located")
    parser.add_argument("--dest_dir",required=True,action=WriteableDir,
                        help="destination directory for all the embeddings  ")

    args = parser.parse_args()
    glove_dir = args.glove_dir
    dest_dir = args.dest_dir

    main(glove_dir, dest_dir)
 
