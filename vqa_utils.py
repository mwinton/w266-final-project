# standard modules
import numpy as np

# our own imports
from vqa_options import ModelOptions

class FakeData(object):

    def __init__(self, options):
        ''' generate fake image, sentence data of the right shapes; also labels'''

        verbose = options['verbose']
       
        print('Generating fake data to exercise the pipeline...')
        # specify how many observations to generate (axis=0)
        # TODO: test with non-integer n_fake_batches to make sure pipeline works with partial batches
        n_fake_batches = 3
        batch_size = options['batch_size']
        n_fake_rows = n_fake_batches * batch_size
        
        # generate fake images as
        #   [batch_size, image_input_dim, image_input_dim, image_input_depth] of floats
        image_input_dim = options['vggnet_input_dim']
        image_input_depth = options['image_depth']        
        self.fake_images_x = np.random.random(size=(n_fake_rows,
                                                    image_input_dim,
                                                    image_input_dim,
                                                    image_input_depth))
        self.fake_images_x = 255 * self.fake_images_x
        if verbose: print('fake_images_x shape:', self.fake_images_x.shape)
        # also generate smaller (batch_size) test set
        self.fake_images_x_test = np.random.random(size=(batch_size,
                                                         image_input_dim,
                                                         image_input_dim,
                                                         image_input_depth))
        self.fake_images_x_test = 255 * self.fake_images_x_test
        if verbose: print('fake_images_x_test shape:', self.fake_images_x_test.shape)
        
        # generate fake sentences as 
        #   [batch_size, max_time] integers between 1 and V
        max_t = options['max_sentence_len']
        V = options['n_vocab']        
        self.fake_sentences_x = np.random.randint(V, size=(n_fake_rows, max_t))
        if verbose: print('fake_sentences_x shape:', self.fake_sentences_x.shape)
        # also generate smaller (batch_size) test set
        self.fake_sentences_x_test = np.random.randint(V, size=(batch_size, max_t))
        if verbose: print('fake_sentences_x_test shape:', self.fake_sentences_x_test.shape)

        # generate list of fake labels
        n_answer_classes = options['n_answer_classes']
        self.fake_y = np.random.randint(n_answer_classes, size=(n_fake_rows, 1))
        if verbose: print('fake_y shape:', self.fake_y.shape)
        # also generate smaller (batch_size) test set
        self.fake_y_test = np.random.randint(n_answer_classes, size=(batch_size, 1))
        if verbose: print('fake_y_test shape:', self.fake_y_test.shape)
        
    def get_fakes(self):
        return (self.fake_images_x, self.fake_sentences_x, self.fake_y,
                self.fake_images_x_test, self.fake_sentences_x_test, self.fake_y_test)
    
# for debugging, vqa_util.py can be run directly
if __name__ == '__main__':
    options = ModelOptions().get_options()
    _, _, _, _, _, _ = FakeData(options).get_fakes()