from keras.engine.topology import Layer
from keras import backend as K
import numpy as np
from keras.layers import *
import tensorflow as tf
from keras.models import Model
from keras.utils import Sequence
from keras.engine import InputSpec
import cv2
import random

def gram_matrix(x):
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    bs, c, h, w = K.int_shape(x)
    features = K.reshape(x, (-1, c, h*w))
    features_T = K.permute_dimensions(features, (0, 2, 1))
    gram = K.batch_dot(features, features_T)
    return gram

def vgg_preprocess_input(x):
    # BGR Mean values
    vgg_mean = np.array([103.939, 116.779, 123.68], dtype='float32').reshape((1,1,3))
    return x - vgg_mean

def get_output(m, ln, conv=2): return m.get_layer('block{}_conv{}'.format(ln, conv)).output

def RMSE(diff): 
    return K.expand_dims(K.sqrt(K.mean(K.square(diff), [1,2,3])), 0)

def total_variation_loss(x):
    a = K.square(
        x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = K.square(
        x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return K.expand_dims(K.sum(K.pow(a + b, 1.25), axis=[1,2,3]), 0)

def content_fn(x):
    content_loss = 0
    n=len(x)//2
    for i in range(n): content_loss += RMSE(x[i]-x[i+n])/n
    return content_loss

def loadImage(img, img_size):
    I = cv2.imread(img)
    I = cv2.resize(I, img_size)
    return I
#flips a batch of images, flipMode is an integer in range(8)
def flip(x, flipMode):
    if flipMode in [4,5,6,7]:
        x = np.swapaxes(x,1,2)
    if flipMode in [1,3,5,7]:
        x = np.flip(x,1)
    if flipMode in [2,3,6,7]:
        x = np.flip(x,2)
    return x

#works with channels last
class ImageLoader(Sequence):
    def __init__(self, files, img_size = (256, 256), batchSize = 16, flip=False):
        self.files = files
        self.batchSize = batchSize
        self.img_size = img_size
        self.flip = flip
    #gets the number of batches this generator returns
    def __len__(self):
        l,rem = divmod(len(self.files), self.batchSize)
        return (l + (1 if rem > 0 else 0))
    #shuffles data on epoch end
    def on_epoch_end(self):
        random.shuffle(self.files)
    #gets a batch with index = i
    def __getitem__(self, i):
        images = self.files[i*self.batchSize:(i+1)*self.batchSize]
        x = [loadImage(f, self.img_size) for f in images]
        x = np.stack(x, axis=0)
        #cropping and flipping when training        
        if self.flip:
            flipMode = random.randint(0,7) #see flip functoin defined above
            x = flip(x, flipMode)
        targs = np.zeros((len(x),1))
        return x.astype('float32'), [targs]*3
    
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)
    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])
    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

def get_generator(img_shape = (256, 256, 3)):
    def conv(x, nf, kernel_size=3, stride=1, actn=True, pad='valid', bn=True):
        if pad is None: pad = kernel_size//2
        x = Conv2D(nf, kernel_size, strides=stride, padding=pad, use_bias = not bn)(x)
        if actn: 
            x = Activation('relu')(x)
        if bn: 
            x = BatchNormalization()(x)
        return x
    def res_block(x, nf):
        m = conv(x, nf)
        m = conv(m, nf)
        cropped_x = Cropping2D(cropping=((2,2),(2,2)))(x)
        return add([cropped_x, m])
    inp = Input(img_shape)
    d0 = Lambda(lambda x: x/127.5 - 1)(inp)
    d0 = ReflectionPadding2D(padding=(54, 54))(d0)
    #d0 = Lambda(lambda x: tf.pad(x, [[0,0], [54,54], [54,54], [0,0]], 'REFLECT'))(d0)
    x = conv(d0, 32, 9)
    x = conv(x, 64, stride=2)
    x = conv(x, 128, stride=2)
    for i in range(5): x = res_block(x, 128)
    x = UpSampling2D()(x)
    x = conv(x, 64)
    x = UpSampling2D()(x)
    x = conv(x, 32)
    x = conv(x, 3, 11, actn=False)
    x = Activation('tanh')(x)
    output_img = Lambda(lambda x: 127.5*(x+1))(x)
    gen = Model(inp, output_img, name = 'Style_Generator')
    return gen
