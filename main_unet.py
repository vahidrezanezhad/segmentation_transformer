import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
#from keras import layers
from tensorflow.keras.regularizers import l2
from sacred import Experiment
import os
from tensorflow.python.keras import backend as tensorflow_backend
from utils import *
from metrics import *
import sys

weight_decay = 1e-7
mlp_head_units = [2048, 1024]
input_shape = (896, 896, 3)#(32, 32, 3)
projection_dim = 64
num_classes = 30
transformer_layers =8
num_heads =4
MERGE_AXIS=-1
##n_classes =2
IMAGE_ORDERING = 'channels_last'
bn_axis=3
resnet50_Weights_path='./pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        print(tf.shape(images)[1],'images')
        print(self.patch_size,'self.patch_size')
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        print(patches.shape,patch_dims,'patch_dims')
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config
    
#import matplotlib.pyplot as plt

#plt.figure(figsize=(4, 4))
#image = x_train[np.random.choice(range(x_train.shape[0]))]
#plt.imshow(image.astype("uint8"))
#plt.axis("off")

#resized_image = tf.image.resize(
    #tf.convert_to_tensor([image]), size=(image_size, image_size)
#)
#patches = Patches(patch_size)(resized_image)
#print(f"Image size: {image_size} X {image_size}")
#print(f"Patch size: {patch_size} X {patch_size}")
#print(f"Patches per image: {patches.shape[1]}")
#print(f"Elements per patch: {patches.shape[-1]}")

#n = int(np.sqrt(patches.shape[1]))
#plt.figure(figsize=(4, 4))
#for i, patch in enumerate(patches[0]):
    #ax = plt.subplot(n, n, i + 1)
    #patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    #plt.imshow(patch_img.numpy().astype("uint8"))
    #plt.axis("off")
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
        })
        return config
    
    
def create_vit_classifier(n_classes):
    inputs = layers.Input(shape=input_shape)
    IMAGE_ORDERING = 'channels_last'
    bn_axis=3
    
    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(inputs)
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING, strides=(2, 2),kernel_regularizer=l2(weight_decay), name='conv1')(x)
    f1 = x
    
    
    x = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING, padding='same', strides=(2, 2),kernel_regularizer=l2(weight_decay), name='conv11')(x)
    f2 = x
    
    x = Conv2D(256, (3, 3), data_format=IMAGE_ORDERING, padding='same', strides=(2, 2),kernel_regularizer=l2(weight_decay), name='conv111')(x)
    f3 = x
    
    #print(x.shape(),'gada')
    
    patch_size = 1
    num_patches = 28*28
    patches = Patches(patch_size)(x)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
    
    encoded_patches = tf.reshape(encoded_patches, [-1, 28, 28, 64])
    
    
    v1024_2048 =  Conv2D( 512 , (1, 1) , padding='same', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) )( encoded_patches )
    v1024_2048 = ( BatchNormalization(axis=bn_axis))(v1024_2048)
    v1024_2048 = Activation('relu')(v1024_2048)
    
    
    #o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(v1024_2048)
    o = ( concatenate([ v1024_2048 ,f3],axis=MERGE_AXIS )  )
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    
    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([ o ,f2],axis=MERGE_AXIS )  )
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([ o ,f1],axis=MERGE_AXIS )  )
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([o,inputs],axis=MERGE_AXIS ) )
    o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 32 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) ))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    o =  Conv2D( n_classes , (1, 1) , padding='same', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) )( o )
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = (Activation('softmax'))(o)
    
    
    #representation = layers.Flatten()(o)
    #representation = layers.Dropout(0.5)(representation)
    ## Add MLP.
    #features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    ## Classify outputs.
    #logits = layers.Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=o)
    return model
def one_side_pad( x ):
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    if IMAGE_ORDERING == 'channels_first':
        x = Lambda(lambda x : x[: , : , :-1 , :-1 ] )(x)
    elif IMAGE_ORDERING == 'channels_last':
        x = Lambda(lambda x : x[: , :-1 , :-1 , :  ] )(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    
    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1) , data_format=IMAGE_ORDERING , name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size , data_format=IMAGE_ORDERING ,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3 , (1, 1), data_format=IMAGE_ORDERING , name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    
    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1) , data_format=IMAGE_ORDERING  , strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size , data_format=IMAGE_ORDERING  , padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1) , data_format=IMAGE_ORDERING  , name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1) , data_format=IMAGE_ORDERING  , strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def create_vit_classifier_resnet(n_classes, pretraining=False):
    
    inputs = layers.Input(shape=input_shape)
    IMAGE_ORDERING = 'channels_last'
    bn_axis=3


    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(inputs)
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING, strides=(2, 2),kernel_regularizer=l2(weight_decay), name='conv1')(x)
    f1 = x

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3) , data_format=IMAGE_ORDERING , strides=(2, 2))(x)
    

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = one_side_pad(x )


    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x 

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    f4 = x 

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    f5 = x 
    
    if pretraining:
        model=keras.Model( inputs , x ).load_weights(resnet50_Weights_path)
    ##print(x.shape[2],'gada pahooo')
    patch_size = 1
    num_patches = x.shape[1]*x.shape[2]
    patches = Patches(patch_size)(x)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
    
    encoded_patches = tf.reshape(encoded_patches, [-1, x.shape[1], x.shape[2], 64])

    
    v1024_2048 =  Conv2D( 1024 , (1, 1) , padding='same', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) )( encoded_patches )
    v1024_2048 = ( BatchNormalization(axis=bn_axis))(v1024_2048)
    v1024_2048 = Activation('relu')(v1024_2048)
    
    
    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(v1024_2048)
    o = ( concatenate([ o ,f4],axis=MERGE_AXIS )  )
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    
    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([ o ,f3],axis=MERGE_AXIS )  )
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    
    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([ o ,f2],axis=MERGE_AXIS )  )
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    
    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([ o ,f1],axis=MERGE_AXIS )  )
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    
    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([o,inputs],axis=MERGE_AXIS ) )
    o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 32 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) ))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    o =  Conv2D( n_classes , (1, 1) , padding='same', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) )( o )
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = (Activation('softmax'))(o)
    
    
    model = keras.Model(inputs=inputs, outputs=o)
    return model



def configuration():
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    session = tf.compat.v1.Session(config=config)  # tf.InteractiveSession()
    tensorflow_backend.set_session(session)
    
    #keras.backend.clear_session()
    #tf.reset_default_graph()
    #warnings.filterwarnings('ignore')
    
    #os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    #config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    
    
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction=0.95#0.95
    #config.gpu_options.visible_device_list="0"
    #set_session(tf.Session(config=config))

def get_dirs_or_files(input_data):
    if os.path.isdir(input_data):
        image_input, labels_input = os.path.join(input_data, 'images/'), os.path.join(input_data, 'labels/')
        # Check if training dir exists
        assert os.path.isdir(image_input), "{} is not a directory".format(image_input)
        assert os.path.isdir(labels_input), "{} is not a directory".format(labels_input)
    return image_input, labels_input

ex = Experiment()

@ex.config
def config_params():
    n_classes=None # Number of classes. If your case study is binary case the set it to 2 and otherwise give your number of cases.
    n_epochs=1
    input_height=224*1
    input_width=224*1 
    weight_decay=1e-6 # Weight decay of l2 regularization of model layers.
    n_batch=1 # Number of batches at each iteration.
    learning_rate=1e-4
    patches=False # Make patches of image in order to use all information of image. In the case of page
    # extraction this should be set to false since model should see all image.
    augmentation=False
    flip_aug=False # Flip image (augmentation).
    blur_aug=False # Blur patches of image (augmentation). 
    scaling=False # Scaling of patches (augmentation) will be imposed if this set to true.
    binarization=False # Otsu thresholding. Used for augmentation in the case of binary case like textline prediction. For multicases should not be applied.
    dir_train=None # Directory of training dataset (sub-folders should be named images and labels).
    dir_eval=None # Directory of validation dataset (sub-folders should be named images and labels).
    dir_output=None # Directory of output where the model should be saved.
    pretraining=False # Set true to load pretrained weights of resnet50 encoder.
    scaling_bluring=False
    scaling_binarization=False
    scaling_flip=False
    thetha=[10,-10]
    blur_k=['blur','guass','median'] # Used in order to blur image. Used for augmentation.
    scales= [ 0.5, 2 ] # Scale patches with these scales. Used for augmentation.
    flip_index=[0,1,-1] # Flip image. Used for augmentation.
    continue_training = False # If 
    index_start = 0
    dir_of_start_model = ''
    is_loss_soft_dice = False
    weighted_loss = False
    data_is_provided = False

@ex.automain
def run(n_classes,n_epochs,input_height,
        input_width,weight_decay,weighted_loss,
        index_start,dir_of_start_model,is_loss_soft_dice,
        n_batch,patches,augmentation,flip_aug
        ,blur_aug,scaling, binarization,
        blur_k,scales,dir_train,data_is_provided,
        scaling_bluring,scaling_binarization,rotation,
        rotation_not_90,thetha,scaling_flip,continue_training,
        flip_index,dir_eval ,dir_output,pretraining,learning_rate):
    
    
        
    
    model = create_vit_classifier_resnet(n_classes,pretraining)
    #if you want to see the model structure just uncomment model summary.
    model.summary()
    #sys.exit()
    
    
    if data_is_provided:
        dir_train_flowing=os.path.join(dir_output,'train')
        dir_eval_flowing=os.path.join(dir_output,'eval')
        
        dir_flow_train_imgs=os.path.join(dir_train_flowing,'images')
        dir_flow_train_labels=os.path.join(dir_train_flowing,'labels')
        
        dir_flow_eval_imgs=os.path.join(dir_eval_flowing,'images')
        dir_flow_eval_labels=os.path.join(dir_eval_flowing,'labels')
        
        configuration()
        
    else:
        dir_img,dir_seg=get_dirs_or_files(dir_train)
        dir_img_val,dir_seg_val=get_dirs_or_files(dir_eval)
        
        # make first a directory in output for both training and evaluations in order to flow data from these directories.
        dir_train_flowing=os.path.join(dir_output,'train')
        dir_eval_flowing=os.path.join(dir_output,'eval')
        
        dir_flow_train_imgs=os.path.join(dir_train_flowing,'images/')
        dir_flow_train_labels=os.path.join(dir_train_flowing,'labels/')
        
        dir_flow_eval_imgs=os.path.join(dir_eval_flowing,'images/')
        dir_flow_eval_labels=os.path.join(dir_eval_flowing,'labels/')
        
        if os.path.isdir(dir_train_flowing):
            os.system('rm -rf '+dir_train_flowing)
            os.makedirs(dir_train_flowing)
        else:
            os.makedirs(dir_train_flowing)
            
        if os.path.isdir(dir_eval_flowing):
            os.system('rm -rf '+dir_eval_flowing)
            os.makedirs(dir_eval_flowing)
        else:
            os.makedirs(dir_eval_flowing)
            

        os.mkdir(dir_flow_train_imgs)
        os.mkdir(dir_flow_train_labels)
        
        os.mkdir(dir_flow_eval_imgs)
        os.mkdir(dir_flow_eval_labels)
        
        
        #set the gpu configuration
        configuration()


        #writing patches into a sub-folder in order to be flowed from directory. 
        provide_patches(dir_img,dir_seg,dir_flow_train_imgs,
                        dir_flow_train_labels,
                        input_height,input_width,blur_k,blur_aug,
                        flip_aug,binarization,scaling,scales,flip_index,
                        scaling_bluring,scaling_binarization,rotation,
                        rotation_not_90,thetha,scaling_flip,
                        augmentation=augmentation,patches=patches)
        
        provide_patches(dir_img_val,dir_seg_val,dir_flow_eval_imgs,
                        dir_flow_eval_labels,
                        input_height,input_width,blur_k,blur_aug,
                        flip_aug,binarization,scaling,scales,flip_index,
                        scaling_bluring,scaling_binarization,rotation,
                        rotation_not_90,thetha,scaling_flip,
                        augmentation=False,patches=patches)
        

    
    if weighted_loss:
        weights=np.zeros(n_classes)
        if data_is_provided:
            for obj in os.listdir(dir_flow_train_labels):
                try:
                    label_obj=cv2.imread(dir_flow_train_labels+'/'+obj)
                    label_obj_one_hot=get_one_hot( label_obj,label_obj.shape[0],label_obj.shape[1],n_classes)
                    weights+=(label_obj_one_hot.sum(axis=0)).sum(axis=0)
                except:
                    pass
        else:
            
            for obj in os.listdir(dir_seg):
                try:
                    label_obj=cv2.imread(dir_seg+'/'+obj)
                    label_obj_one_hot=get_one_hot( label_obj,label_obj.shape[0],label_obj.shape[1],n_classes)
                    weights+=(label_obj_one_hot.sum(axis=0)).sum(axis=0)
                except:
                    pass
            

        weights=1.00/weights
        
        weights=weights/float(np.sum(weights))
        weights=weights/float(np.min(weights))
        weights=weights/float(np.sum(weights))
        
        
    model = create_vit_classifier_resnet(n_classes,pretraining)
    
    #if you want to see the model structure just uncomment model summary.
    #model.summary()
    

    if not is_loss_soft_dice and not weighted_loss:
        model.compile(loss='categorical_crossentropy',
                            optimizer = Adam(lr=learning_rate),metrics=['accuracy'])
    if is_loss_soft_dice:                    
        model.compile(loss=soft_dice_loss,
                            optimizer = Adam(lr=learning_rate),metrics=['accuracy'])
    
    if weighted_loss:
        model.compile(loss=weighted_categorical_crossentropy(weights),
                            optimizer = Adam(lr=learning_rate),metrics=['accuracy'])
    
    #generating train and evaluation data
    train_gen = data_gen(dir_flow_train_imgs,dir_flow_train_labels, batch_size =  n_batch,
                         input_height=input_height, input_width=input_width,n_classes=n_classes  )
    val_gen = data_gen(dir_flow_eval_imgs,dir_flow_eval_labels, batch_size =  n_batch,
                         input_height=input_height, input_width=input_width,n_classes=n_classes  )
    
    for i in tqdm(range(index_start, n_epochs+index_start)):
        model.fit_generator(
            train_gen,
            steps_per_epoch=int(len(os.listdir(dir_flow_train_imgs))/n_batch)-1,
            validation_data=val_gen,
            validation_steps=1,
            epochs=1)
        model.save(dir_output+'/'+'model_'+str(i)+'.h5')
    

    #os.system('rm -rf '+dir_train_flowing)
    #os.system('rm -rf '+dir_eval_flowing)

    #model.save(dir_output+'/'+'model'+'.h5')

    
