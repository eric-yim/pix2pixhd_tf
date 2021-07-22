import glob
import tensorflow as tf
import os
import numpy as np
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def get_listing(globs):
    """
    Takes in list of globs like some\path\to\*.jpg
    Returns sorted listing
    """
    listing = []
    for globber in globs:
        listing += glob.glob(globber)
    listing.sort()
    return listing
def get_listings(args):
    """
    Returns Train Listing, Val Listing
    Each item in a listing is a pair: [source file,target_file]
        Where target file is defined by target_dir + source_file_name
    """
    train_listing = get_listing(args.data.train_globs)
    val_listing = get_listing(args.data.val_globs)
    train_listing = [[item,os.path.join(args.data.train_target_dir,os.path.split(item)[1])] for item in train_listing]
    val_listing = [[item,os.path.join(args.data.val_target_dir,os.path.split(item)[1])] for item in val_listing]
    return train_listing,val_listing



def tensor2img(img):
    """
    From -1,1 back to 0,255
    """
    img = np.clip(np.round((img+1)*127.5),0,255).astype(np.uint8)
    return img

@tf.function
def random_jitter(img0, img1, resize_to=256,resize_to_large=286):
    # resizing to 286 x 286 x 3
    img0 = tf.image.resize(img0,(resize_to_large,resize_to_large))
    img1 = tf.image.resize(img1,(resize_to_large,resize_to_large))

    # randomly cropping to 256 x 256 x 3
    img0, img1 = random_crop(img0, img1,resize_to)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        img0 = tf.image.flip_left_right(img0)
        img1 = tf.image.flip_left_right(img1)

    return img0,img1

def load_image_train(items,resize_to,resize_to_large):
    """
    Loads 2 images, source and target
    """
    img0,img1 = load_image(items[0]),load_image(items[1])
    img0,img1 = random_jitter(img0,img1,resize_to,resize_to_large)
    return normalize(img0),normalize(img1)


def load_image_val(items,resize_to):
    img0,img1 = load_image(items[0]),load_image(items[1])
    img0 = tf.image.resize(img0,(resize_to,resize_to))
    img1 = tf.image.resize(img1,(resize_to,resize_to))
    return normalize(img0),normalize(img1)


def random_crop(input_image, real_image,crop_size=3):
    """
    Channel is hardcoded as 3
    """
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2,crop_size, crop_size,3])
    return cropped_image[0], cropped_image[1]

def load_image(item):
    """
    Load normalized image
    """
    # may need decode_png also.
    # decode_image does not work with image.resize because no shape
    # alternatively can resize in main code
    return tf.io.decode_jpeg(tf.io.read_file(item)) 
    
def normalize(input_image):
    """
    0,255 to -1,1
    """
    input_image = tf.cast(input_image, tf.float32) / 127.5 -1
    return input_image

