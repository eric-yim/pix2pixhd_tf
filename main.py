from omegaconf import OmegaConf
import argparse
from models.pix2pixHD_model import Pix2PixHDModel
from utils.data_utils import get_listings,load_image_train,load_image_val,check_folder
from utils.print_utils import AverageMeter
from utils.eval_utils import save_image,evaluate
import tensorflow as tf

import os
import logging
from tensorflow.keras import mixed_precision
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='arg_files/args_256.yaml', help='')
    config_path =parser.parse_args()

    args = OmegaConf.load(config_path.config_path)
    return args   

def get_datasets(args):
        """
        Loads Data
        """
        train_listing, val_listing = get_listings(args)
        logging.info(f"Found Files: Train {len(train_listing)} Val {len(val_listing)}")
        train_dataset = tf.data.Dataset.from_tensor_slices(train_listing)

        train_dataset = train_dataset.map(lambda x: load_image_train(x,args.transforms.resize_to,args.transforms.resize_to_large), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(args.train.buffer_size).batch(args.train.batch_size)
        #train_dataset = train_dataset.apply(tf.python.data.experimental.prefetch_to_device(gpu_device, buffer_size=tf.python.data.experimental.AUTOTUNE))
        train_dataset = train_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices(val_listing)
        val_dataset = val_dataset.map(lambda x: load_image_val(x,args.transforms.resize_to), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.shuffle(args.train.buffer_size).batch(args.train.batch_size)
        val_dataset = val_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
        return train_dataset,val_dataset

@tf.function
def train_step(model,data):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        losses, generated = model.call(*data)
        loss_dict = dict(zip(model.loss_names, losses))
        
        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG']
    gen_grads = gen_tape.gradient(loss_G,model.netG.trainable_variables)
    disc_grads = disc_tape.gradient(loss_D,model.netD.trainable_variables)
    model.optimizer_G.apply_gradients(zip(gen_grads,model.netG.trainable_variables))
    model.optimizer_D.apply_gradients(zip(disc_grads,model.netD.trainable_variables))
    return loss_D,loss_G,generated
def setup_procedures(args):
    check_folder(args.save_paths.sample_dir)
    check_folder(args.save_paths.checkpoint_dir)




def main(args):
    sample_dir=args.save_paths.sample_dir
    sample_format = args.save_paths.sample_format
    setup_procedures(args)
    #Measure Losses
    gen = AverageMeter()
    disc = AverageMeter()
    train_dataset,val_dataset = get_datasets(args)

    pix2pix = Pix2PixHDModel(args,length_of_data=len(train_dataset))

    checkpoint_prefix= os.path.join(args.save_paths.checkpoint_dir,"ckpt")
    checkpoint = tf.train.Checkpoint(
            optimizer_G=pix2pix.optimizer_G,
            optimizer_D=pix2pix.optimizer_D,
            netG=pix2pix.netG,
            netD=pix2pix.netD
            )
    if (not args.is_train) or args.train.resume_latest:
        chk_path = tf.train.latest_checkpoint(args.save_paths.checkpoint_dir)
        
        logging.info(f"Restoring checkpoint {chk_path}...")
        checkpoint.restore(chk_path)
    #Evaluation: save a saved model format. and create a sample on val dataset
    if not args.is_train:
        
        evaluate(pix2pix,val_dataset,sample_dir,sample_format)
        pix2pix.netG.model.save(args.save_paths.saved_model)
        print(f"Model saved to {args.save_paths.saved_model}")
        return

    for epoch in range(args.train.start_epoch,args.train.epochs + args.train.epochs_decay):
        gen.reset()
        disc.reset()
        for i,data in enumerate(train_dataset):
            loss_D,loss_G,generated=train_step(pix2pix,data)
            loss_D = loss_D.numpy()
            loss_G = loss_G.numpy()
            generated = generated.numpy()
            b = len(data[0])
            gen.update(loss_G,b)
            disc.update(loss_D,b)
            if (i+1)%args.train.print_freq==0:
                logging.info(f"Epoch: {epoch} ({i+1}/{len(train_dataset)}) || Gen Loss {gen.avg:0.3f} || Disc Loss {disc.avg:0.3f}")
                #import IPython ; IPython.embed() ; exit(1)
                save_image(sample_dir,sample_format,epoch,generated)
        if (epoch+1)%args.train.save_freq == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            logging.info(f"Checkpoint saved to {checkpoint_prefix}")
if __name__ == '__main__':
    args = get_args()
    #mixed_precision.set_global_policy('mixed_float16')
    #tf 2.3 mixed_precision.experimental.set_policy('mixed_float16')
    logging.basicConfig(filename=args.logging.path,format=args.logging.format, level=logging.INFO,datefmt="%Y%m%d %H:%M:%S")
    main(args)