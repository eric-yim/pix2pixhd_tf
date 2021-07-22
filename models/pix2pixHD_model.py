import tensorflow as tf
from models import networks
class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear decay after niter
    """
    def __init__(self, learning_rate,niter=100,niter_decay=100,steps_per_epoch=12):
        self.initial_learning_rate = tf.constant(learning_rate,dtype=tf.float32)
        self.niter = tf.constant(niter*steps_per_epoch,dtype=tf.float32)
        self.niter_decay=tf.constant(niter_decay*steps_per_epoch,dtype=tf.float32)
    def __call__(self, step):
        return tf.cond(tf.math.less(step,self.niter),lambda: self.initial_learning_rate,lambda: tf.maximum(self.initial_learning_rate* (1- ((step+1 - self.niter)/self.niter_decay)),0.0))



class Pix2PixHDModel:

    def __init__(self,args,length_of_data=28000):
        self.args = args
        self.is_train = args.is_train
        self.gan_feat_loss = args.train.discriminator_loss.gan_feat_loss
        self.n_layers_D = args.train.discriminator_loss.n_layers_D
        self.num_D = args.train.discriminator_loss.num_D
        self.lambda_feat = args.train.discriminator_loss.lambda_feat
        self.vgg_loss = args.train.use_vgg_loss
    
        # Definite Networks
        # Generator
        self.netG = networks.define_G(**args.train.generator)
        
        # Disciminator
        self.netD = networks.define_D(**args.train.discriminator)

        # Load Checkpoints
        # TODO (done in main.py)

        # Losses
        self.criterionGAN = networks.GANLoss(use_lsgan= args.train.use_lsgan)
        self.criterionFeat = tf.keras.losses.MeanAbsoluteError()
        if self.vgg_loss:
            self.criterionVGG = networks.VGGLoss(**args.train.vgg_loss)
        self.loss_names = ['G_GAN','G_GAN_Feat','G_VGG','D_real','D_fake']

        # Schduler
        scheduler_G = MyLRSchedule(
            learning_rate = args.train.hyperparameters.lr,
            niter=args.train.epochs,
            niter_decay=args.train.epochs_decay,
            steps_per_epoch=length_of_data)
        scheduler_D = MyLRSchedule(
            learning_rate = args.train.hyperparameters.lr,
            niter=args.train.epochs,
            niter_decay=args.train.epochs_decay,
            steps_per_epoch=length_of_data)

        # Optimizers
        self.optimizer_G = tf.keras.optimizers.Adam(learning_rate=scheduler_G,beta_1 = args.train.hyperparameters.beta_1)
        self.optimizer_D = tf.keras.optimizers.Adam(learning_rate = scheduler_D,beta_1 = args.train.hyperparameters.beta_1)

    def call(self,label,real_image):
        """
        Label is source,
        Image is target
        """

        # Fake Generation
        fake_image = self.netG.call(label)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(label,fake_image)
        loss_D_fake = self.criterionGAN(pred_fake_pool,False)

        # Real Detection and Loss
        pred_real = self.discriminate(label,real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.call(tf.concat((label, fake_image), axis=-1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0.0#tf.constant(0,dtype='float32')
        if self.gan_feat_loss:
            feat_weights = 4.0 / (self.n_layers_D + 1)
            D_weights = 1.0 / self.num_D
            for i in range(self.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(tf.stop_gradient(pred_real[i][j]),pred_fake[i][j] ) * self.lambda_feat
        # VGG feature matching loss
        loss_G_VGG = 0
        if self.vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.lambda_feat

        return [loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake], fake_image
    def discriminate(self,label,image):
        input_concat = tf.concat((label,tf.stop_gradient(image)),axis=-1)
        return self.netD.call(input_concat)