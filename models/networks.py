import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
def init_weights(model):
    init0 = tf.random_normal_initializer(
        mean=0.0, stddev=0.02, seed=None
    )
    init1 = tf.random_normal_initializer(
        mean=1.0, stddev=0.02, seed=None
    )
    init2 = tf.keras.initializers.Zeros()
    for weights in model.weights:
    
        if (weights.name.find('conv')!= -1) :
            weights.assign( init0(shape=weights.shape,dtype=weights.dtype) )
        elif (weights.name.find('batch_norm')!= -1) and (weights.name.find('gamma')!= -1):
            weights.assign(init1(shape=weights.shape,dtype=weights.dtype))
        elif (weights.name.find('batch_norm')!= -1) and (weights.name.find('beta')!= -1):
            weights.assign(init2(shape=weights.shape,dtype=weights.dtype))
###########################
###LOSSES
###########################

class Vgg19(tf.keras.Model):
    def __init__(self, input_shape,requires_grad=False,mixed_precision=False):
        super(Vgg19, self).__init__()
        m_tf = tf.keras.applications.vgg19.VGG19(include_top=False,input_shape=input_shape)
        layer_ids = [2,5,8,13,18]
        if mixed_precision:
            base_model_outputs = [layers.Activation('linear',dtype='float32')(m_tf.layers[id].output) for id in layer_ids]
        else:
            base_model_outputs = [m_tf.layers[id].output for id in layer_ids]
        self.model =  tf.keras.Model(inputs=m_tf.input,outputs = base_model_outputs)
        self.model.trainable=False
        
    def call(self, x):
        return self.model(x)
class VGGLoss(tf.keras.losses.Loss):
    def __init__(self,input_shape=[512,512,3]):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19(input_shape=input_shape)
        self.criterion = tf.keras.losses.MeanAbsoluteError()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def call(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(tf.stop_gradient(y_vgg[i]), x_vgg[i])
        return loss
class GANLoss(tf.keras.losses.Loss):
    def __init__(self, use_lsgan=True):
        """
        Use_lsgan not implemented yet
        """
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = tf.keras.losses.MeanSquaredError()
        else:
            self.loss = tf.keras.losses.BinaryCrossentropy()

    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for input_i in input:
                target_tensor = tf.ones_like(input_i) if target_is_real else tf.zeros_like(input_i)
                loss += self.loss(target_tensor, input_i)
            return loss
        else:
            target_tensor = tf.ones_like(input) if target_is_real else tf.zeros_like(input)
            return self.loss(target_tensor,input)

###########################
###Nets
###########################
def define_G(image_size,input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(image_size,input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':
        netG = LocalEnhancer(image_size,input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(image_size,input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')

    init_weights(netG)
    return netG
def define_D(image_size,input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D)
    dummy = tf.random.uniform(shape=(1,image_size,image_size,input_nc),dtype=tf.float32)
    netD(dummy)
    init_weights(netD)
    return netD
def get_norm_layer(norm_type='instance'):
    import tensorflow_addons as tfa
    if norm_type == 'batch':
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm_type == 'instance':
        norm_layer = tfa.layers.InstanceNormalization
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
class LocalEnhancer(tf.keras.Model):
    ### Only n_local_enhancer==1 available
    def __init__(self,image_size=512,input_nc=3, output_nc=3,ngf=32,n_downsample_global=3,n_blocks_global=9,n_blocks_local=3,norm_layer=tf.keras.layers.BatchNormalization,mixed_precision=False):
        super(LocalEnhancer,self).__init__()

        activation = layers.ReLU
        paddings = tf.constant([[0,0],[3,3],[3,3],[0,0]]) #nn.ReflectionPad2d(3)

        input_0 = layers.Input(shape=(image_size,image_size,input_nc))
        ### downsample
        ngf_global = ngf * (2**1)
        self.downsample = layers.AveragePooling2D(pool_size=(3,3),strides=(2,2),padding='same')
        x = self.downsample(input_0)
        self.global_generator = GlobalGenerator(image_size//2,input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model_trunk
        x = self.global_generator(x)

        ### downsample
        ngf_global = ngf * (2**0)
        y = tf.pad(input_0,paddings,mode='REFLECT')
        y = layers.Conv2D(ngf_global,kernel_size=7,padding='valid')(y)
        y = norm_layer()(y)
        y = activation()(y)
        y = layers.Conv2D(ngf_global*2,kernel_size=3,strides=2,padding='same')(y)
        y = norm_layer()(y)
        y = activation()(y)

        input_1 = layers.Input(shape=(image_size//2,image_size//2,ngf_global*2))
        ### resblocks
        z = ResnetBlock(ngf_global*2,norm_layer=norm_layer,activation=activation)(input_1) #merge with result from global generator

        ### upsample
        z = layers.Conv2DTranspose(ngf_global,kernel_size=3,strides=2,padding='same')(z)
        z=norm_layer()(z)
        z= activation()(z)

        ### Final
        z = tf.pad(z,paddings,mode='REFLECT')
        if mixed_precision:
            last_layer = layers.Conv2D(output_nc,kernel_size=7,padding='valid',activation='tanh',dtype='float32')
        else:
            last_layer = layers.Conv2D(output_nc,kernel_size=7,padding='valid',activation='tanh')
        z = last_layer(z)

        self.model_down = tf.keras.Model(inputs=input_0,outputs=[x,y])
        self.model_up = tf.keras.Model(inputs=input_1,outputs=z)
    def call(self,x):
        out0,out1 = self.model_down(x)
        return self.model_up(out0+out1)
class GlobalGenerator(tf.keras.Model):#512
    def __init__(self,image_size=512,input_nc=3,output_nc=3,ngf=64,n_downsampling=3,n_blocks=9,norm_layer=tf.keras.layers.BatchNormalization,mixed_precision=False):
        super(GlobalGenerator,self).__init__()
        activation = layers.ReLU
        paddings = tf.constant([[0,0],[3,3],[3,3],[0,0]]) #nn.ReflectionPad2d(3)
        
        input_ = layers.Input(shape=(image_size,image_size,input_nc))
        x = tf.pad(input_,paddings,mode='REFLECT')
        x = layers.Conv2D(filters=ngf,kernel_size=7,padding='valid')(x)
        x = norm_layer()(x)
        x = activation()(x)
        
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            x = layers.Conv2D(filters=ngf*mult*2,kernel_size=3,strides=2,padding='same')(x)
            x = norm_layer()(x)
            x = activation()(x)
        
        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            x = ResnetBlock(ngf*mult,activation=activation,norm_layer=norm_layer)(x)
        
        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            x = layers.Conv2DTranspose(int(ngf*mult/2),kernel_size=3,strides=2,padding='same')(x)
            x = norm_layer()(x)
            x = activation()(x)
        
        
        y = tf.pad(x,paddings,mode='REFLECT')
        
        if mixed_precision:
            last_layer = layers.Conv2D(output_nc,kernel_size=7,padding='valid',activation='tanh',dtype='float32')
        else:
            last_layer = layers.Conv2D(output_nc,kernel_size=7,padding='valid',activation='tanh')
        y = last_layer(y)
        self.model_trunk = tf.keras.Model(inputs=input_,outputs=x)
        self.model = tf.keras.Model(inputs=input_,outputs=y)
        
    def call(self,x):
        return self.model(x)


class ResnetBlock(tf.keras.Model):
    def __init__(self, dim, norm_layer, activation=None, use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.build_conv_block(dim, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim,  norm_layer, activation, use_dropout):
        conv_block = []
        #Original gives option of reflect,replicate, or 1
        self.paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]]) #nn.ReflectionPad2d(3)
        self.norm_layer0=norm_layer()
        self.activation=activation()
        self.conv0 = layers.Conv2D(dim,kernel_size=3,padding='valid')
        self.use_dropout=use_dropout
        self.conv1 = layers.Conv2D(dim,kernel_size=3,padding='valid')
        self.norm_layer1 = norm_layer()

    def call(self, x):
        out = self.conv0(x)
        out = tf.pad(out,self.paddings,mode='REFLECT')
        out = self.norm_layer0(out)
        out = self.activation(out)
        if self.use_dropout:
            out=layers.Dropout(0.5)(out)
        out = self.conv1(out)
        out = tf.pad(out,self.paddings,mode='REFLECT')
        out = self.norm_layer1(out)
        out = x + out
        return out

class Encoder(tf.keras.Model):
    def __init__(self, image_size,input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=tf.keras.layers.BatchNormalization,mixed_precision=False):
        super(Encoder, self).__init__()
        activation = tf.layers.ReLU
        self.output_nc = output_nc

        paddings = tf.constant([[0,0],[3,3],[3,3],[0,0]]) #nn.ReflectionPad2d(3)
        
        input_ = layers.Input(shape=(image_size,image_size,input_nc))
        x = tf.pad(input_,paddings,mode='REFLECT')
        x = layers.Conv2D(ngf,kernel_size=7,padding='valid')(x)
        x = norm_layer()(x)
        x = activation()(x)

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            x = layers.Conv2d(ngf*mult*2,kernel_size=3,strides=2,padding='same')(x)
            x = norm_layer()(x)
            x = activation()(x)

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            x = layers.Conv2DTranspose(int(ngf*mult/2),kernel_size=3,strides=2,padding='same')(x)
            x = norm_layer()(x)
            x = activation()(x)

        x = tf.pad(x,paddings,mode='REFLECT')
       
        if mixed_precision:
            last_layer = layers.Conv2D(output_nc,kernel_size=7,padding='valid',activation='tanh',dtype='float32')
        else:
            last_layer = layers.Conv2D(output_nc,kernel_size=7,padding='valid',activation='tanh')
        x = last_layer(x)
        self.model = tf.keras.Model(inputs=input_,outputs=x)


    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = tf.identity(outputs)#.clone()
        inst_list = np.unique(inst.numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]
                    mean_feat = tf.broadcast_to(tf.reduce_mean(output_ins),output_ins.shape)
                    #mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat
        return outputs_mean
class MultiscaleDiscriminator(tf.keras.Model):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=tf.keras.layers.BatchNormalization,
                 use_sigmoid=False, num_D=3,mixed_precision=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid,mixed_precision=mixed_precision)
            setattr(self, 'layer'+str(i), netD.model)
        self.downsample = layers.AveragePooling2D(pool_size=(3,3),strides=(2,2),padding='same')
    def singleD_forward(self,model,input):
        return model(input)
    def call(self,input):
        result = []
        input_downsampled = input
        for i in range(self.num_D):
            model = getattr(self, 'layer'+str(self.num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (self.num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class NLayerDiscriminator(tf.keras.Model):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=tf.keras.layers.BatchNormalization, use_sigmoid=False,mixed_precision=False):
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers
        kw = 4
        my_layers = []
        my_layers += [
            layers.Conv2D(ndf,kernel_size=kw,strides=2,padding='same'),
            layers.LeakyReLU(alpha=0.2)
        ]

        nf = ndf
        for n in range(1,n_layers):
            nf = min(nf*2,512)
            my_layers += [
                layers.Conv2D(nf,kernel_size=kw,strides = 2,padding='same'),
                norm_layer(),
                layers.LeakyReLU(alpha=0.2)
            ]

        nf = min(nf*2,512)
        my_layers += [
            layers.Conv2D(nf,kernel_size=kw,strides=1,padding='same'),
            norm_layer(),
            layers.LeakyReLU(alpha=0.2)
        ]
        my_layers+= [layers.Conv2D(1,kernel_size=kw,strides=1,padding='same')]
        if use_sigmoid:
            my_layers += [layers.Activation(tf.keras.activations.sigmoid)]
        if mixed_precision:
            my_layers += [layers.Activation('linear',dtype='float32')]
        self.model = Sequential(my_layers)
    def call(self,input):
        return self.model(input)