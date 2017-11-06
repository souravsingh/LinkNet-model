import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
slim = tf.contrib.slim

'''
LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
Based on the paper: https://arxiv.org/pdf/1707.03718.pdf
'''

@slim.add_arg_scope
def convBnRelu(input, num_channel, kernel_size, stride, is_training, scope, padding = 'SAME'):
    x = slim.conv2d(input, num_channel, [kernel_size, kernel_size], stride=stride, activation_fn=None, scope=scope+'_conv1', padding = padding)
    x = slim.batch_norm(x, is_training=is_training, fused=True, scope=scope+'_batchnorm1')
    x = tf.nn.relu(x, name=scope+'_relu1')
    return x


@slim.add_arg_scope
def deconvBnRelu(input, num_channel, kernel_size, stride, is_training, scope, padding = 'VALID'):
    x = slim.conv2d_transpose(input, num_channel, [kernel_size, kernel_size], stride=stride, activation_fn=None, scope=scope+'_fullconv1', padding = padding)
    x = slim.batch_norm(x, is_training=is_training, fused=True, scope=scope+'_batchnorm1')
    x = tf.nn.relu(x, name=scope+'_relu1')
    return x    

@slim.add_arg_scope
def initial_block(inputs, is_training=True, scope='initial_block'):
    '''
    The initial block for Linknet has 2 branches: The convolution branch and Maxpool branch.
    INPUTS:
    - inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    OUTPUTS:
    - net_concatenated(Tensor): a 4D Tensor that contains the 
    '''
    #Convolutional branch
    net_conv = slim.conv2d(inputs, 64, [7,7], stride=2, activation_fn=None, scope=scope+'_conv')
    net_conv = slim.batch_norm(net_conv, is_training=is_training, fused=True, scope=scope+'_batchnorm')
    net_conv = tf.nn.relu(net_conv, name=scope+'_relu')

    #Max pool branch
    net_pool = slim.max_pool2d(net_conv, [3,3], stride=2, scope=scope+'_max_pool')

    return net_conv

@slim.add_arg_scope
def residualBlock(input, n_filters, is_training, stride=1, downsample= None, scope='residualBlock'):
    # Shortcut connection

    # Downsample the data or just pass original
    if downsample == None:
        shortcut = input
    else:
        shortcut = downsample

    # Residual
    x = convBnRelu(input, n_filters, kernel_size = 3, stride = stride, is_training = is_training, scope = scope + '/cvbnrelu')
    x = slim.conv2d(x, n_filters, [3,3], stride=1, activation_fn=None, scope=scope+'_conv2',  padding = 'SAME')
    x = slim.batch_norm(x, is_training=is_training, fused=True, scope=scope+'_batchnorm2')
    
    # Shortcutr connection
    x = x + shortcut
    x = tf.nn.relu(x, name=scope+'_relu2')
    return x


@slim.add_arg_scope
def encoder(inputs, inplanes, planes, blocks, stride, is_training=True, scope='encoder'):
    '''
    Decoder of LinkNet
    INPUTS:
    - inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    OUTPUTS:
    - net_concatenated(Tensor): a 4D Tensor that contains the 
    '''
    
    # make downsample at skip connection if needed
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample   = slim.conv2d(inputs, planes, [1,1], stride=stride, activation_fn=None, scope=scope+'_conv_downsample')
        downsample   = slim.batch_norm(downsample, is_training=is_training, fused=True, scope=scope+'_batchnorm_downsample')

    # Create mupliple block of ResNet
    output = residualBlock(inputs, planes, is_training, stride, downsample, scope = scope +'/residualBlock0')
    for i in range(1, blocks):
        output = residualBlock(output, planes, is_training, 1, scope = scope +'/residualBlock{}'.format(i))


    return output

@slim.add_arg_scope
def decoder(inputs, n_filters, planes, is_training=True, scope='decoder'):
    '''
    Encoder use ResNet block. As in paper, we  will use ResNet18 block for learning.
    INPUTS:
    - inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    OUTPUTS:
    - net_concatenated(Tensor): a 4D Tensor that contains the 
    '''
    
    x = convBnRelu(inputs, n_filters/2, kernel_size = 1, stride = 1, is_training = is_training,  padding = 'SAME', scope = scope + "/c1")
    x = deconvBnRelu(x, n_filters/2, kernel_size = 3, stride = 2, is_training = is_training, padding = 'SAME', scope = scope+ "/dc1")
    x = convBnRelu(x, planes, kernel_size = 1, stride = 1, is_training = is_training,  padding = 'SAME', scope = scope+ "/c2")

    return  x    


#Now actually start building the network
def LinkNet(inputs,
         num_classes,
         reuse=None,
         is_training=True,
         feature_scale=4,
         scope='LinkNet'):
    #Set the shape of the inputs first to get the batch_size information
    inputs_shape = inputs.get_shape().as_list()
    # inputs.set_shape(shape=(batch_size, inputs_shape[1], inputs_shape[2], inputs_shape[3]))

    layers = [2, 2, 2, 2]
    filters = [64, 128, 256, 512]
    filters = [x / feature_scale for x in filters]

    with tf.variable_scope(scope, reuse=reuse):
        #Set the primary arg scopes. Fused batch_norm is faster than normal batch norm.
        with slim.arg_scope([initial_block, encoder], is_training=is_training),\
             slim.arg_scope([slim.batch_norm], fused=True), \
             slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=None): 
            #=================INITIAL BLOCK=================
           
            net = initial_block(inputs, scope='initial_block')

            enc1 = encoder(net,  64, filters[0], layers[0], stride=1, is_training=is_training, scope='encoder1')
            enc2 = encoder(enc1, filters[0], filters[1], layers[1], stride=2, is_training=is_training, scope='encoder2')
            enc3 = encoder(enc2, filters[1], filters[2], layers[2], stride=2, is_training=is_training, scope='encoder3')
            enc4 = encoder(enc3, filters[2], filters[3], layers[3], stride=2, is_training=is_training, scope='encoder4')

           
            decoder4 = decoder(enc4, filters[3], filters[2], is_training=is_training, scope='decoder4')
            decoder4 += enc3
            decoder3 = decoder(decoder4, filters[2], filters[1], is_training=is_training, scope='decoder3')
            decoder3 += enc2
            decoder2 = decoder(decoder3, filters[1], filters[0], is_training=is_training, scope='decoder2')
            decoder2 += enc1
            decoder1 = decoder(decoder2, filters[0], filters[0], is_training=is_training, scope='decoder1')

            f1     = deconvBnRelu(decoder1, 32/feature_scale, 3, stride = 2, is_training=is_training, scope='f1',padding = 'SAME')
            f2     = convBnRelu(f1, 32/feature_scale, 3, stride = 1, is_training=is_training, padding = 'SAME', scope='f2')
            logits = slim.conv2d(f2, num_classes, [2,2], stride=2, activation_fn=None, padding = 'SAME', scope='logits')

        return logits


def LinkNet_arg(weight_decay=2e-4,
                   batch_norm_decay=0.1,
                   batch_norm_epsilon=0.001):
  with slim.arg_scope([slim.conv2d],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    # Set parameters for batch_norm.
    with slim.arg_scope([slim.batch_norm],
                        decay=batch_norm_decay,
                        epsilon=batch_norm_epsilon) as scope:
      return scope
