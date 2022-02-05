import numpy as np
import os
from keras import backend as K
from keras.preprocessing.image import load_img, save_img, img_to_array
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from scipy.optimize import fmin_l_bfgs_b
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from ArtProjectFunctions import preprocess_image, get_content_loss, get_style_loss, \
    deprocess_image, gram_matrix, Evaluator

def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

def preprocess_image_instantiator(image_path,img_nrows,img_ncols):
    from keras.applications import vgg19
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img



CurPath = os.getcwd()
base_image_path = 'E:\\OneDrive\\DataScience\\IST 652\\pythonProject\\base_images\\girlbosses-2.jpg'
style_image_path = 'E:\\OneDrive\\Pictures\\Wallpaper\\Fractal\\blood-moon-rising-gallery.jpg'
output = 'E:\\OneDrive\\DataScience\\IST 652\\pythonProject\\output\\1.jpg'
content_weight=0.025
style_weight=1.0
iterations = 5

# plot base image
#plt.figure()
#plt.title("Base Image",fontsize=20)
#img1 = load_img(base_image_path)
#plt.imshow(img1)

# plot style image
#plt.figure()
#plt.title("Style Image",fontsize=20)
#img1 = load_img(style_image_path)
#plt.imshow(img1)


# get dimensions of the generated picture
width, height = load_img(base_image_path).size
img_nrows = 846
img_ncols = int(width * img_nrows / height)

# get tensor representations of our images
# converts image to numpy array then feeds it to Keras bakend
# runs in gpu
base_image = K.variable(preprocess_image(base_image_path, img_nrows, img_ncols))
style_reference_image = K.variable(preprocess_image(style_image_path, img_nrows, img_ncols))

K.image_data_format()

# this will contain our generated image
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1,3,img_nrows, img_ncols))
else:
    combination_image = K.placeholder((1,img_nrows, img_ncols,3))

# combine the 3 images into a single Keras tensor
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image
                              ], axis=0)

# build the VGG19 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights
vgg19_weights = CurPath + '\\vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = VGG19(input_tensor=input_tensor,
              include_top = False,
              weights=vgg19_weights)
#model = vgg19.VGG19(input_tensor=input_tensor,
#                    weights='imagenet', include_top=False)
print('Model loaded.')

# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
print(outputs_dict['block5_conv2'])


# combine these loss functions into a single scalar
loss = K.variable(0.0)
layer_features = outputs_dict['block5_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
print('Layer Feature for Content Layers :: ' + str(layer_features))
print('Base Image Feature :: ' + str(base_image_features))
print('Combination Image Feature for Content Layers:: ' + str(combination_features)+'\n')
loss = loss + content_weight * get_content_loss(base_image_features, combination_features)

#  take 5 feature layers
feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    print('Layer Feature for Style Layers :: '+str(layer_features))
    print('Style Image Feature :: '+str(style_reference_features))
    print('Combination Image Feature for Style Layers:: '+str(combination_features)+'\n')
    sl = get_style_loss(style_reference_features, combination_features, img_nrows, img_ncols)
    loss += (style_weight / len(feature_layers)) * sl

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)
grads

outputs = [loss]
if isinstance(grads, (list,tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)
f_outputs


# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
x_opt = preprocess_image(base_image_path, img_nrows, img_ncols)

evaluator = Evaluator()


# Store our best result
best_loss, best_img = float('inf'), None
for i in range(iterations):
    print('Start of iteration', i)
    x_opt, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                        x_opt.flatten(),
                                        fprime=evaluator.grads,
                                        maxfun=20,
                                        disp=True,
                                       )
    print('Current loss value:', min_val)
    if min_val < best_loss:
        # Update best loss and best image from total loss.
        best_loss = min_val
        best_img = x_opt.copy()

imgx = deprocess_image(best_img.copy(), img_nrows, img_ncols)
plt.imshow(imgx)

save_img(path = output, x= imgx)

plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.title("Base Image",fontsize=20)
img_base = load_img(base_image_path)
plt.imshow(img_base)

plt.subplot(1,3,1+1)
plt.title("Style Image",fontsize=20)
img_style = load_img(style_image_path)
plt.imshow(img_style)

plt.subplot(1,3,1+2)
plt.title("Final Image",fontsize=20)
plt.imshow(imgx)



