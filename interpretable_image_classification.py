import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_formats = {'png', 'retina'}
plt.rcParams['figure.figsize'] = 8,8

from scipy.ndimage.interpolation import zoom
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

import numpy as np
import os
import interpretutils


if __name__ == '__main__':
    model = VGG16(include_top=True, weights='imagenet',input_shape=(224,224,3))

    img_paths = ["cat_dog.png","multiple_dogs.jpg","collies.JPG","snake.JPEG","water-bird.JPEG", "cat_dog.png"]

    for path in img_paths:
        savepath = os.path.join("results",path)
        img_path = os.path.join("image",path)
        orig_img = np.array(load_img(img_path,target_size=(224,224)),dtype=np.uint8)
        img = np.array(load_img(img_path,target_size=(224,224)),dtype=np.float64)
        img = np.expand_dims(img,axis=0)
        img = preprocess_input(img)
        
        predictions = model.predict(img)
        top_n = 5
        top = decode_predictions(predictions, top=top_n)[0]
        cls = np.argsort(predictions[0])[-top_n:][::-1]
        
        # cam, raw_cam = interpretutils.grad_cam(model, img, layer_name='block5_conv3')
        cam, raw_cam  = interpretutils.grad_cam_plus(model, img, layer_name='block5_conv3')
        
        print("Class activation map for:",top[0])
        
        fig, ax = plt.subplots(nrows=1,ncols=3)
        plt.subplot(131)
        plt.imshow(orig_img)
        plt.title("input image")
        
        plt.subplot(132)
        plt.imshow(raw_cam,cmap="jet")
        plt.title("heatmap result")
        
        plt.subplot(133)
        plt.imshow(orig_img)
        plt.imshow(cam,alpha=0.6,cmap="jet")
        plt.title("input with heatmap")
        
        plt.show()

        if savepath is not None:
            fig.savefig(savepath)
    print("Saved all results in './results'")