import numpy as np
import matplotlib.pyplot as plt
# from data import x_val, y_val
import tensorflow as tf
from tensorflow import keras
# from keras_unet_collection.models import unet_2d

seed = 64
batch_size = 32
tf.random.set_seed(seed)



###########


# 이미지 전처리로 스케일링을 합니다
img_datagen = keras.preprocessing.image.ImageDataGenerator( rescale = 1.0/255. )



# x_val
x_val_dir = '/water_segmentation/Validation/[원천]validate_water_data'
x_val_generator = img_datagen.flow_from_directory(x_val_dir, batch_size=300, shuffle=False, seed=seed, class_mode=None, color_mode="grayscale")#"grayscale"

x_val = x_val_generator.next()
print("x_val.shape :", x_val.shape)



from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# y_val
y_val_dir = '/water_segmentation/Validation/[라벨]validate_water_labeling'
y_val_generator = img_datagen.flow_from_directory(y_val_dir, batch_size=300, shuffle=False, seed=seed, class_mode=None, color_mode="grayscale")#"grayscale"

y_val = y_val_generator.next()
print("y_val.shape :", y_val.shape)

ohy_val = tf.one_hot(
    y_val, 2, on_value=None, off_value=None, axis=None, dtype=None, name=None
)
print("resahpe 전 ohy_val.shape :", ohy_val.shape)
ohy_val = ohy_val.reshape(ohy_val.shape[0], 256, 256, 2)
print("resahpe 후 ohy_val.shape :", ohy_val.shape)


###########


from keras_unet_collection.activations import GELU
from keras_unet_collection.losses import iou_seg

# /media/visbic/MGTEC/water_seg/water_segmentation/unet_model/20211211-040154/10-0.9812.hdf5
model = keras.models.load_model('/water_segmentation/unet_model/20211211-040154/10-0.9812.hdf5', 
                                custom_objects={
                                                "GELU":GELU,
                                                "iou_seg":iou_seg
                                                })

pred = model.predict(x_val)
print(pred.shape)

pred2 = np.zeros((300,256,256,1))

print((pred[:,:,:,0] > pred[:,:,:,1]).sum())
print((pred[:,:,:,0] < pred[:,:,:,0]).sum())

pred2 = pred[:,:,:,0]
print(pred2.shape)

pred2[pred[:,:,:,0] > 0.5] = 0
pred2[pred[:,:,:,0] < 0.5] = 255

def showPred(idx):
    # 본래 이미지
    plt.subplot(131)
    plt.imshow(x_val[idx].reshape(256,256), cmap='gray')
    plt.title("predicted")

    # ground truth
    plt.subplot(132)
    plt.imshow(y_val[idx].reshape(256,256), cmap='gray')
    plt.title("label")
    
    # 예측
    plt.subplot(133)
    plt.imshow(pred2[idx].reshape(256,256), cmap='gray')
    plt.title("predicted")

    plt.show()


for i in range(x_val.shape[0]):
    showPred(i)