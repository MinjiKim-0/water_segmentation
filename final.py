import tensorflow as tf
from tensorflow import keras

print('TensorFlow {}; Keras {}'.format(tf.__version__, keras.__version__))

from keras_unet_collection import models
import keras_unet_collection.losses
from sklearn.model_selection import train_test_split
# No module named 'PIL' -> pip install pillow

# import matplotlib.pyplot as plt

import datetime


# # 텐서 생성
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)

# print(c)
# # Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0


######## DES ######## 


# 내가 하고싶은거는 10개의 Unet like model, 3개(original function)의 activation function(GELU, snake), 9개(original 포함)의 loss funcion의 조합을 모두 찾아보는 것이다.
# 10*3*9 = 270개의 경우의 수..ㅋㅋ
# 거기다가 augmentation 된 데이터 안 된 데이터 차이 찾아야지!

# U-net, V-net, U-net++, R2U-Net, Attention U-net, 
# ResUnet-a, U^2-Net, UNET 3+, TransUNET, Swin-UNET

# GELU, Snake

# Dice loss, Tversky loss, Focal Tversky loss, Multi-scale Structural Similarity Index loss, iou_seg, 
# Intersection over Union (IoU) loss for segmentation, (Generalized) IoU loss for object detection
# Semi-hard triplet loss (experimental), CRPS loss (experimental)



# 그리고 train, validation, test 이미지 세트 제대로 해야 함.

# train 에서 train이랑 validationd으로 쪼개고, validation을 테스트 이미지로 가야 함


####################

seed = 64
batch_size = 7203 #7203 #26411
# batch_size_val = 300 #900 #300

test_size=round(batch_size/3)
train_size=batch_size-test_size
tf.random.set_seed(seed)



path_aug_x = "/water_segmentation/Training/[원천]train_water_data/aug_train"
# /media/visbic/MGTEC/water_seg/water_segmentation/Training/[원천]train_water_data/aug_train
# /water_segmentation/Training/[원천]train_water_data/aug_train

path_aug_y = "/water_segmentation/Training/[라벨]train_water_labeling/aug_train_label"
# /media/visbic/MGTEC/water_seg/water_segmentation/Training/[라벨]train_water_labeling/aug_train_label
# /water_segmentation/Training/[라벨]train_water_labeling/aug_train_label

# path_test_x = '/water_segmentation/Validation/[원천]validate_water_data/val300'
# # /water_segmentation/Validation/[원천]validate_water_data/aug_val

# path_test_y = '/water_segmentation/Validation/[라벨]validate_water_labeling/val_label300'
# # /water_segmentation/Validation/[라벨]validate_water_labeling/aug_val_label


# 이미지 전처리로 스케일링을 합니다
img_datagen = keras.preprocessing.image.ImageDataGenerator( rescale = 1.0/255. )

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# X
X_dir = path_aug_x
X_generator = img_datagen.flow_from_directory(X_dir, batch_size=batch_size, shuffle=False, seed=seed, class_mode=None, color_mode="grayscale")#"grayscale"

X = X_generator.next()
print("X.shape :", X.shape)
# X = X.astype(np.uint8)


# # x_test
# x_test_dir = path_test_x 
# x_test_generator = img_datagen.flow_from_directory(x_test_dir, batch_size=batch_size_test, shuffle=False, seed=seed, class_mode=None, color_mode="grayscale")#"grayscale"

# x_test = x_test_generator.next()
# print("x_test.shape :", x_test.shape)
# # x_test = x_test.astype(np.uint8)


# Y
Y_dir = path_aug_y
Y_generator = img_datagen.flow_from_directory(Y_dir, batch_size=batch_size, shuffle=False, seed=seed, class_mode=None, color_mode="grayscale")#"grayscale"

Y = Y_generator.next()
print("Y.shape :", Y.shape)
# print(Y.max())# -> 1
# print(Y.min())# -> 0
# print(Y.mean()) #-> 0.19562566
# print(np.median(Y)) #->0
# 0-> 흑 / 백->1


# Y[Y >= 0.5] = 1
# Y[Y < 0.5] = 0

x_train, x_val, y_train, y_val = train_test_split(X, Y, train_size=train_size, test_size=test_size, random_state=seed)

ohy_train = tf.one_hot(
    y_train, 2, on_value=None, off_value=None, axis=None, dtype=None, name=None
)
print("resahpe 전 ohy_train.shape :", ohy_train.shape)
ohy_train = ohy_train.reshape(ohy_train.shape[0], 256, 256, 2)
print("resahpe 후 ohy_train.shape :", ohy_train.shape)
# print(ohy_train.max())
# print(ohy_train.min())

ohy_val = tf.one_hot(
    y_val, 2, on_value=None, off_value=None, axis=None, dtype=None, name=None
)
print("resahpe 전 ohy_val.shape :", ohy_val.shape)
ohy_val = ohy_val.reshape(ohy_val.shape[0], 256, 256, 2)
print("resahpe 후 ohy_val.shape :", ohy_val.shape)


# # y_test
# y_test_dir = path_test_y
# y_test_generator = img_datagen.flow_from_directory(y_test_dir, batch_size=batch_size_test, shuffle=False, seed=seed, class_mode=None, color_mode="grayscale")#"grayscale"

# y_test = y_test_generator.next()
# print("y_test.shape :", y_test.shape)

# # y_test[y_test >= 0.5] = 1
# # y_test[y_test < 0.5] = 0



###########################



model = models.unet_2d((256, 256, 1), [64, 128, 256, 512, 1024], n_labels=2,
                      stack_num_down=2, stack_num_up=1,
                      activation='GELU', output_activation='Softmax', 
                      batch_norm=True, pool='max', unpool='nearest', name='unet')

model.compile(optimizer="adam", loss=[keras_unet_collection.losses.iou_seg], metrics=['accuracy'])

data_path = '/water_segmentation/unet_model/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
modelpath = data_path + '/{epoch:02d}-{val_accuracy:.4f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath=modelpath, monitor='val_accuracy', verbose=1)

# log_dir = "logs_unet/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0) 
# histogram_freq=0 -> metric에 dice_coef까지 쓴 후 killed 문제 때문에 0으로 바꿈

model.fit(x_train, ohy_train, batch_size=8, epochs=100, verbose=1, validation_data=(x_val, ohy_val)) # , callbacks=[checkpointer]
# , tensorboard_callback