import tensorflow as tf
from tensorflow import keras

print('TensorFlow {}; Keras {}'.format(tf.__version__, keras.__version__))

from keras_unet_collection import models
import keras_unet_collection.losses
# import matplotlib.pyplot as plt

import datetime

from data import x_train, ohy_train, x_val, ohy_val

# tf.debugging.set_log_device_placement(True)

# # 텐서 생성
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)

# print(c)
# # Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0


##################### 

seed = 64
tf.random.set_seed(seed)


# model
model = models.unet_2d((256,256,1), [64, 128, 256, 512, 1024], n_labels=2,
                      stack_num_down=2, stack_num_up=1,
                      activation='GELU', output_activation='Softmax', 
                      batch_norm=True, pool='max', unpool='nearest', name='unet')

model.compile(optimizer='adam', loss=[keras_unet_collection.losses.iou_seg, keras_unet_collection.losses.dice_coef], metrics=['accuracy'])



data_path = '/water_segmentation/unet_model/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
modelpath = data_path + '/{epoch:02d}-{val_accuracy:.4f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath=modelpath, monitor='val_accuracy', verbose=1)

log_dir = "logs_unet/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0) 
# histogram_freq=0 -> metric에 dice_coef까지 쓴 후 killed 문제 때문에 0으로 바꿈


# (256,256,1), n_labels=2, loss=keras_unet_collection.losses.focal_tversky, batch_size=8 이상은 에러남
model.fit(x_train, ohy_train, batch_size=8, epochs=20, verbose=1, validation_data=(x_val, ohy_val), callbacks=[checkpointer]) # , tensorboard_callback