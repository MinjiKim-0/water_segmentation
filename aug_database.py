import tensorflow as tf
from tensorflow import keras
import numpy as np

print('TensorFlow {}; Keras {}'.format(tf.__version__, keras.__version__))


# tf.debugging.set_log_device_placement(True)

# # 텐서 생성
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)

# print(c)
# # Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0


##################### 

seed = 64
batch_size_train = 10
batch_size_val = 10
tf.random.set_seed(seed)


x_train_db = {}
y_train_db = {}
x_val_db = {}
y_val_db = {}


path_aug_x = "/water_segmentation/Training/[원천]train_water_data/aug_train"
# /media/visbic/MGTEC/water_seg/water_segmentation/Training/[원천]train_water_data/aug_train
path_aug_y = "/water_segmentation/Training/[라벨]train_water_labeling/aug_train_label"
# /media/visbic/MGTEC/water_seg/water_segmentation/Training/[라벨]train_water_labeling/aug_train_label

# 이미지 전처리로 스케일링을 합니다
img_datagen = keras.preprocessing.image.ImageDataGenerator( rescale = 1.0/255. )

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# x_train
if x_train_db.get('x_train') is not None: # 데이터베이스에 자료가 있으면
    x_train = x_train_db.get('x_train')
    pass

else: # 데이터베이스에 자료가 없으면
    x_train_dir = path_aug_x
    x_train_generator = img_datagen.flow_from_directory(x_train_dir, batch_size=batch_size_train, shuffle=False, seed=seed, class_mode=None, color_mode="grayscale")#"grayscale"

    x_train = x_train_generator.next()
    print("x_train.shape :", x_train.shape)
    x_train_db.setdefault('x_train', x_train)


# x_val
if x_val_db.get('x_val') is not None:
    x_val = x_val_db.get('x_val')
    pass

else:
    x_val_dir = '/water_segmentation/Validation/[원천]validate_water_data'
    x_val_generator = img_datagen.flow_from_directory(x_val_dir, batch_size=batch_size_val, shuffle=False, seed=seed, class_mode=None, color_mode="grayscale")#"grayscale"

    x_val = x_val_generator.next()
    print("x_val.shape :", x_val.shape)
    x_val_db.setdefault('x_val', x_val)



# y_train
if y_train_db.get('y_train') is not None:
    y_train = y_train_db.get('y_train')
    pass

else:
    y_train_dir = path_aug_y
    y_train_generator = img_datagen.flow_from_directory(y_train_dir, batch_size=batch_size_train, shuffle=False, seed=seed, class_mode=None, color_mode="grayscale")#"grayscale"

    y_train = y_train_generator.next()
    print("y_train.shape :", y_train.shape)

    y_train[y_train >= 128] = 255
    y_train[y_train < 128] = 0

    y_train = y_train.astype(np.uint8)

    ohy_train = tf.one_hot(
        y_train, 2, on_value=None, off_value=None, axis=-1, dtype="uint8", name=None
    )
    print("resahpe 전 ohy_train.shape :", ohy_train.shape)
    ohy_train = ohy_train.reshape(ohy_train.shape[0], 256, 256, 2)
    print("resahpe 후 ohy_train.shape :", ohy_train.shape)

    y_train_db.setdefault('y_train', ohy_train)



# y_val
if y_val_db.get('y_val') is not None:
    y_val = y_val_db.get('y_val')
    pass

else:
    y_val_dir = '/water_segmentation/Validation/[라벨]validate_water_labeling'
    y_val_generator = img_datagen.flow_from_directory(y_val_dir, batch_size=batch_size_val, shuffle=False, seed=seed, class_mode=None, color_mode="grayscale")#"grayscale"

    y_val = y_val_generator.next()
    print("y_val.shape :", y_val.shape)


    ohy_val = tf.one_hot(
        y_val, 2, on_value=None, off_value=None, axis=None, dtype=None, name=None
    )
    print("resahpe 전 ohy_val.shape :", ohy_val.shape)
    ohy_val = ohy_val.reshape(ohy_val.shape[0], 256, 256, 2)
    print("resahpe 후 ohy_val.shape :", ohy_val.shape)

    y_val_db.setdefault('y_val', ohy_val)