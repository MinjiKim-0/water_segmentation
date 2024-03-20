import tensorflow as tf
from tensorflow import keras
import numpy as np
# pip install pillow
# pip install SciPy
# pip install keras_unet_collection

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_memory_growth(gpus[0], True)
#   except RuntimeError as e:
#     # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
#     print(e)


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#   except RuntimeError as e:
#     # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
#     print(e)


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
batch_size_train = 7203 #7203 #26411
batch_size_val = 900 #900 #300
tf.random.set_seed(seed)

'''
7203
900
'''

'''
26411 -> 뻑
20000 뻑
15000 뻑
설마 2401해도 뻑나는지 봐야한다 ㄱㅊ다!!!!
10000 된다!!!!!!!!
'''


path_aug_x = "/water_segmentation/Training/[원천]train_water_data/aug_train"
# /media/visbic/MGTEC/water_seg/water_segmentation/Training/[원천]train_water_data/aug_train
path_aug_y = "/water_segmentation/Training/[라벨]train_water_labeling/aug_train_label"
# /media/visbic/MGTEC/water_seg/water_segmentation/Training/[라벨]train_water_labeling/aug_train_label
path_val_aug_x = '/water_segmentation/Validation/[원천]validate_water_data/aug_val'

path_val_aug_y = '/water_segmentation/Validation/[라벨]validate_water_labeling/aug_val_label'

# 이미지 전처리로 스케일링을 합니다
img_datagen = keras.preprocessing.image.ImageDataGenerator( rescale = 1.0/255. )

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# x_train
x_train_dir = path_aug_x
x_train_generator = img_datagen.flow_from_directory(x_train_dir, batch_size=batch_size_train, shuffle=False, seed=seed, class_mode=None, color_mode="grayscale")#"grayscale"

x_train = x_train_generator.next()
print("x_train.shape :", x_train.shape)
# x_train = x_train.astype(np.uint8)


# x_val
x_val_dir = path_val_aug_x 
x_val_generator = img_datagen.flow_from_directory(x_val_dir, batch_size=batch_size_val, shuffle=False, seed=seed, class_mode=None, color_mode="grayscale")#"grayscale"

x_val = x_val_generator.next()
print("x_val.shape :", x_val.shape)
# x_val = x_val.astype(np.uint8)


# y_train
y_train_dir = path_aug_y
y_train_generator = img_datagen.flow_from_directory(y_train_dir, batch_size=batch_size_train, shuffle=False, seed=seed, class_mode=None, color_mode="grayscale")#"grayscale"

y_train = y_train_generator.next()
print("y_train.shape :", y_train.shape)
# print(y_train.max())# -> 1
# print(y_train.min())# -> 0
# print(y_train.mean()) #-> 0.19562566
# print(np.median(y_train)) #->0
# 0-> 흑 / 백->1


# y_train[y_train >= 0.5] = 1
# y_train[y_train < 0.5] = 0

ohy_train = tf.one_hot(
    y_train, 2, on_value=None, off_value=None, axis=None, dtype=None, name=None
)
print("resahpe 전 ohy_train.shape :", ohy_train.shape)
ohy_train = ohy_train.reshape(ohy_train.shape[0], 256, 256, 2)
print("resahpe 후 ohy_train.shape :", ohy_train.shape)
# print(ohy_train.max())
# print(ohy_train.min())


# y_val
y_val_dir = path_val_aug_y
y_val_generator = img_datagen.flow_from_directory(y_val_dir, batch_size=batch_size_val, shuffle=False, seed=seed, class_mode=None, color_mode="grayscale")#"grayscale"

y_val = y_val_generator.next()
print("y_val.shape :", y_val.shape)

# y_val[y_val >= 0.5] = 1
# y_val[y_val < 0.5] = 0

ohy_val = tf.one_hot(
    y_val, 2, on_value=None, off_value=None, axis=None, dtype=None, name=None
)
print("resahpe 전 ohy_val.shape :", ohy_val.shape)
ohy_val = ohy_val.reshape(ohy_val.shape[0], 256, 256, 2)
print("resahpe 후 ohy_val.shape :", ohy_val.shape)

#################


from keras_unet_collection import models
import keras_unet_collection.losses
import datetime


# model
model = models.unet_2d((256,256,1), [64, 128, 256, 512, 1024], n_labels=2,
                      stack_num_down=2, stack_num_up=1,
                      activation='GELU', output_activation='Softmax', 
                      batch_norm=True, pool='max', unpool='nearest', name='unet')

# metrics : 측정항목 함수는 loss function와 비슷하지만, 측정항목을 평가한 결과는 모델을 학습시키는데 사용되지 않는다는 점에서 다릅니다. 어느 손실 함수나 측정항목 함수로 사용할 수 있습니다.
model.compile(optimizer='adam', loss=[keras_unet_collection.losses.iou_seg], metrics=['accuracy'])



data_path = '/water_segmentation/unet_model/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
modelpath = data_path + '/{epoch:02d}-{val_accuracy:.4f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath=modelpath, monitor='val_accuracy', verbose=1)

# log_dir = "logs_unet/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0) 
# histogram_freq=0 -> metric에 dice_coef까지 쓴 후 killed 문제 때문에 0으로 바꿈

# (256,256,1), n_labels=2, loss=keras_unet_collection.losses.focal_tversky, batch_size=8 이상은 에러남
history = model.fit(x_train, ohy_train, batch_size=8, epochs=20, verbose=1, validation_data=(x_val, ohy_val), callbacks=[checkpointer]) # , tensorboard_callback

# batch 4를 해도 뻑나네..


'''
threshold 잘못 줌 ㅠㅠ 결과보고 기겁.. 클날 뻔..ㅜㅜ
y_train >= y_train.sum()/256 => 일케 하니까 (n, 256, 256, 1)을 다 sum하니까 당여니 생각과 달리 이상이상...
y_train >= 0.5 이로케 바꿈 ㅜ

0. batch 8, loss iou, metrics accuracy
1. RELU
-. GELU
-. GELU+elastic deformation(7203, 900) 
4. GELU+general augmentation(7203, 900)

0. batch 32, loss iou, metrics binary_accuracy
1. RELU
2. GELU
3. GELU+augmentation
4. GELU+general augmentation

saving model to /water_segmentation/unet_model/20211213-022725/~.hdf5
Epoch 00001: 298s 323ms/step - loss: 0.4580 - accuracy: 0.7004 - val_loss: 0.5919 - val_accuracy: 0.5617
Epoch 00002: 291s 323ms/step - loss: 0.4459 - accuracy: 0.7056 - val_loss: 0.5919 - val_accuracy: 0.5617

결과가 너무 안좋다. shuffle=True 탓?(아냐ㅠ) axis=-1탓?(이건 아닐듯....) uint8탓?(이것도 아닐 것 같은데 ㅜㅜ)
아니 aug했을 뿐인데 이렇게나 성과가 떨어진다고......?
아!!!! ㅇㅋㅇㅋ알겠음. label이 쫌 이상함. water seg가 애초에 잘못됐음. data aug 할 때 threshold 문제인 것 같음
'''




'''
바꾸고싶다.. metric을 무슨 accuracy인지 정확하게..
binary_accuracy
categorical_accuracy
sparse_categorical_accuracy
top_k_categorical_accuracy
sparse_top_k_categorical_accuracy

그리고 RELU로도 돌려보고싶다
0. batch 8, loss iou, metrics accuracy
1. RELU
2. GELU
3. GELU+elastic deformation(10000) 
4. GELU+general augmentation(10000)

0. batch 32, loss iou, metrics binary_accuracy
1. RELU
2. GELU
3. GELU+augmentation

그리고 augmentation(sigma=1, zoom=1.05)했는데 학습 잘 안 될 수도 있다고 생각한다..
augmentation되면서 이미지가 원본이랑 많이 달라져서..
흐려지고 해상도 낮아져서 오히려 그게 noise로 작용할 수도 있다고 생각한다
따라서 학습 잘 안되면 aug다시 해야 할 수도 있다 ㅜㅜ sigma랑 zoom좀 낮춰서.... 
=> 아니다 효과 있다 만세 ㅠㅠㅠㅠㅠㅠㅠ 아 없을지도... val_accuracy가 낮네...형편이 없네... test 데이터가 너무 많아서 그런가봄 ㅜㅜㅜ
하긴 10,000 대 300은 쫌 심햇네.. 100:3... 아무도 이런 비율로는 안나누지...
val_accuracy 오르긴 하는데 너무 코딱지만해.. 이 정도로는 20epoch만에 전과 같은 accuracy(0.9812) 달성은 불가..

355s 280ms/step - loss: 0.0117 - accuracy: 0.9968 - val_loss: 0.7962 - val_accuracy: 0.3309
350s 280ms/step - loss: 6.0289e-04 - accuracy: 0.9997 - val_loss: 0.7897 - val_accuracy: 0.3395
352s 282ms/step - loss: 5.5075e-04 - accuracy: 0.9997 - val_loss: 0.7866 - val_accuracy: 0.3434
354s 283ms/step - loss: 5.3740e-04 - accuracy: 0.9997 - val_loss: 0.7896 - val_accuracy: 0.3393
=> train_set의 성과는 매우 좋으나 val_set은 처참하다....
'''