import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

from augmentation import init_augmentation

start_time = datetime.now()

TRAIN_DIRECTORY = 'dataset/train/'
VAL_DIRECTORY = 'dataset/val/'
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 256
IMAGE_CHANNEL = 3
CLASS_HOLE = (255, 0, 0)
CLASS_OTHER = (0, 0, 0)
MULTIPLE = 10

CLASS_LABELS = (
    CLASS_HOLE,  # дырки
    CLASS_OTHER  # всё остальное
)

init_augmentation(TRAIN_DIRECTORY, (IMAGE_WIDTH, IMAGE_HEIGHT), MULTIPLE)
init_augmentation(VAL_DIRECTORY, (IMAGE_WIDTH, IMAGE_HEIGHT), MULTIPLE * 5)


# Служебная функция загрузки выборки изображений из файлов в папке
def load_imageset(folder,  # имя папки
                  subset,  # подмножество изображений - оригинальные или сегментированные
                  title,  # имя выборки
                  show_print=False
                  ):
    image_list = []
    cur_time = time.time()
    for filename in sorted(os.listdir(f'{folder}/{subset}')):
        image_list.append(image.load_img(os.path.join(f'{folder}/{subset}', filename),
                                         target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)))
    if show_print:
        print('{} выборка загружена. Время загрузки: {:.2f} с'.format(title,
                                                                      time.time() - cur_time))
        print('Количество изображений:', len(image_list))
    return image_list


def rgb_to_labels(image_list):
    result = []
    for d in image_list:
        sample = np.array(d)
        y = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 1), dtype='uint8')
        for i, cl in enumerate(CLASS_LABELS):
            y[np.where(np.all(sample == CLASS_LABELS[i], axis=-1))] = i
        result.append(y)
    return np.array(result)


def labels_to_rgb(image_list):
    result = []
    for y in image_list:
        temp = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype='uint8')
        for i, cl in enumerate(CLASS_LABELS):
            temp[np.where(np.all(y == i, axis=-1))] = CLASS_LABELS[i]
        result.append(temp)
    return np.array(result)


def simple_unet(input_shape  # форма входного изображения
                ):
    img_input = Input(input_shape)  # Создаем входной слой формой input_shape

    # Block 1
    x = Conv2D(32, (3, 3), padding='same', name='block1_conv1')(img_input)  # Добавляем Conv2D-слой с 32-нейронами
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(32, (3, 3), padding='same', name='block1_conv2')(x)  # Добавляем Conv2D-слой с 32-нейронами
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    block_1_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_1_out

    x = MaxPooling2D(4)(block_1_out)  # Добавляем слой MaxPooling2D чтобы сжать картинку в 4 раза

    # Block 2
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv1')(x)  # Добавляем Conv2D-слой с 64-нейронами
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same', name='block2_conv2')(x)  # Добавляем Conv2D-слой с 64-нейронами
    block_2_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling2D(4)(block_2_out)  # Добавляем слой MaxPooling2D чтобы сжать картинку в 4 раза

    # UP 1
    x = Conv2DTranspose(64, (4, 4), strides=(4, 4), padding='same')(
        x)  # Добавляем Conv2DTranspose-слой с 64-нейронами и размером ядра + шага равным 4
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same')(x)  # Добавляем Conv2D-слой с 64-нейронами
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same')(x)  # Добавляем Conv2D-слой с 64-нейронами
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 2
    x = Conv2DTranspose(32, (4, 4), strides=(4, 4), padding='same')(
        x)  # Добавляем Conv2DTranspose-слой с 32-нейронами и размером ядра + шага равным 4
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(32, (3, 3), padding='same')(x)  # Добавляем Conv2D-слой с 32-нейронами
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(32, (3, 3), padding='same')(x)  # Добавляем Conv2D-слой с 32-нейронами
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(len(CLASS_LABELS), (3, 3), activation='softmax', padding='same')(
        x)  # Добавляем Conv2D-Слой с softmax-активацией на class_count-нейронов

    model = Model(img_input, x)  # Создаем модель с входом 'img_input' и выходом 'x'

    # Компилируем модель
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    # Возвращаем сформированную модель
    return model


def create_model(input_shape):
    img_input = Input(input_shape)  # Создаем входной слой формой input_shape
    print(img_input)
    x = Conv2D(128, (3, 3), padding='same', name='block1_conv1')(img_input)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    # x = Dropout(0.3)(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv3')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    # x = Dropout(0.3)(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(32, (3, 3), padding='same', name='block1_conv4')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    # x = Dropout(0.3)(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)

    x = Conv2D(16, (3, 3), padding='same', name='block1_conv5')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    # x = Dropout(0.3)(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)

    x = Conv2D(len(CLASS_LABELS), (3, 3), activation='softmax', padding='same')(
        x)  # Добавляем Conv2D-Слой с softmax-активацией на len(CLASS_LABELS)-нейронов

    model = Model(img_input, x)  # Создаем модель с входом 'img_input' и выходом 'x'

    # Компилируем модель
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    # Возвращаем сформированную модель
    return model


def process_images(model,  # обученная модель
                   x_val,
                   val_segments,
                   val_images,
                   count=2  # количество случайных картинок для сегментации
                   ):
    # Генерация случайного списка индексов в количестве count между (0, len(x_val)
    indexes = np.random.randint(0, len(x_val), count)

    # Вычисление предсказания сети для картинок с отобранными индексами
    predict = np.argmax(model.predict(x_val[indexes]), axis=-1)

    # Подготовка цветов классов для отрисовки предсказания
    orig = labels_to_rgb(predict[..., None])
    fig, axs = plt.subplots(3, count, figsize=(25, 15))

    # Отрисовка результата работы модели
    for i in range(count):
        # Отображение на графике в первой линии предсказания модели
        axs[0, 0].set_title('Результат работы модели:')
        axs[0, i].imshow(orig[i])
        axs[0, i].axis('off')

        # Отображение на графике во второй линии сегментированного изображения из y_val
        axs[1, 0].set_title('Оригинальное сегментированное')
        axs[1, i].imshow(val_segments[indexes[i]])
        axs[1, i].axis('off')

        # Отображение на графике в третьей линии оригинального изображения
        axs[2, 0].set_title('Оригинальное изображение')
        axs[2, i].imshow(val_images[indexes[i]])
        axs[2, i].axis('off')

    plt.show()


train_images = load_imageset(TRAIN_DIRECTORY, 'original', 'Обучающая')
val_images = load_imageset(VAL_DIRECTORY, 'original', 'Проверочная')

train_segments = load_imageset(TRAIN_DIRECTORY, 'segment', 'Обучающая')
val_segments = load_imageset(VAL_DIRECTORY, 'segment', 'Проверочная')

y = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 1), dtype='uint8')

x_train = []  # Cписок под обучающую выборку
for img in train_images:  # Для всех изображений выборки:
    x = image.img_to_array(img)  # Перевод изображения в numpy-массив формы: высота x ширина x количество каналов
    x_train.append(x)  # Добавление элемента в x_train

x_train = np.array(x_train)  # Перевод всей выборки в numpy
x_val = []  # Список под проверочную выборку

for img in val_images:  # Для всех изображений выборки:
    x = image.img_to_array(img)  # Перевод изображения в numpy-массив формы: высота x ширина x количество каналов
    x_val.append(x)  # Добавление элемента в x_train

x_val = np.array(x_val)  # Перевод всей выборки в numpy

y_train = rgb_to_labels(train_segments)
y_val = rgb_to_labels(val_segments)
print(f'{datetime.now() - start_time} потребовалось на обработку фото')

model_test = create_model((IMAGE_WIDTH, IMAGE_HEIGHT, 3))
# model_test = simple_unet((IMAGE_WIDTH, IMAGE_HEIGHT, 3))

start_time = datetime.now()

history = model_test.fit(x_train, y_train,
                         epochs=60, batch_size=int(1 * MULTIPLE / 2) if MULTIPLE > 1 else 2,
                         validation_data=(x_val, y_val))

plt.figure(figsize=(14, 7))
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

# process_images(model_test, x_train, train_segments, train_images, 6)
process_images(model_test, x_val, val_segments, val_images, 6)

print(f'{datetime.now() - start_time} потребовалось на обучение')
