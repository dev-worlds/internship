import math  # Математические функции
import random  # Генерация случайных чисел
from PIL import Image, ImageEnhance  # Инструменты для работы с изображениями
import os
from tensorflow.keras.preprocessing import image


def init_augmentation(path, size, multiplier=1, remove_old_augmentation_image=True):
    (IMAGE_WIDTH, IMAGE_HEIGHT) = size
    if remove_old_augmentation_image:
        remove_augmentation_image([f'{path}/original/', f'{path}/segment/'])

    for iteration in range(multiplier):
        for filename in sorted(os.listdir(f'{path}/original/')):
            if 'augmentation' not in filename:
                images = []
                images.append(image.load_img(os.path.join(f'{path}/original/', filename),
                                             target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)))
                images.append(image.load_img(os.path.join(f'{path}/segment/', filename),
                                             target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)))
                images = augment_image(images)
                image.save_img(os.path.join(f'{path}/original/', f'augmentation{iteration + 1}_' + filename), images[0])
                image.save_img(os.path.join(f'{path}/segment/', f'augmentation{iteration + 1}_' + filename), images[1])


def remove_augmentation_image(paths=[]):
    for path in paths:
        for filename in sorted(os.listdir(f'{path}')):
            if 'augmentation' in filename:
                os.remove(os.path.join(f'{path}', filename))


def augment_image(images,  # Изображение для аугментации
                  ang=45,  # Максимальный угол поворота
                  f_x=0.10,  # Максимальная подрезка по ширине
                  f_y=0.10,  # Максимальная подрезка по высоте
                  level_contr=0.3,  # Максимальное отклонение коэффициента контраста от нормы
                  level_brght=0.3):  # Максимальное отклонение коэффициента яркости от нормы

    # Функция нахождения ширины и высоты прямоугольника наибольшей площади
    # после поворота заданного прямоугольника на угол в градусах

    def rotated_rect(w, h, angle):
        angle = math.radians(angle)
        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

        return wr, hr

    # Функция случайной обрезки

    def random_crop(x,  # Подаваемое изображение
                    f_x=f_x,  # Предел обрезки справа и слева (в масштабе ширины)
                    f_y=f_x  # Предел обрезки сверху и снизу (в масштабе высоты)
                    ):
        # Получение левой и правой границ обрезки
        left = x[0].width * random.random() * f_x
        right = x[0].width * (1. - random.random() * f_x) - 1.

        # Получение верхней и нижней границ обрезки
        upper = x[0].height * random.random() * f_y
        lower = x[0].height * (1. - random.random() * f_y) - 1.
        return [x[0].crop((left, upper, right, lower)), x[1].crop((left, upper, right, lower))]

    # Функция случайного поворота

    def random_rot(x,  # Подаваемое изображение
                   ang=ang  # Максимальный угол поворота
                   ):

        # Случайное значение угла в диапазоне [-ang, ang]
        a = random.uniform(-1., 1.) * ang

        # Вращение картинки с расширением рамки
        r0 = x[0].rotate(a, expand=True)
        r1 = x[1].rotate(a, expand=True)

        # Вычисление размеров прямоугольника обрезки максимальной площади
        # для размеров исходной картинки и угла поворота в градусах
        crop_w0, crop_h0 = rotated_rect(x[0].width, x[0].height, a)
        crop_w1, crop_h1 = rotated_rect(x[1].width, x[1].height, a)

        # Обрезка повернутого изображения и возврат результата
        w0, h0 = r0.size
        w1, h1 = r1.size
        return [r0.crop(((w0 - crop_w0) * 0.5, (h0 - crop_h0) * 0.5,
                         (w0 + crop_w0) * 0.5, (h0 + crop_h0) * 0.5)),
                r1.crop(((w1 - crop_w1) * 0.5, (h1 - crop_h1) * 0.5,
                         (w1 + crop_w1) * 0.5, (h1 + crop_h1) * 0.5))]

    # Функция отражения

    def trans_img(x):
        return x.transpose(Image.FLIP_LEFT_RIGHT)

    # Функция случайного изменения контрастности

    def random_contrast(x,  # Подаваемое изображение
                        level=level_contr  # Максимальное отклонение коэффициента контраста от нормы - число от 0. до 1.
                        ):

        enh = ImageEnhance.Contrast(x[0])  # Создание экземпляра класса Contrast
        factor = random.uniform(1. - level,
                                1. + level)  # Cлучайный коэффициент контраста из указанного интервала
        return [enh.enhance(factor), x[1]]  # Изменение коэффициента контраста

    # Функция случайного изменения яркости

    def random_brightness(x,  # Подаваемое изображение
                          level=level_brght  # Максимальное отклонение коэффициента яркости от нормы - число от 0. до 1.
                          ):

        enh = ImageEnhance.Brightness(x[0])  # Создание экземпляра класса Brightness
        factor = random.uniform(1. - level,
                                1. + level)  # Cлучайный коэффициент контраста из указанного интервала

        return [enh.enhance(factor), x[1]]  # Изменение коэффициента яркости

    # Тело основной функции

    # Cоздание списка модификаций
    mod_oper = [random_rot,
                random_crop,
                random_contrast,
                random_brightness]

    # Cлучайное количество изменений из списка; минимум одно изменение
    mod_count = random.randrange(len(mod_oper) + 1)

    # Случайный отбор индексов изменений в количестве mod_count без повторений
    mod_list = random.sample(range(len(mod_oper)), mod_count)

    # Применение модификаций по индексам из mod_list
    for mod_index in mod_list:
        images = mod_oper[mod_index](images)

    # Возврат результата
    return images
