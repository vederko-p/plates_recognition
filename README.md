# plates_recognition

## Модуль тестирования модели на видео

**Настройка**

Основные настройки задаются в конфиге **src/configs/config_test_video**:

* video_filepath - путь до видео
* gt_filepath - путь до разметки видео
* st_filepath [str | None] - путь до предсказаний по видео (если не None, то
* будут произведены вычисления)
* output_csv_directory_path - путь до директории, куда будут сохранены результаты предсказаний модели
* display_video [True | False] - воспроизводить видео или нет
* skip_frames_by_model [0, 1, 2, ...] - количество пропускаемых моделью кадров
* metrics_csv_directory_path - путь до директории, куда будут сохранены значения
* метрик

Для выбора определенной модели необходимо импортировать и создать соответсвующий
экземпляр в файле **test_video.py**. Модель должна иметь метод \_\_call\_\_(),
принимать на вход кадр и возвращать словарь с фиксированным набором ключей, где
значения по ключам - строки.

**Запуск**

После всех настроек. Из src:

    python3 test_video.py

## Метрики

Для видео метрики считаются по объектам - по машинам. В данном пункте можно
считать слова **номер** и **машина** синонимами.

1. Верные предсказания по машинам. Данная метрика позволяет отслеживать, сколько
машин из представленных на видео модель распознает верно:

TPR = \dfrac{TP}{TP+FN}

2. Ложные предсказания по машинам. Данная метрика позволяет отслеживать, сколько
модель распознает машин, которых не было на видео:

FPR^{*} = \dfrac{FP}{TP+FP}

3. Среднее время распознавания машин. Данная метрика позволяет отслеживать, 
сколько времени модели требуется на распознавание автомобиля:

V = E_{t_j}

## Данные

MIOvision Traffic Camera Dataset - детекция транспортных средств (38200 изображений, 11 классов)
Car License Plate Detection - детекция номера автомобиля (1118 изображений)
Nomeroff Russian license plates - распознавание номера автомобиля (104400 изображений)

## Результаты обучения

| model        | Parameters                      | P    | R   | mAP50| mAP50-95 |
| -------------|:-------------------------------:| -----|-----|------|----------|
| Yolo5m       |size 736, 11 classes, 10 epochs  | 0.67 | 0.66| 0.69 | 0.47     |


