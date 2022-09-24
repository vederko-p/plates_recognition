# plates_recognition

## Модуль тестирования модели на видео

**Настройка**

Основные настройки задаются в конфиге *src/configs/config_test_video*:

* video_filepath - путь до видео
* gt_filepath - путь до разметки видео
* st_filepath [str | None] - путь до предсказаний по видео (если не None, то будут произведены вычисления)
* output_csv_directory_path - путь до директории, куда будут сохранены результаты предсказаний модели
* display_video [True | False] - воспроизводить видео или нет
* skip_frames_by_model [0, 1, 2, ...] - количество пропускаемых моделью кадров
* metrics_csv_directory_path - путь до директории, куда будут сохранены значения метрик

Для выбора определенной модели необходимо импортировать и создать соответсвующий экземпляр в файле test_video.py.
Модель должна иметь метод __call__(), принимать на вход кадр и возвращать словарь с фиксированным набором ключей, где
значения по ключам - строки.

**Запуск**

После всех настроек. Из src:
    python3 test_video.py