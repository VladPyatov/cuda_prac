Выбранный фильтр - Детектор границ Canny

Cборка:
$ cd canny
$ make
$ ./canny.bin ./tea.png

Детали сборки:
Используются OpenMP, CUDA и OpenCV4 (если используете OpenCV3 в makefile надо прописать в переменную opencv4 = `pkg-config --cflags --libs opencv`)

Если не импортится OpenCV:
Если при установке OpenCV было CMAKE_INSTALL_PREFIX=/usr/local
То нужно проверить/добавить
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig
(если этой директории нет, то при установке через CMake нужно добавить флаг -DOPENCV_GENERATE_PKGCONFIG=ON или -D OPENCV_GENERATE_PKGCONFIG=YES)
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

Аргументы командной строки:
-image - изображение
-sigma - параметр размытия
-low_t - нижняя граница
-high_t - верхняя граница
пример: ./canny.bin ./tea.png -sigma=3 -low_5=0.05 -high_t=0.09

Спецификация ПК (Cервер лаборатории ММОИ):
CPU: Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
GPU: NVIDIA RTX 2080TI

Результаты замеров времени (использовались данные из архива с примерами):
--WasteWhite_LDR_0001.png--
CPU 1 thread time 738.025334 ms
CPU 8 threads time 157.613574 ms
GPU without copy: 1.050208
GPU with copy: 8.863008
GPU copy time: 7.812800

--Bathroom_LDR_0001.png--
CPU 1 thread time 4223.946100 ms
CPU 8 threads time 793.443495 ms
GPU without copy: 5.533184
GPU with copy: 47.379711
GPU copy time: 41.846527

--Animation01_LDR_0000.png--
CPU 1 thread time 262.480224 ms
CPU 8 threads time 56.424804 ms
GPU without copy: 0.448512
GPU with copy: 3.480192
GPU copy time: 3.031680

Отчёт о выполненных оптимизациях:
Размытие с ядром Гаусса делалось сепарабельным фильтром (ускорение ~21%)


