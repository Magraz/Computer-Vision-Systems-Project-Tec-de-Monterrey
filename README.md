# Computer Vision Systems Project
Repositorio para el Proyecto Final de Sistemas de Visión Computarizada

This repository contains code and instructions to execute a distance detector between vehicles and people base in YOLOV3 and DeepSort algorithms.
<ul>
  <li> Humans</li>
  <li> Vehicles (forklifts, trucks, cars, etc.)</li>
</ul>

## Execution Instructions
This instructions assume that you are working on Windows or Linux OS.
NOTE: It is possible that there are problems running the code on GPU in a Windows system, therefore to use GPU it is recommended to work on some Linux distribution.

First you have to install Anaconda https://www.anaconda.com/products/individual

We download this repository in .zip format and extract it in the directory of your choice. To facilitate the explanation we will assume that the repository was extracted into a folder called `` `Repositories```

Now we go to the following link https://drive.google.com/file/d/1vmWhZ7rmHbu_fpeOnfXl7l33qQq2aoFC/view?usp=sharing and download the file ```yolov3_custom_last.weights``` inside the directory ```Repositories/Project-Systems -de-Vision-Computarizada/yolov3_deepsort/weights```. This file contains the configuration of our model.

Having an NVIDIA GPU allows the model to make inferences faster. In case you have and want to use an NVIDIA GPU, you must download the driver for your NVIDIA GPU from the following page https://www.nvidia.com/Download/index.aspx

Subsequently, the Anaconda execution environment must be created with the help of the file ```Proyecto-Sistemas-de-Vision-Computarizada/yolov3_deepsort/conda-gpu.yml``` in case of having NVIDIA GPUs (allows the model to run faster), or in case of not having NVIDIA GPU ```Proyecto-Sistemas-de-Vision-Computarizada/yolov3_deepsort/conda-cpu.yml``` in an anaconda terminal we move to the repository folder and execute.

```
cd yolov3_deepsort
# If we count with a CPU
conda env create -f conda-cpu.yml
conda activate tracker-cpu

# If we count with a GPU
conda env create -f conda-gpu.yml
conda activate tracker-gpu
```

Install all the necessary dependencies and libraries within our execution environment. Within the same terminal we execute:
```
# If we only count CPU
pip install -r requirements.txt

# If we count with GPU
pip install -r requirements-gpu.txt
```
Now we transform our downloaded configuration to a Tensorflow model. Within the same terminal we execute:
```
python load_weights.py --weights ./weights/yolov3_custom_last.weights --output ./weights/yolov3-custom.tf --num_classes 2
```

Done! Now we can run our detector, with the video of our choice, the repository contains an example video called ```despacho.mp4```, so we will use it in the following command. Within the same terminal we execute:
```
python object_tracker.py --video ./data/video/despacho.mp4 --output ./data/video/results.avi --yolo_score_threshold 0.3
#The resulting video with the detections is saved in ./data/video/results.avi
```

El programa también se puede correr tomando los datos de una cámara con el siguiente comando:
```
#Debemos conocer el número asignado a nuestro dispotivo cámara, en este caso el número es 0, el cual en caso de ejecutar el código sobre una laptop representa la cámara web.
python object_tracker.py --video 0 --output ./data/video/results.avi --yolo_score_threshold 0.3
```
## Salida del programa
Al ejecutar el comando anterior deberíamos obtener una pantalla mostrando el video y las detecciones como en las siguientes imágenes:

![alt text](https://github.com/Magraz/Proyecto-Sistemas-de-Vision-Computarizada/blob/master/images/example.PNG "Ejemplo 1")
---
![alt text](https://github.com/Magraz/Proyecto-Sistemas-de-Vision-Computarizada/blob/master/images/example2.PNG "Ejemplo 2")

El programa genera detecciones y rastrea estas detecciones a lo largo del video. En cuanto un vehículo se encuentra en un radio de 300px. (este número se puede modificar en el archivo object_tracker.py, por motivos de demostración en el video el valor es más alto de lo necesario) de un humano se aumenta el contador de personas en peligro  el humano es rodeado por un círculo rojo y el vehículo por un círculo verde, las dos clases son conectadas por una línea azul que indica la conexión entre los objetos. Por último, el video es guardado en `Proyecto-Sistemas-de-Vision-Computarizada/yolov3_deepsort/data/videos` con el nombre que se asigne en el comando.

## Solución experimental
Contamos con una versión experimental del detector que compensa la diferencia de perspectiva y maneja las medidas en metros.
Esta versión tiene un uso igual al original, por lo que todos los comandos anteriores aplican siempre y cuando se cambie el nombre por el de esta versión. Por lo tanto, para ejecutar detección en un video abrimos una terminal de anaconda y nos movemos a la carpeta ```Repositorios/Proyecto-Sistemas-de-Vision-Computarizada``` y corremos los siguientes comandos:
```
conda activate tracker-gpu  # En caso de contar con GPU
conda activate tracker-gpu # En caso de contar solo con CPU
cd yolov3_deepsort
python object_tracker_perspective.py --video ./data/video/despacho.mp4 --output ./data/video/results.avi --yolo_score_threshold 0.3
```
Se abrirá una ventana con el primer cuadro del video congelado y se pedirá que se realicen algunos pasos de iniciación. 
El primer paso es elegir dos líneas que estén en el mismo plano que el piso y sean paralelas en el mundo. Por decirlo de otra manera, se seleccionan los puntos que harían un rectángulo en el mundo.

El orden de selección es el siguiente: punto superior izquierdo, luego inferior izquierdo, punto superior derecho y finalmente, punto inferior derecho.

El segundo paso es seleccionar la altura de una persona. Simplemente seleccionar la cabeza y los pies.

El último paso es seleccionar el área que se desea monitorear. Solo encerrar el área deseada dentro del rectángulo verde.
Y listo. El video se analizará hasta completarse. Del lado izquierdo de la pantalla aparecerá la vista aérea de los puntos y del lado derecho aparecerá el video con las detecciones. Una línea roja se dibujará entre objetos si están a una menor distancia que la establecida, esta distancia (metros) se puede modificar en el archivo.

Un video con el proceso puede verse aqui https://www.youtube.com/watch?v=E8dFuK5nDh0

---
![alt text](https://github.com/Magraz/Proyecto-Sistemas-de-Vision-Computarizada/blob/master/images/example3.PNG "Ejemplo 3")

## Referencias
Este repositorio esta basado en código de los siguientes repositorios:
* https://github.com/theAIGuysCode/yolov3_deepsort
* https://github.com/aqeelanwar/SocialDistancingAI
