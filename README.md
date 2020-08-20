# Computer Vision Systems Final Project
Repository for the Computer Vision Systems Final Project at Tec de Monterrey

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
# If we only count with CPU
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

The program can also be run taking data from a camera with the following command:
```
#We must know the number assigned to our camera device, in this case the number is 0, which in case of executing the code on a laptop represents the webcam.
python object_tracker.py --video 0 --output ./data/video/results.avi --yolo_score_threshold 0.3
```
## Program's Output
When executing the previous command we should obtain a screen showing the video and the detections as in the following images:

![alt text](https://github.com/Magraz/Proyecto-Sistemas-de-Vision-Computarizada/blob/master/images/example.PNG "Ejemplo 1")
---
![alt text](https://github.com/Magraz/Proyecto-Sistemas-de-Vision-Computarizada/blob/master/images/example2.PNG "Ejemplo 2")

A video with results can be seen here https://www.youtube.com/watch?v=Y-Pjx-q0dsA&feature=emb_title

El programa genera detecciones y rastrea estas detecciones a lo largo del video. En cuanto un vehículo se encuentra en un radio de 300px. (este número se puede modificar en el archivo object_tracker.py, por motivos de demostración en el video el valor es más alto de lo necesario) de un humano se aumenta el contador de personas en peligro  el humano es rodeado por un círculo rojo y el vehículo por un círculo verde, las dos clases son conectadas por una línea azul que indica la conexión entre los objetos. Por último, el video es guardado en `Proyecto-Sistemas-de-Vision-Computarizada/yolov3_deepsort/data/videos` con el nombre que se asigne en el comando.

The program generates detections and tracks these detections throughout the video. As soon as a vehicle is within a 300px radius (this number can be modified in the object_tracker.py file, for demonstration purposes in the video the value is higher than necessary) of a human, the counter of people in danger is increased and the human is surrounded by a red circle and the vehicle by a green circle, the two classes are connected by a blue line that indicates the connection between the objects. Finally, the video is saved in `Proyecto-Sistemas-de-Vision-Computarizada/yolov3_deepsort/data/ videos` with the name assigned in the command.

## Experimental Solution
Contamos con una versión experimental del detector que compensa la diferencia de perspectiva y maneja las medidas en metros.
Esta versión tiene un uso igual al original, por lo que todos los comandos anteriores aplican siempre y cuando se cambie el nombre por el de esta versión. Por lo tanto, para ejecutar detección en un video abrimos una terminal de anaconda y nos movemos a la carpeta ```Repositorios/Proyecto-Sistemas-de-Vision-Computarizada``` y corremos los siguientes comandos:

We have an experimental version of the detector that compensates for the difference in perspective and handles measurements in meters.
This version has a use equal to the original one, so all the previous commands apply as long as the name is changed to that of this version. Therefore, to run detection in a video we open an anaconda terminal and move to the folder ```Repositorios/Proyecto-Sistemas-de-Vision-Computarizada``` and run the following commands:

```
conda activate tracker-gpu  # If we count with a CPU
conda activate tracker-gpu # If we count with a GPU
cd yolov3_deepsort
python object_tracker_perspective.py --video ./data/video/despacho.mp4 --output ./data/video/results.avi --yolo_score_threshold 0.3
```
A window will open with the first frame of the video frozen and you will be prompted for some startup steps.
The first step is to choose two lines that are in the same plane as the floor and are parallel in the world. To put it another way, the points that would make a rectangle in the world are selected.

The order of selection is as follows: upper left point, then lower left point, upper right point and finally lower right point.

The second step is to select a person's height. Simply stretch the line from the head to the feet.

The last step is to select the area you want to monitor. Just enclose the desired area within the green rectangle.

We are done. The video will be analyzed to completion. On the left side of the screen the aerial view of the points will appear and on the right side the video with the detections will appear. A red line will be drawn between objects if they are at a shorter distance than the established one, this distance (meters) can be modified in the file.

A video with the calibration process can be seen here https://www.youtube.com/watch?v=E8dFuK5nDh0
A video with the results can be seen here https://www.youtube.com/watch?v=KrtP9w_ymEM

---
![alt text](https://github.com/Magraz/Proyecto-Sistemas-de-Vision-Computarizada/blob/master/images/example3.PNG "Ejemplo 3")

## References
This repository is based on code from the following repositories:
* https://github.com/theAIGuysCode/yolov3_deepsort
* https://github.com/aqeelanwar/SocialDistancingAI
