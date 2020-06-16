# Proyecto-Sistemas-de-Visión-Computarizada
Repositorio para el Proyecto Final de Sistemas de Visión Computarizada

Este repositorio contiene código e instrucciones para poder ejecutar un detector de distancia entre vehículos y personas basado en YOLOV3 y Deepsort
<ul>
  <li> Humanos</li>
  <li> Vehículos (montacargas, camiones, carros, etc.)</li>
</ul>

## Instrucciones de ejecución
Estas instrucciones asumen que se esta trabajando en un sistema operativo Windows o Linux.

NOTA: Es posible que haya problemas corriendo el código sobre GPU en un sistema Windows, por lo tanto para usar GPU es recomendado trabajar sobre alguna distribución de linux.

Primeramente hay que instalar Anaconda https://www.anaconda.com/products/individual

Descargamos este repositorio en formato .zip y lo extraemos en el directorio de su preferencia. Para facilitar la explicación asumiremos que el repositorio se extrajo dentro un folder llamado ```Repositorios```

Ahora vamos al siguiente link https://drive.google.com/file/d/1vmWhZ7rmHbu_fpeOnfXl7l33qQq2aoFC/view?usp=sharing y descargamos el archivo ```yolov3_custom_last.weights``` dentro del directorio ```Repositorios/Proyecto-Sistemas-de-Vision-Computarizada/yolov3_deepsort/weights```. Este archivo contiene la configuración de nuestro modelo.

El tener un GPU NVIDIA permite al modelo realizar inferencias más rápido. En caso de tener y querer utilizar un GPU NVIDIA, hay que descargar el driver de su GPU NVIDIA de la siguiente página https://www.nvidia.com/Download/index.aspx

Posteriormente hay que crear el ambiente de ejecución de Anaconda con ayuda del archivo ```Proyecto-Sistemas-de-Vision-Computarizada/yolov3_deepsort/conda-gpu.yml``` en caso de tener GPU NVIDIA (permite al modelo correr más rápido), o en caso de no contar con GPU NVIDIA ```Proyecto-Sistemas-de-Vision-Computarizada/yolov3_deepsort/conda-cpu.yml``` en una terminal de anaconda nos movemos a la carpeta del repositorio y ejecutamos.

```
cd yolov3_deepsort
# Si solo contamos con CPU
conda env create -f conda-cpu.yml
conda activate tracker-cpu

# Si contamos con GPU
conda env create -f conda-gpu.yml
conda activate tracker-gpu
```

Instalamos todas las dependecias y librerías necesarias dentro de nuestro ambiente de ejecución.Dentro de la misma terminal ejecutamos: 
```
# Si solo contamos con CPU
pip install -r requirements.txt

# Si contamos con GPU
pip install -r requirements-gpu.txt
```
Ahora transformamos nuestra configuración descargada a un modelo de Tensorflow. Dentro de la misma terminal ejecutamos: 
```
python load_weights.py --weights ./weights/yolov3_custom_last.weights --output ./weights/yolov3-custom.tf --num_classes 2
```

Listo! Ahora ya podemos ejecutar nuestro detector, con el video de nuestra elección, el repositorio contiene un video de ejemplo llamado ```despacho.mp4```, por lo que lo usaremos en el siguiente comando. Dentro de la misma terminal ejecutamos:
```
python object_tracker.py --video ./data/video/despacho.mp4 --output ./data/video/results.avi --yolo_score_threshold 0.3
#El video resultante con las detecciones se guarda en ./data/video/results.avi
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
