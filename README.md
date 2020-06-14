# Proyecto-Sistemas-de-Visión-Computarizada
Repositorio para el Proyecto Final de Sistemas de Visión Computarizada

Este repositorio contiene código e instrucciones para poder ejecutar un detector de distancia entre vehículos y personas basado en YOLOV3 y Deepsort
<ul>
  <li> Humanos</li>
  <li> Vehículos (montacargas, camiones, carros, etc.)</li>
</ul>

## Instrucciones de ejecución
Estas instrucciones asumen que se esta trabajando en un sistema operativo Windows.

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
python load_weights.py --weights ./weights/<YOUR CUSTOM WEIGHTS FILE> --output ./weights/yolov3-custom.tf --num_classes <# CLASSES>
```

Estos comandos realizan la configuración necesaria para habilitar nuestro clasificador web. Para poder utilizar la aplicación web hay que accessar a cualquiera de las siguientes páginas locales:
* http://localhost:5000/static/predict_with_visuals.html Clasificador que los resultados a través de gráficas
* http://localhost:5000/static/predict.html Clasificador que muestra las probabilidades de cada clase

 <b>NOTA:</b> Estas páginas solo funcionan desde la computadora en la que se esta realizando esta configuración

## Ejecución esperada
Si accedemos a la página local http://localhost:5000/static/predict_with_visuals.html nos aparece la siguiente pantalla:

![alt text](https://github.com/Magraz/ProyectoFinal-Tec.-Info-Emergentes/blob/master/images/start.PNG "Pantalla de inicio")

Primeramente debemos elegir una imagen a clasificar presionando el botón "Choose file", ya que este cargada la imagen presionamos el botón "Predict" y obtendremos un resultado como este:

![alt text](https://github.com/Magraz/ProyectoFinal-Tec.-Info-Emergentes/blob/master/images/visuals.PNG "Resultado")

Si accedemos a la página local http://localhost:5000/static/predict.html el proceso es el mismo al anterior, solo que ahora se presentan los resultados en forma de probabilidades por clase:

![alt text](https://github.com/Magraz/ProyectoFinal-Tec.-Info-Emergentes/blob/master/images/no_visuals.PNG "Resultado")
