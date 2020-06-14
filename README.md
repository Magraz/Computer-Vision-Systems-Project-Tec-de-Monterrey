# Proyecto-Sistemas-de-Visión-Computarizada
Repositorio para el Proyecto Final de Sistemas de Visión Computarizada

Este repositorio contiene código e instrucciones para poder ejecutar un detector de distancia entre vehículos y personas basado en YOLOV3 y Deepsort
<ul>
  <li> Humanos</li>
  <li> Vehículos (montacargas, camiones, carros, etc.)</li>
</ul>

## Instrucciones de ejecución

Primeramente hay que instalar Anaconda https://www.anaconda.com/products/individual

Descargamos este repositorio en formato .zip y lo extraemos en el directorio de su preferencia. Para facilitar la explicación asumiremos que el repositorio se extrajo dentro un folder llamado ```Repositorios```

Ahora vamos al siguiente link https://drive.google.com/file/d/1FyiScOQduz6I3KBoRksW2lj43sAxZVAZ/view?usp=sharing y descargamos el archivo ```VGG16_bikes_cars_trains_rollers_trucks.h5``` dentro del directorio ```Repositorios/ProyectoFinal-Tec.-Info-Emergentes```. Este archivo contiene la configuración de nuestro clasificador

Posteriormente hay que crear el ambiente de ejecución de Anaconda con ayuda del archivo ```clasificadorWeb.yml``` para poder correr la aplicación web. En una terminal de Anaconda con permisos de administrador ejecutamos:

```
conda env create -f clasificadorWeb.yml
conda activate clasificadorWeb
```

Dentro de la misma terminal nos movemos al directorio ```Repositorios```  y ejecutamos: 
```
cd ProyectoFinal-Tec.-Info-Emergentes/flask_apps
set FLASK_APP=predict_app.py
flask run --host=0.0.0.0
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
