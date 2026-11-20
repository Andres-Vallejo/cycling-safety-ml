# CyclingSafety

**Clasificación de eventos de seguridad ciclista con sensores de movimiento y visualización de sus ubicaciones.**

Proyecto académico en equipo que combina señales IMU de smartphone, **MLP y CNN 1D**, mapas GPS y un prototipo experimental **ESP32/TinyML**.

**Mi contribución — Juan Andrés Vallejo Rozo:** construcción del modelo y entrenamiento preliminar, que posteriormente utilizó un compañero para generar los mapas. Esta copia conserva el [repositorio original](https://github.com/w1sec0d/CyclingSafety), los créditos del equipo y su [licencia MIT](LICENSE).

[English](README.md) · [Autores](AUTHORS.md) · [Galería](docs/GALLERY.md) · [Revisión](docs/REVIEW.md) · [Ejecución](docs/SETUP.md)

## De las predicciones al mapa

[![Ubicación de las predicciones guardadas](media/prediction-locations.png)](outputs/map_combined.html)

*Vista local de las predicciones existentes, sin teselas externas. No representa accidentes verificados ni una clasificación validada de peligrosidad vial.*

- [Mapa combinado con capas](outputs/map_combined.html).
- [Mapa de eventos clasificados](outputs/map_mlp_events.html).
- [Mapa de calor de severidad](outputs/map_severity_heatmap.html).

Descarga el repositorio y abre el HTML en el navegador para interactuar con los mapas. GitHub muestra su código fuente. Los archivos Folium consultan scripts y teselas externas; la vista estática del README no utiliza esos servicios.

[![Vista espacial local de los puntajes de severidad guardados](media/severity-preview.png)](outputs/map_severity_heatmap.html)

*Promedio del puntaje guardado por celda espacial, generado localmente. Es una agregación de los resultados existentes, no un nuevo modelo validado de riesgo.*

## Qué hicimos

Exploramos la identificación de impactos por **baches**, **esquivadas** y **frenadas de emergencia** a partir del acelerómetro y el giroscopio. El flujo remuestrea las señales a 50 Hz, crea ventanas de 128 muestras y compara:

- **MLP:** recibe características estadísticas extraídas de cada ventana.
- **CNN 1D:** aprende patrones temporales directamente de los seis canales IMU.

Las predicciones se asocian con GPS para generar mapas interactivos. También se incluyen modelos entrenados, experimentos de búsqueda de hiperparámetros, una exportación INT8 y firmware para un nodo ESP32 con MPU-6050, GPS y microSD.

## Comparación preliminar

![Comparación supervisada guardada en el notebook](media/supervised-comparison.png)

| Modelo supervisado, cuatro clases | Exactitud registrada | Macro-F1 registrado |
|---|---:|---:|
| MLP con características estadísticas | 97,91 % | 0,8942 |
| CNN 1D manual | 96,13 % | 0,6936 |
| CNN 1D con KerasTuner | 99,14 % | 0,9468 |

Son resultados guardados; no se reentrenaron para esta publicación. La partición mezcla ventanas solapadas de los mismos recorridos. Además, el entrenamiento contiene **303 copias aumentadas derivadas de ventanas de prueba y 300 de validación**. Las cifras son exploratorias y no prueban generalización a recorridos nuevos. La [revisión técnica](docs/REVIEW.md) detalla la evidencia.

<details>
<summary>Ver matrices de confusión de la CNN supervisada</summary>

![Evaluación supervisada archivada](media/supervised-cnn-confusion.png)

Se conserva el título original de la figura; la partición real es por ventanas.

</details>

## Qué contiene

- 45.062 filas de metadatos: 32.218 ventanas naturales, 10.840 controladas y 2.004 aumentadas.
- 32.218 predicciones georreferenciadas: 31.084 normales, 812 baches, 283 frenadas y 39 esquivadas.
- Modelos MLP de 72 características y CNN de 128 × 6.
- Un modelo INT8 TFLite de 232.656 bytes y su representación idéntica como arreglo C.

Las grabaciones crudas se obtienen por separado mediante los enlaces del [README original](README_UPSTREAM_ES.md). El tensor completo de ventanas crudas no está incluido: para repetir el entrenamiento CNN hay que reconstruirlo desde las entradas.

## Inspección rápida

```powershell
python examples/inspect_artifacts.py
```

Inspecciona archivos locales con la biblioteca estándar de Python. No entrena, no publica mensajes ni carga archivos pickle.

## Ajustes de esta copia

Se añadieron créditos, documentación bilingüe, figuras del notebook, vista espacial local, inspector de artefactos, dependencias faltantes y configuración MQTT de ejemplo.

El firmware se organizó para Arduino IDE y se corrigió la ventana de **120 a 128 muestras** para coincidir con el modelo. También se añadieron comprobaciones de tensores y conversión de la salida INT8 a probabilidad. Los cambios se contrastaron con los artefactos, pero no se probaron en una placa física. La calibración entre smartphone y MPU-6050 y la validación en campo siguen pendientes.

## Autores

**Juan Andrés Vallejo Rozo · Andrés Felipe Rojas Aguilar · Brayan Camilo Rodríguez Diaz · Carlos David Ramírez Muñoz**

Contribuciones: [AUTHORS.md](AUTHORS.md). Se conserva la licencia MIT del proyecto original.
