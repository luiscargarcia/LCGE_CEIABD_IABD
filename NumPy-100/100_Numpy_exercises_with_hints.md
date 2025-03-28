<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
>>>>>>> luiscargarcia/main
# 100 ejercicios de numpy

Esta es una colección de ejercicios que han sido recopilados de la lista de correo de numpy, 
Stack Overflow y la documentación de numpy. El objetivo de esta colección es ofrecer una referencia
rápida tanto para usuarios nuevos como experimentados, y también proporcionar un conjunto de 
ejercicios para quienes enseñan.

Si encuentras un error o crees que tienes una mejor manera de resolver alguno de ellos,
no dudes en abrir un issue en <https://github.com/rougier/numpy-100>.
Archivo generado automáticamente. Consulta la documentación para actualizar preguntas/respuestas/pistas programáticamente.

#### 1. Importar el paquete numpy bajo el nombre `np` (★☆☆)
`hint: import … as`
#### 2. Imprimir la versión de numpy y la configuración (★☆☆)
`hint: np.__version__, np.show_config)`
#### 3. Crear un vector nulo de tamaño 10 (★☆☆)
`hint: np.zeros`
#### 4. ¿Cómo encontrar el tamaño de memoria de cualquier array? (★☆☆)
`hint: size, itemsize`
#### 5. ¿Cómo obtener la documentación de la función numpy add desde la línea de comandos? (★☆☆)
`hint: np.info`
#### 6. Crear un vector nulo de tamaño 10 pero el quinto valor que sea 1 (★☆☆)
`hint: array[4]`
#### 7. Crear un vector con valores que van de 10 a 49 (★☆☆)
`hint: arange`
#### 8. Invertir un vector (el primer elemento se convierte en el último) (★☆☆)
`hint: array[::-1]`
#### 9. Crear una matriz 3x3 con valores que van de 0 a 8 (★☆☆)
`hint: reshape`
#### 10. Encontrar los índices de los elementos no nulos de [1,2,0,0,4,0] (★☆☆)
`hint: np.nonzero`
#### 11. Crear una matriz identidad 3x3 (★☆☆)
`hint: np.eye`
#### 12. Crear una matriz 3x3x3 con valores aleatorios (★☆☆)
`hint: np.random.random`
#### 13. Crear una matriz 10x10 con valores aleatorios y encontrar los valores mínimo y máximo (★☆☆)
`hint: min, max`
#### 14. Crear un vector aleatorio de tamaño 30 y encontrar el valor medio (★☆☆)
`hint: mean`
#### 15. Crear una matriz 2D con 1 en el borde y 0 en el interior (★☆☆)
`hint: array[1:-1, 1:-1]`
#### 16. ¿Cómo agregar un borde (relleno con ceros) alrededor de una matriz existente? (★☆☆)
`hint: np.pad`
#### 17. ¿Cuál es el resultado de la siguiente expresión? (★☆☆)
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
=======
# 100 ejercicios de numpy
>>>>>>> 847158e (06-02-2025_19:20)

Esta es una colección de ejercicios que han sido recopilados de la lista de correo de numpy, 
Stack Overflow y la documentación de numpy. El objetivo de esta colección es ofrecer una referencia
rápida tanto para usuarios nuevos como experimentados, y también proporcionar un conjunto de 
ejercicios para quienes enseñan.

Si encuentras un error o crees que tienes una mejor manera de resolver alguno de ellos,
no dudes en abrir un issue en <https://github.com/rougier/numpy-100>.
Archivo generado automáticamente. Consulta la documentación para actualizar preguntas/respuestas/pistas programáticamente.

#### 1. Importar el paquete numpy bajo el nombre `np` (★☆☆)
`hint: import … as`
#### 2. Imprimir la versión de numpy y la configuración (★☆☆)
`hint: np.__version__, np.show_config)`
#### 3. Crear un vector nulo de tamaño 10 (★☆☆)
`hint: np.zeros`
#### 4. ¿Cómo encontrar el tamaño de memoria de cualquier array? (★☆☆)
`hint: size, itemsize`
#### 5. ¿Cómo obtener la documentación de la función numpy add desde la línea de comandos? (★☆☆)
`hint: np.info`
#### 6. Crear un vector nulo de tamaño 10 pero el quinto valor que sea 1 (★☆☆)
`hint: array[4]`
#### 7. Crear un vector con valores que van de 10 a 49 (★☆☆)
`hint: arange`
#### 8. Invertir un vector (el primer elemento se convierte en el último) (★☆☆)
`hint: array[::-1]`
#### 9. Crear una matriz 3x3 con valores que van de 0 a 8 (★☆☆)
`hint: reshape`
#### 10. Encontrar los índices de los elementos no nulos de [1,2,0,0,4,0] (★☆☆)
`hint: np.nonzero`
#### 11. Crear una matriz identidad 3x3 (★☆☆)
`hint: np.eye`
#### 12. Crear una matriz 3x3x3 con valores aleatorios (★☆☆)
`hint: np.random.random`
#### 13. Crear una matriz 10x10 con valores aleatorios y encontrar los valores mínimo y máximo (★☆☆)
`hint: min, max`
#### 14. Crear un vector aleatorio de tamaño 30 y encontrar el valor medio (★☆☆)
`hint: mean`
#### 15. Crear una matriz 2D con 1 en el borde y 0 en el interior (★☆☆)
`hint: array[1:-1, 1:-1]`
#### 16. ¿Cómo agregar un borde (relleno con ceros) alrededor de una matriz existente? (★☆☆)
`hint: np.pad`
<<<<<<< HEAD
#### 17. What is the result of the following expression? (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 17. ¿Cuál es el resultado de la siguiente expresión? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
>>>>>>> luiscargarcia/main
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
```
`hint: NaN = not a number, inf = infinity`
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
>>>>>>> luiscargarcia/main
#### 18. Crear una matriz 5x5 con valores 1,2,3,4 justo debajo de la diagonal (★☆☆)
`hint: np.diag`
#### 19. Crear una matriz 8x8 y llenarla con un patrón de tablero de ajedrez (★☆☆)
`hint: array[::2]`
#### 20. Considerar una matriz de forma (6,7,8), ¿cuál es el índice (x,y,z) del elemento 100? (★☆☆)
`hint: np.unravel_index`
#### 21. Crear una matriz de tablero de ajedrez 8x8 usando la función tile (★☆☆)
`hint: np.tile`
#### 22. Normalizar una matriz aleatoria 5x5 (★☆☆)
`hint: (x -mean)/std`
#### 23. Crear un dtype personalizado que describa un color como cuatro bytes sin signo (RGBA) (★☆☆)
`hint: np.dtype`
#### 24. Multiplicar una matriz 5x3 por una matriz 3x2 (producto de matrices real) (★☆☆)
`hint:`
#### 25. Dado un array 1D, negar todos los elementos que están entre 3 y 8, en su lugar. (★☆☆)
`hint: >, <`
#### 26. ¿Cuál es la salida del siguiente script? (★☆☆)
```python
# Autor: Jake VanderPlas
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
=======
#### 18. Crear una matriz 5x5 con valores 1,2,3,4 justo debajo de la diagonal (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
`hint: np.diag`
#### 19. Crear una matriz 8x8 y llenarla con un patrón de tablero de ajedrez (★☆☆)
`hint: array[::2]`
#### 20. Considerar una matriz de forma (6,7,8), ¿cuál es el índice (x,y,z) del elemento 100? (★☆☆)
`hint: np.unravel_index`
#### 21. Crear una matriz de tablero de ajedrez 8x8 usando la función tile (★☆☆)
`hint: np.tile`
#### 22. Normalizar una matriz aleatoria 5x5 (★☆☆)
`hint: (x -mean)/std`
#### 23. Crear un dtype personalizado que describa un color como cuatro bytes sin signo (RGBA) (★☆☆)
`hint: np.dtype`
#### 24. Multiplicar una matriz 5x3 por una matriz 3x2 (producto de matrices real) (★☆☆)
`hint:`
#### 25. Dado un array 1D, negar todos los elementos que están entre 3 y 8, en su lugar. (★☆☆)
`hint: >, <`
#### 26. ¿Cuál es la salida del siguiente script? (★☆☆)
```python
<<<<<<< HEAD
# Author: Jake VanderPlas
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Jake VanderPlas
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
>>>>>>> luiscargarcia/main

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```
`hint: np.sum`
<<<<<<< HEAD
#### 27. Considerar un vector entero Z, ¿cuál de estas expresiones son legales? (★☆☆)
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 27. Considerar un vector entero Z, ¿cuál de estas expresiones son legales? (★☆☆)
=======
#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 27. Considerar un vector entero Z, ¿cuál de estas expresiones son legales? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 27. Considerar un vector entero Z, ¿cuál de estas expresiones son legales? (★☆☆)
>>>>>>> origin/main
>>>>>>> luiscargarcia/main
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```
<<<<<<< HEAD
`No se proporcionan pistas...`
#### 28. ¿Cuáles son los resultados de las siguientes expresiones? (★☆☆)
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
`No se proporcionan pistas...`
#### 28. ¿Cuáles son los resultados de las siguientes expresiones? (★☆☆)
=======
`No hints provided...`
#### 28. What are the result of the following expressions? (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
`No se proporcionan pistas...`
#### 28. ¿Cuáles son los resultados de las siguientes expresiones? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
`No se proporcionan pistas...`
#### 28. ¿Cuáles son los resultados de las siguientes expresiones? (★☆☆)
>>>>>>> origin/main
>>>>>>> luiscargarcia/main
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
>>>>>>> luiscargarcia/main
`No se proporcionan pistas...`
#### 29. ¿Cómo redondear un array de flotantes alejándose de cero? (★☆☆)
`hint: np.uniform, np.copysign, np.ceil, np.abs, np.where`
#### 30. ¿Cómo encontrar valores comunes entre dos arrays? (★☆☆)
`hint: np.intersect1d`
#### 31. ¿Cómo ignorar todas las advertencias de numpy (no recomendado)? (★☆☆)
`hint: np.seterr, np.errstate`
#### 32. ¿Es la siguiente expresión verdadera? (★☆☆)
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```
`hint: número imaginario`
#### 33. ¿Cómo obtener las fechas de ayer, hoy y mañana? (★☆☆)
`hint: np.datetime64, np.timedelta64`
#### 34. ¿Cómo obtener todas las fechas correspondientes al mes de julio de 2016? (★★☆)
`hint: np.arange(dtype=datetime64['D'])`
#### 35. ¿Cómo calcular ((A+B)*(-A/2)) en su lugar (sin copia)? (★★☆)
`hint: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=)`
#### 36. Extraer la parte entera de un array aleatorio de números positivos usando 4 métodos diferentes (★★☆)
`hint: %, np.floor, astype, np.trunc`
#### 37. Crear una matriz 5x5 con valores de fila que van de 0 a 4 (★★☆)
`hint: np.arange`
#### 38. Considerar una función generadora que genera 10 enteros y usarla para construir un array (★☆☆)
`hint: np.fromiter`
#### 39. Crear un vector de tamaño 10 con valores que van de 0 a 1, ambos excluidos (★★☆)
`hint: np.linspace`
#### 40. Crear un vector aleatorio de tamaño 10 y ordenarlo (★★☆)
`hint: sort`
#### 41. ¿Cómo sumar un array pequeño más rápido que np.sum? (★★☆)
`hint: np.add.reduce`
#### 42. Considerar dos arrays aleatorios A y B, comprobar si son iguales (★★☆)
`hint: np.allclose, np.array_equal`
#### 43. Hacer un array inmutable (solo lectura) (★★☆)
`hint: flags.writeable`
#### 44. Considerar una matriz aleatoria 10x2 que representa coordenadas cartesianas, convertirlas a coordenadas polares (★★☆)
`hint: np.sqrt, np.arctan2`
#### 45. Crear un vector aleatorio de tamaño 10 y reemplazar el valor máximo por 0 (★★☆)
`hint: argmax`
#### 46. Crear un array estructurado con coordenadas `x` e `y` que cubren el área [0,1]x[0,1] (★★☆)
`hint: np.meshgrid`
#### 47. Dado dos arrays, X e Y, construir la matriz de Cauchy C (Cij =1/(xi - yj)) (★★☆)
`hint: np.subtract.outer`
#### 48. Imprimir el valor mínimo y máximo representable para cada tipo escalar de numpy (★★☆)
`hint: np.iinfo, np.finfo, eps`
#### 49. ¿Cómo imprimir todos los valores de un array? (★★☆)
`hint: np.set_printoptions`
#### 50. ¿Cómo encontrar el valor más cercano (a un escalar dado) en un vector? (★★☆)
`hint: argmin`
#### 51. Crear un array estructurado que represente una posición (x,y) y un color (r,g,b) (★★☆)
`hint: dtype`
#### 52. Considerar un vector aleatorio con forma (100,2) que representa coordenadas, encontrar distancias punto a punto (★★☆)
`hint: np.atleast_2d, T, np.sqrt`
#### 53. ¿Cómo convertir un array de flotantes (32 bits) en un entero (32 bits) en su lugar?
`hint: view and [:] =`
#### 54. ¿Cómo leer el siguiente archivo? (★★☆)
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
`No hints provided...`
#### 29. How to round away from zero a float array ? (★☆☆)
=======
`No se proporcionan pistas...`
#### 29. ¿Cómo redondear un array de flotantes alejándose de cero? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
`hint: np.uniform, np.copysign, np.ceil, np.abs, np.where`
#### 30. ¿Cómo encontrar valores comunes entre dos arrays? (★☆☆)
`hint: np.intersect1d`
#### 31. ¿Cómo ignorar todas las advertencias de numpy (no recomendado)? (★☆☆)
`hint: np.seterr, np.errstate`
#### 32. ¿Es la siguiente expresión verdadera? (★☆☆)
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```
`hint: número imaginario`
#### 33. ¿Cómo obtener las fechas de ayer, hoy y mañana? (★☆☆)
`hint: np.datetime64, np.timedelta64`
#### 34. ¿Cómo obtener todas las fechas correspondientes al mes de julio de 2016? (★★☆)
`hint: np.arange(dtype=datetime64['D'])`
#### 35. ¿Cómo calcular ((A+B)*(-A/2)) en su lugar (sin copia)? (★★☆)
`hint: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=)`
#### 36. Extraer la parte entera de un array aleatorio de números positivos usando 4 métodos diferentes (★★☆)
`hint: %, np.floor, astype, np.trunc`
#### 37. Crear una matriz 5x5 con valores de fila que van de 0 a 4 (★★☆)
`hint: np.arange`
#### 38. Considerar una función generadora que genera 10 enteros y usarla para construir un array (★☆☆)
`hint: np.fromiter`
#### 39. Crear un vector de tamaño 10 con valores que van de 0 a 1, ambos excluidos (★★☆)
`hint: np.linspace`
#### 40. Crear un vector aleatorio de tamaño 10 y ordenarlo (★★☆)
`hint: sort`
#### 41. ¿Cómo sumar un array pequeño más rápido que np.sum? (★★☆)
`hint: np.add.reduce`
#### 42. Considerar dos arrays aleatorios A y B, comprobar si son iguales (★★☆)
`hint: np.allclose, np.array_equal`
#### 43. Hacer un array inmutable (solo lectura) (★★☆)
`hint: flags.writeable`
#### 44. Considerar una matriz aleatoria 10x2 que representa coordenadas cartesianas, convertirlas a coordenadas polares (★★☆)
`hint: np.sqrt, np.arctan2`
#### 45. Crear un vector aleatorio de tamaño 10 y reemplazar el valor máximo por 0 (★★☆)
`hint: argmax`
#### 46. Crear un array estructurado con coordenadas `x` e `y` que cubren el área [0,1]x[0,1] (★★☆)
`hint: np.meshgrid`
#### 47. Dado dos arrays, X e Y, construir la matriz de Cauchy C (Cij =1/(xi - yj)) (★★☆)
`hint: np.subtract.outer`
#### 48. Imprimir el valor mínimo y máximo representable para cada tipo escalar de numpy (★★☆)
`hint: np.iinfo, np.finfo, eps`
#### 49. ¿Cómo imprimir todos los valores de un array? (★★☆)
`hint: np.set_printoptions`
#### 50. ¿Cómo encontrar el valor más cercano (a un escalar dado) en un vector? (★★☆)
`hint: argmin`
#### 51. Crear un array estructurado que represente una posición (x,y) y un color (r,g,b) (★★☆)
`hint: dtype`
#### 52. Considerar un vector aleatorio con forma (100,2) que representa coordenadas, encontrar distancias punto a punto (★★☆)
`hint: np.atleast_2d, T, np.sqrt`
#### 53. ¿Cómo convertir un array de flotantes (32 bits) en un entero (32 bits) en su lugar?
`hint: view and [:] =`
<<<<<<< HEAD
#### 54. How to read the following file? (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 54. ¿Cómo leer el siguiente archivo? (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
>>>>>>> luiscargarcia/main
```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```
`hint: np.genfromtxt`
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
>>>>>>> luiscargarcia/main
#### 55. ¿Cuál es el equivalente de enumerate para arrays de numpy? (★★☆)
`hint: np.ndenumerate, np.ndindex`
#### 56. Generar un array genérico 2D similar a una Gaussiana (★★☆)
`hint: np.meshgrid, np.exp`
#### 57. ¿Cómo colocar aleatoriamente p elementos en un array 2D? (★★☆)
`hint: np.put, np.random.choice`
#### 58. Restar la media de cada fila de una matriz (★★☆)
`hint: mean(axis=,keepdims=)`
#### 59. ¿Cómo ordenar un array por la n-ésima columna? (★★☆)
`hint: argsort`
#### 60. ¿Cómo saber si un array 2D dado tiene columnas nulas? (★★☆)
`hint: any, ~`
#### 61. Encontrar el valor más cercano a un valor dado en un array (★★☆)
`hint: np.abs, argmin, flat`
#### 62. Considerando dos arrays con forma (1,3) y (3,1), ¿cómo calcular su suma usando un iterador? (★★☆)
`hint: np.nditer`
#### 63. Crear una clase de array que tenga un atributo de nombre (★★☆)
`hint: método de clase`
#### 64. Considerar un vector dado, ¿cómo sumar 1 a cada elemento indexado por un segundo vector (tener cuidado con los índices repetidos)? (★★★)
`hint: np.bincount | np.add.at`
#### 65. ¿Cómo acumular elementos de un vector (X) en un array (F) basado en una lista de índices (I)? (★★★)
`hint: np.bincount`
#### 66. Considerando una imagen de cuatro dimensiones (dtype=ubyte), calcular el número de colores únicos (★★☆)
`hint: np.unique`
#### 67. Considerando un array de cuatro dimensiones, ¿cómo obtener la suma sobre los dos últimos ejes a la vez? (★★★)
`hint: sum(axis=(-2,-1))`
#### 68. Considerando un vector unidimensional D, ¿cómo calcular las medias de subconjuntos de D usando un vector S del mismo tamaño que describe los índices de los subconjuntos? (★★★)
`hint: np.bincount`
#### 69. ¿Cómo obtener la diagonal de un producto punto? (★★★)
`hint: np.diag`
#### 70. Considerar el vector [1, 2, 3, 4, 5], ¿cómo construir un nuevo vector con 3 ceros consecutivos intercalados entre cada valor? (★★★)
`hint: array[::4]`
#### 71. Considerar un array de dimensión (5,5,3), ¿cómo multiplicarlo por un array con dimensiones (5,5)? (★★★)
`hint: array[:, :, None]`
#### 72. ¿Cómo intercambiar dos filas de un array? (★★★)
`hint: array[[]] = array[[]]`
#### 73. Considerar un conjunto de 10 tríos que describen 10 triángulos (con vértices compartidos), encontrar el conjunto de segmentos de línea únicos que componen todos los triángulos (★★★)
`hint: repeat, np.roll, np.sort, view, np.unique`
#### 74. Dado un array ordenado C que corresponde a un bincount, ¿cómo producir un array A tal que np.bincount(A) == C? (★★★)
`hint: np.repeat`
#### 75. ¿Cómo calcular promedios usando una ventana deslizante sobre un array? (★★★)
`hint: np.cumsum, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 76. Considerar un array unidimensional Z, construir un array bidimensional cuya primera fila sea (Z[0],Z[1],Z[2]) y cada fila subsiguiente se desplace en 1 (la última fila debe ser (Z[-3],Z[-2],Z[-1]) (★★★)
`hint: from numpy.lib import stride_tricks, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 77. ¿Cómo negar un booleano o cambiar el signo de un flotante en su lugar? (★★★)
`hint: np.logical_not, np.negative`
#### 78. Considerar 2 conjuntos de puntos P0,P1 que describen líneas (2D) y un punto p, ¿cómo calcular la distancia desde p a cada línea i (P0[i],P1[i])? (★★★)
`No se proporcionan pistas...`
#### 79. Considerar 2 conjuntos de puntos P0,P1 que describen líneas (2D) y un conjunto de puntos P, ¿cómo calcular la distancia desde cada punto j (P[j]) a cada línea i (P0[i],P1[i])? (★★★)
`No se proporcionan pistas...`
#### 80. Considerar un array arbitrario, escribir una función que extraiga una subparte con una forma fija y centrada en un elemento dado (rellenar con un valor `fill` cuando sea necesario) (★★★)
`hint: mínimo máximo`
#### 81. Considerar un array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ¿cómo generar un array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)
`hint: stride_tricks.as_strided, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 82. Calcular el rango de una matriz (★★★)
`hint: np.linalg.svd, np.linalg.matrix_rank`
#### 83. ¿Cómo encontrar el valor más frecuente en un array?
`hint: np.bincount, argmax`
#### 84. Extraer todos los bloques contiguos 3x3 de una matriz aleatoria 10x10 (★★★)
`hint: stride_tricks.as_strided, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 85. Crear una subclase de array 2D tal que Z[i,j] == Z[j,i] (★★★)
`hint: método de clase`
#### 86. Considerar un conjunto de p matrices con forma (n,n) y un conjunto de p vectores con forma (n,1). ¿Cómo calcular la suma de los productos de matrices p a la vez? (el resultado tiene forma (n,1)) (★★★)
`hint: np.tensordot`
#### 87. Considerar un array 16x16, ¿cómo obtener la suma de bloques (el tamaño del bloque es 4x4)? (★★★)
`hint: np.add.reduceat, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 88. ¿Cómo implementar el Juego de la Vida usando arrays de numpy? (★★★)
`No se proporcionan pistas...`
#### 89. ¿Cómo obtener los n valores más grandes de un array (★★★)
`hint: np.argsort | np.argpartition`
#### 90. Dado un número arbitrario de vectores, construir el producto cartesiano (todas las combinaciones de cada elemento) (★★★)
`hint: np.indices`
#### 91. ¿Cómo crear un array de registros a partir de un array regular? (★★★)
`hint: np.core.records.fromarrays`
#### 92. Considerar un vector grande Z, calcular Z a la potencia de 3 usando 3 métodos diferentes (★★★)
`hint: np.power, *, np.einsum`
#### 93. Considerar dos arrays A y B de forma (8,3) y (2,2). ¿Cómo encontrar filas de A que contengan elementos de cada fila de B sin importar el orden de los elementos en B? (★★★)
`hint: np.where`
#### 94. Considerando una matriz 10x3, extraer filas con valores desiguales (por ejemplo, [2,2,3]) (★★★)
`No se proporcionan pistas...`
#### 95. Convertir un vector de enteros en una representación binaria de matriz (★★★)
`hint: np.unpackbits`
#### 96. Dado un array bidimensional, ¿cómo extraer filas únicas? (★★★)
`hint: np.ascontiguousarray | np.unique`
#### 97. Considerando 2 vectores A y B, escribir el equivalente de einsum de las funciones inner, outer, sum y mul (★★★)
`hint: np.einsum`
#### 98. Considerando un camino descrito por dos vectores (X,Y), ¿cómo muestrearlo usando muestras equidistantes (★★★)?
`hint: np.cumsum, np.interp`
#### 99. Dado un entero n y un array 2D X, seleccionar de X las filas que pueden interpretarse como extracciones de una distribución multinomial con n grados, es decir, las filas que solo contienen enteros y que suman n. (★★★)
`hint: np.logical_and.reduce, np.mod`
#### 100. Calcular intervalos de confianza del 95% bootstrap para la media de un array 1D X (es decir, re-muestrear los elementos de un array con reemplazo N veces, calcular la media de cada muestra y luego calcular percentiles sobre las medias). (★★★)
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)
=======
#### 55. ¿Cuál es el equivalente de enumerate para arrays de numpy? (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
`hint: np.ndenumerate, np.ndindex`
#### 56. Generar un array genérico 2D similar a una Gaussiana (★★☆)
`hint: np.meshgrid, np.exp`
#### 57. ¿Cómo colocar aleatoriamente p elementos en un array 2D? (★★☆)
`hint: np.put, np.random.choice`
#### 58. Restar la media de cada fila de una matriz (★★☆)
`hint: mean(axis=,keepdims=)`
#### 59. ¿Cómo ordenar un array por la n-ésima columna? (★★☆)
`hint: argsort`
#### 60. ¿Cómo saber si un array 2D dado tiene columnas nulas? (★★☆)
`hint: any, ~`
#### 61. Encontrar el valor más cercano a un valor dado en un array (★★☆)
`hint: np.abs, argmin, flat`
#### 62. Considerando dos arrays con forma (1,3) y (3,1), ¿cómo calcular su suma usando un iterador? (★★☆)
`hint: np.nditer`
#### 63. Crear una clase de array que tenga un atributo de nombre (★★☆)
`hint: método de clase`
#### 64. Considerar un vector dado, ¿cómo sumar 1 a cada elemento indexado por un segundo vector (tener cuidado con los índices repetidos)? (★★★)
`hint: np.bincount | np.add.at`
#### 65. ¿Cómo acumular elementos de un vector (X) en un array (F) basado en una lista de índices (I)? (★★★)
`hint: np.bincount`
#### 66. Considerando una imagen de cuatro dimensiones (dtype=ubyte), calcular el número de colores únicos (★★☆)
`hint: np.unique`
#### 67. Considerando un array de cuatro dimensiones, ¿cómo obtener la suma sobre los dos últimos ejes a la vez? (★★★)
`hint: sum(axis=(-2,-1))`
#### 68. Considerando un vector unidimensional D, ¿cómo calcular las medias de subconjuntos de D usando un vector S del mismo tamaño que describe los índices de los subconjuntos? (★★★)
`hint: np.bincount`
#### 69. ¿Cómo obtener la diagonal de un producto punto? (★★★)
`hint: np.diag`
#### 70. Considerar el vector [1, 2, 3, 4, 5], ¿cómo construir un nuevo vector con 3 ceros consecutivos intercalados entre cada valor? (★★★)
`hint: array[::4]`
#### 71. Considerar un array de dimensión (5,5,3), ¿cómo multiplicarlo por un array con dimensiones (5,5)? (★★★)
`hint: array[:, :, None]`
#### 72. ¿Cómo intercambiar dos filas de un array? (★★★)
`hint: array[[]] = array[[]]`
#### 73. Considerar un conjunto de 10 tríos que describen 10 triángulos (con vértices compartidos), encontrar el conjunto de segmentos de línea únicos que componen todos los triángulos (★★★)
`hint: repeat, np.roll, np.sort, view, np.unique`
#### 74. Dado un array ordenado C que corresponde a un bincount, ¿cómo producir un array A tal que np.bincount(A) == C? (★★★)
`hint: np.repeat`
#### 75. ¿Cómo calcular promedios usando una ventana deslizante sobre un array? (★★★)
`hint: np.cumsum, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 76. Considerar un array unidimensional Z, construir un array bidimensional cuya primera fila sea (Z[0],Z[1],Z[2]) y cada fila subsiguiente se desplace en 1 (la última fila debe ser (Z[-3],Z[-2],Z[-1]) (★★★)
`hint: from numpy.lib import stride_tricks, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 77. ¿Cómo negar un booleano o cambiar el signo de un flotante en su lugar? (★★★)
`hint: np.logical_not, np.negative`
#### 78. Considerar 2 conjuntos de puntos P0,P1 que describen líneas (2D) y un punto p, ¿cómo calcular la distancia desde p a cada línea i (P0[i],P1[i])? (★★★)
`No se proporcionan pistas...`
#### 79. Considerar 2 conjuntos de puntos P0,P1 que describen líneas (2D) y un conjunto de puntos P, ¿cómo calcular la distancia desde cada punto j (P[j]) a cada línea i (P0[i],P1[i])? (★★★)
`No se proporcionan pistas...`
#### 80. Considerar un array arbitrario, escribir una función que extraiga una subparte con una forma fija y centrada en un elemento dado (rellenar con un valor `fill` cuando sea necesario) (★★★)
`hint: mínimo máximo`
#### 81. Considerar un array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ¿cómo generar un array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)
`hint: stride_tricks.as_strided, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 82. Calcular el rango de una matriz (★★★)
`hint: np.linalg.svd, np.linalg.matrix_rank`
#### 83. ¿Cómo encontrar el valor más frecuente en un array?
`hint: np.bincount, argmax`
#### 84. Extraer todos los bloques contiguos 3x3 de una matriz aleatoria 10x10 (★★★)
`hint: stride_tricks.as_strided, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 85. Crear una subclase de array 2D tal que Z[i,j] == Z[j,i] (★★★)
`hint: método de clase`
#### 86. Considerar un conjunto de p matrices con forma (n,n) y un conjunto de p vectores con forma (n,1). ¿Cómo calcular la suma de los productos de matrices p a la vez? (el resultado tiene forma (n,1)) (★★★)
`hint: np.tensordot`
#### 87. Considerar un array 16x16, ¿cómo obtener la suma de bloques (el tamaño del bloque es 4x4)? (★★★)
`hint: np.add.reduceat, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 88. ¿Cómo implementar el Juego de la Vida usando arrays de numpy? (★★★)
`No se proporcionan pistas...`
#### 89. ¿Cómo obtener los n valores más grandes de un array (★★★)
`hint: np.argsort | np.argpartition`
#### 90. Dado un número arbitrario de vectores, construir el producto cartesiano (todas las combinaciones de cada elemento) (★★★)
`hint: np.indices`
#### 91. ¿Cómo crear un array de registros a partir de un array regular? (★★★)
`hint: np.core.records.fromarrays`
#### 92. Considerar un vector grande Z, calcular Z a la potencia de 3 usando 3 métodos diferentes (★★★)
`hint: np.power, *, np.einsum`
#### 93. Considerar dos arrays A y B de forma (8,3) y (2,2). ¿Cómo encontrar filas de A que contengan elementos de cada fila de B sin importar el orden de los elementos en B? (★★★)
`hint: np.where`
#### 94. Considerando una matriz 10x3, extraer filas con valores desiguales (por ejemplo, [2,2,3]) (★★★)
`No se proporcionan pistas...`
#### 95. Convertir un vector de enteros en una representación binaria de matriz (★★★)
`hint: np.unpackbits`
#### 96. Dado un array bidimensional, ¿cómo extraer filas únicas? (★★★)
`hint: np.ascontiguousarray | np.unique`
#### 97. Considerando 2 vectores A y B, escribir el equivalente de einsum de las funciones inner, outer, sum y mul (★★★)
`hint: np.einsum`
#### 98. Considerando un camino descrito por dos vectores (X,Y), ¿cómo muestrearlo usando muestras equidistantes (★★★)?
`hint: np.cumsum, np.interp`
#### 99. Dado un entero n y un array 2D X, seleccionar de X las filas que pueden interpretarse como extracciones de una distribución multinomial con n grados, es decir, las filas que solo contienen enteros y que suman n. (★★★)
`hint: np.logical_and.reduce, np.mod`
<<<<<<< HEAD
#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 100. Calcular intervalos de confianza del 95% bootstrap para la media de un array 1D X (es decir, re-muestrear los elementos de un array con reemplazo N veces, calcular la media de cada muestra y luego calcular percentiles sobre las medias). (★★★)
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
>>>>>>> luiscargarcia/main
`hint: np.percentile`