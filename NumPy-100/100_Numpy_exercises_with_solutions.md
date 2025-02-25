<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
# 100 ejercicios de numpy

Esta es una colección de ejercicios que han sido recopilados de la lista de correo de numpy, 
Stack Overflow y la documentación de numpy. El objetivo de esta colección es ofrecer una referencia
rápida tanto para usuarios nuevos como experimentados, y también proporcionar un conjunto de 
ejercicios para quienes enseñan.

Si encuentras un error o crees que tienes una mejor manera de resolver alguno de ellos,
no dudes en abrir un issue en <https://github.com/rougier/numpy-100>.
Archivo generado automáticamente. Consulta la documentación para actualizar preguntas/respuestas/pistas programáticamente.

#### 1. Importar el paquete numpy bajo el nombre `np` (★☆☆)
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

<<<<<<< HEAD
# 100 numpy exercises

This is a collection of exercises that have been collected in the numpy mailing list, on stack overflow
and in the numpy documentation. The goal of this collection is to offer a quick reference for both old
and new users but also to provide a set of exercises for those who teach.


If you find an error or think you've a better way to solve some of them, feel
free to open an issue at <https://github.com/rougier/numpy-100>.
File automatically generated. See the documentation to update questions/answers/hints programmatically.

#### 1. Import the numpy package under the name `np` (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 1. Importar el paquete numpy bajo el nombre `np` (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main


```python
import numpy as np
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 2. Imprimir la versión de numpy y la configuración (★☆☆)
=======
#### 2. Print the numpy version and the configuration (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 2. Imprimir la versión de numpy y la configuración (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 2. Imprimir la versión de numpy y la configuración (★☆☆)
>>>>>>> origin/main


```python
print(np.__version__)
np.show_config()
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 3. Crear un vector nulo de tamaño 10 (★☆☆)


#### 4. ¿Cómo encontrar el tamaño de memoria de cualquier array? (★☆☆)
<<<<<<< HEAD
=======
#### 3. Create a null vector of size 10 (★☆☆)


```python
Z = np.zeros(10)
print(Z)
```
#### 4. How to find the memory size of any array (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 3. Crear un vector nulo de tamaño 10 (★☆☆)


#### 4. ¿Cómo encontrar el tamaño de memoria de cualquier array? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main


```python
Z = np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 5. ¿Cómo obtener la documentación de la función numpy add desde la línea de comandos? (★☆☆)
=======
#### 5. How to get the documentation of the numpy add function from the command line? (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 5. ¿Cómo obtener la documentación de la función numpy add desde la línea de comandos? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 5. ¿Cómo obtener la documentación de la función numpy add desde la línea de comandos? (★☆☆)
>>>>>>> origin/main


```python
%run `python -c "import numpy; numpy.info(numpy.add)"`
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 6. Crea un vector nulo de tamaño 10 pero el quinto valor que sea 1 (★☆☆)
=======
#### 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 6. Crea un vector nulo de tamaño 10 pero el quinto valor que sea 1 (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 6. Crea un vector nulo de tamaño 10 pero el quinto valor que sea 1 (★☆☆)
>>>>>>> origin/main


```python
Z = np.zeros(10)
Z[4] = 1
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 7. Crea un vector con valores que van de 10 a 49 (★☆☆)
=======
#### 7. Create a vector with values ranging from 10 to 49 (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 7. Crea un vector con valores que van de 10 a 49 (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 7. Crea un vector con valores que van de 10 a 49 (★☆☆)
>>>>>>> origin/main


```python
Z = np.arange(10,50)
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 8. Invierte un vector (el primer elemento se convierte en el último) (★☆☆)
=======
#### 8. Reverse a vector (first element becomes last) (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 8. Invierte un vector (el primer elemento se convierte en el último) (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 8. Invierte un vector (el primer elemento se convierte en el último) (★☆☆)
>>>>>>> origin/main


```python
Z = np.arange(50)
Z = Z[::-1]
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 9. Crea una matriz 3x3 con valores que van de 0 a 8 (★☆☆)
=======
#### 9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 9. Crea una matriz 3x3 con valores que van de 0 a 8 (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 9. Crea una matriz 3x3 con valores que van de 0 a 8 (★☆☆)
>>>>>>> origin/main


```python
Z = np.arange(9).reshape(3, 3)
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
```python
Z = np.arange(9).reshape(3, 3)
print(Z)
```
#### 10. Encuentra los índices de los elementos no nulos de [1,2,0,0,4,0] (★☆☆)
<<<<<<< HEAD
<<<<<<< HEAD
=======
#### 10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main


```python
nz = np.nonzero([1,2,0,0,4,0])
print(nz)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 11. Crea una matriz identidad de 3x3 (★☆☆)
=======
#### 11. Create a 3x3 identity matrix (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 11. Crea una matriz identidad de 3x3 (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 11. Crea una matriz identidad de 3x3 (★☆☆)
>>>>>>> origin/main


```python
Z = np.eye(3)
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 12. Crea un array 3x3x3 con valores aleatorios (★☆☆)
=======
#### 12. Create a 3x3x3 array with random values (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 12. Crea un array 3x3x3 con valores aleatorios (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 12. Crea un array 3x3x3 con valores aleatorios (★☆☆)
>>>>>>> origin/main


```python
Z = np.random.random((3,3,3))
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 13. Crea un array 10x10 con valores aleatorios y encuentra los valores mínimo y máximo (★☆☆)
=======
#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 13. Crea un array 10x10 con valores aleatorios y encuentra los valores mínimo y máximo (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 13. Crea un array 10x10 con valores aleatorios y encuentra los valores mínimo y máximo (★☆☆)
>>>>>>> origin/main


```python
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 14. Crea un vector aleatorio de tamaño 30 y encuentra el valor medio (★☆☆)
=======
#### 14. Create a random vector of size 30 and find the mean value (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 14. Crea un vector aleatorio de tamaño 30 y encuentra el valor medio (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 14. Crea un vector aleatorio de tamaño 30 y encuentra el valor medio (★☆☆)
>>>>>>> origin/main


```python
Z = np.random.random(30)
m = Z.mean()
print(m)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 15. Crea un array 2D con 1 en el borde y 0 en el interior (★☆☆)
=======
#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 15. Crea un array 2D con 1 en el borde y 0 en el interior (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 15. Crea un array 2D con 1 en el borde y 0 en el interior (★☆☆)
>>>>>>> origin/main


```python
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 16. ¿Cómo añadir un borde (relleno con 0's) alrededor de un array existente? (★☆☆)
=======
#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 16. ¿Cómo añadir un borde (relleno con 0's) alrededor de un array existente? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 16. ¿Cómo añadir un borde (relleno con 0's) alrededor de un array existente? (★☆☆)
>>>>>>> origin/main


```python
Z = np.ones((5,5))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
print(Z)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Usando indexación avanzada
=======
# Using fancy indexing
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Usando indexación avanzada
>>>>>>> 847158e (06-02-2025_19:20)
=======
# Usando indexación avanzada
>>>>>>> origin/main
Z[:, [0, -1]] = 0
Z[[0, -1], :] = 0
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 17. ¿Cuál es el resultado de la siguiente expresión? (★☆☆)
=======
#### 17. What is the result of the following expression? (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 17. ¿Cuál es el resultado de la siguiente expresión? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 17. ¿Cuál es el resultado de la siguiente expresión? (★☆☆)
>>>>>>> origin/main
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
```


```python
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 18. Crea una matriz 5x5 con valores 1,2,3,4 justo debajo de la diagonal (★☆☆)
=======
#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 18. Crea una matriz 5x5 con valores 1,2,3,4 justo debajo de la diagonal (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 18. Crea una matriz 5x5 con valores 1,2,3,4 justo debajo de la diagonal (★☆☆)
>>>>>>> origin/main


```python
Z = np.diag(1+np.arange(4),k=-1)
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 19. Crea una matriz 8x8 y rellénala con un patrón de tablero de ajedrez (★☆☆)
=======
#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 19. Crea una matriz 8x8 y rellénala con un patrón de tablero de ajedrez (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 19. Crea una matriz 8x8 y rellénala con un patrón de tablero de ajedrez (★☆☆)
>>>>>>> origin/main


```python
Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 20. Considera un array de forma (6,7,8), ¿cuál es el índice (x,y,z) del elemento 100? (★☆☆)
=======
#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 20. Considera un array de forma (6,7,8), ¿cuál es el índice (x,y,z) del elemento 100? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 20. Considera un array de forma (6,7,8), ¿cuál es el índice (x,y,z) del elemento 100? (★☆☆)
>>>>>>> origin/main


```python
print(np.unravel_index(99,(6,7,8)))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 21. Crea un tablero de ajedrez 8x8 usando la función tile (★☆☆)
=======
#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 21. Crea un tablero de ajedrez 8x8 usando la función tile (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 21. Crea un tablero de ajedrez 8x8 usando la función tile (★☆☆)
>>>>>>> origin/main


```python
Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 22. Normaliza una matriz aleatoria 5x5 (★☆☆)
=======
#### 22. Normalize a 5x5 random matrix (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 22. Normaliza una matriz aleatoria 5x5 (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 22. Normaliza una matriz aleatoria 5x5 (★☆☆)
>>>>>>> origin/main


```python
Z = np.random.random((5,5))
Z = (Z - np.mean (Z)) / (np.std (Z))
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 23. Crea un dtype personalizado que describa un color como cuatro bytes sin signo (RGBA) (★☆☆)
=======
#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 23. Crea un dtype personalizado que describa un color como cuatro bytes sin signo (RGBA) (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 23. Crea un dtype personalizado que describa un color como cuatro bytes sin signo (RGBA) (★☆☆)
>>>>>>> origin/main


```python
color = np.dtype([("r", np.ubyte),
                  ("g", np.ubyte),
                  ("b", np.ubyte),
                  ("a", np.ubyte)])
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 24. Multiplica una matriz 5x3 por una matriz 3x2 (producto de matrices reales) (★☆☆)
=======
#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 24. Multiplica una matriz 5x3 por una matriz 3x2 (producto de matrices reales) (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 24. Multiplica una matriz 5x3 por una matriz 3x2 (producto de matrices reales) (★☆☆)
>>>>>>> origin/main


```python
Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
# Solución alternativa, en Python 3.5 y superior
Z = np.ones((5,3)) @ np.ones((3,2))
print(Z)
```
#### 25. Dado un array 1D, niega todos los elementos que están entre 3 y 8, en su lugar. (★☆☆)


```python
# Autor: Evgeni Burovski
<<<<<<< HEAD
=======
# Alternative solution, in Python 3.5 and above
=======
# Solución alternativa, en Python 3.5 y superior
>>>>>>> 847158e (06-02-2025_19:20)
Z = np.ones((5,3)) @ np.ones((3,2))
print(Z)
```
#### 25. Dado un array 1D, niega todos los elementos que están entre 3 y 8, en su lugar. (★☆☆)


```python
<<<<<<< HEAD
# Author: Evgeni Burovski
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Evgeni Burovski
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

Z = np.arange(11)
Z[(3 < Z) & (Z < 8)] *= -1
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 26. ¿Cuál es la salida del siguiente script? (★☆☆)
```python
# Autor: Jake VanderPlas
=======
#### 26. What is the output of the following script? (★☆☆)
```python
# Author: Jake VanderPlas
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 26. ¿Cuál es la salida del siguiente script? (★☆☆)
```python
# Autor: Jake VanderPlas
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 26. ¿Cuál es la salida del siguiente script? (★☆☆)
```python
# Autor: Jake VanderPlas
>>>>>>> origin/main

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```


```python
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Autor: Jake VanderPlas
=======
# Author: Jake VanderPlas
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Jake VanderPlas
>>>>>>> 847158e (06-02-2025_19:20)
=======
# Autor: Jake VanderPlas
>>>>>>> origin/main

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 27. Considera un vector entero Z, ¿cuál de estas expresiones son legales? (★☆☆)
=======
#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 27. Considera un vector entero Z, ¿cuál de estas expresiones son legales? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 27. Considera un vector entero Z, ¿cuál de estas expresiones son legales? (★☆☆)
>>>>>>> origin/main
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```


```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 28. ¿Cuáles son los resultados de las siguientes expresiones? (★☆☆)
=======
#### 28. What are the result of the following expressions? (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 28. ¿Cuáles son los resultados de las siguientes expresiones? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 28. ¿Cuáles son los resultados de las siguientes expresiones? (★☆☆)
>>>>>>> origin/main
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```


```python
print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 29. ¿Cómo redondear un array de flotantes alejándose de cero? (★☆☆)


```python
# Autor: Charles R Harris
<<<<<<< HEAD
=======
#### 29. How to round away from zero a float array ? (★☆☆)


```python
# Author: Charles R Harris
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 29. ¿Cómo redondear un array de flotantes alejándose de cero? (★☆☆)


```python
# Autor: Charles R Harris
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

Z = np.random.uniform(-10,+10,10)
print(np.copysign(np.ceil(np.abs(Z)), Z))

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
# Más legible pero menos eficiente
print(np.where(Z>0, np.ceil(Z), np.floor(Z)))
```
#### 30. ¿Cómo encontrar valores comunes entre dos arrays? (★☆☆)
<<<<<<< HEAD
=======
# More readable but less efficient
print(np.where(Z>0, np.ceil(Z), np.floor(Z)))
```
#### 30. How to find common values between two arrays? (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Más legible pero menos eficiente
print(np.where(Z>0, np.ceil(Z), np.floor(Z)))
```
#### 30. ¿Cómo encontrar valores comunes entre dos arrays? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main


```python
Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 31. ¿Cómo ignorar todas las advertencias de numpy (no recomendado)? (★☆☆)


```python
# Modo suicida activado
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Volver a la cordura
_ = np.seterr(**defaults)

# Equivalente con un gestor de contexto
with np.errstate(all="ignore"):
    np.arange(3) / 0
```
#### 32. ¿Es verdadera la siguiente expresión? (★☆☆)
<<<<<<< HEAD
=======
#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)
=======
#### 31. ¿Cómo ignorar todas las advertencias de numpy (no recomendado)? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)


```python
# Modo suicida activado
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Volver a la cordura
_ = np.seterr(**defaults)

# Equivalente con un gestor de contexto
with np.errstate(all="ignore"):
    np.arange(3) / 0
```
<<<<<<< HEAD
#### 32. Is the following expressions true? (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 32. ¿Es verdadera la siguiente expresión? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```


```python
np.sqrt(-1) == np.emath.sqrt(-1)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 33. ¿Cómo obtener las fechas de ayer, hoy y mañana? (★☆☆)
=======
#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 33. ¿Cómo obtener las fechas de ayer, hoy y mañana? (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 33. ¿Cómo obtener las fechas de ayer, hoy y mañana? (★☆☆)
>>>>>>> origin/main


```python
yesterday = np.datetime64('today') - np.timedelta64(1)
today     = np.datetime64('today')
tomorrow  = np.datetime64('today') + np.timedelta64(1)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 34. ¿Cómo obtener todas las fechas correspondientes al mes de julio de 2016? (★★☆)
=======
#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 34. ¿Cómo obtener todas las fechas correspondientes al mes de julio de 2016? (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 34. ¿Cómo obtener todas las fechas correspondientes al mes de julio de 2016? (★★☆)
>>>>>>> origin/main


```python
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 35. ¿Cómo calcular ((A+B)*(-A/2)) en su lugar (sin copia)? (★★☆)
=======
#### 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 35. ¿Cómo calcular ((A+B)*(-A/2)) en su lugar (sin copia)? (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 35. ¿Cómo calcular ((A+B)*(-A/2)) en su lugar (sin copia)? (★★☆)
>>>>>>> origin/main


```python
A = np.ones(3)*1
B = np.ones(3)*2
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 36. Extrae la parte entera de un array aleatorio de números positivos usando 4 métodos diferentes (★★☆)
=======
#### 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 36. Extrae la parte entera de un array aleatorio de números positivos usando 4 métodos diferentes (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 36. Extrae la parte entera de un array aleatorio de números positivos usando 4 métodos diferentes (★★☆)
>>>>>>> origin/main


```python
Z = np.random.uniform(0,10,10)

print(Z - Z%1)
print(Z // 1)
print(np.floor(Z))
print(Z.astype(int))
print(np.trunc(Z))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 37. Crea una matriz 5x5 con valores de fila que van de 0 a 4 (★★☆)
=======
#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 37. Crea una matriz 5x5 con valores de fila que van de 0 a 4 (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 37. Crea una matriz 5x5 con valores de fila que van de 0 a 4 (★★☆)
>>>>>>> origin/main


```python
Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
# sin broadcasting
Z = np.tile(np.arange(0, 5), (5,1))
print(Z)
```
#### 38. Considera una función generadora que genera 10 enteros y úsala para construir un array (★☆☆)
<<<<<<< HEAD
=======
# without broadcasting
Z = np.tile(np.arange(0, 5), (5,1))
print(Z)
```
#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# sin broadcasting
Z = np.tile(np.arange(0, 5), (5,1))
print(Z)
```
#### 38. Considera una función generadora que genera 10 enteros y úsala para construir un array (★☆☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main


```python
def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 39. Crea un vector de tamaño 10 con valores que van de 0 a 1, ambos excluidos (★★☆)
=======
#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 39. Crea un vector de tamaño 10 con valores que van de 0 a 1, ambos excluidos (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 39. Crea un vector de tamaño 10 con valores que van de 0 a 1, ambos excluidos (★★☆)
>>>>>>> origin/main


```python
Z = np.linspace(0,1,11,endpoint=False)[1:]
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 40. Crea un vector aleatorio de tamaño 10 y ordénalo (★★☆)
=======
#### 40. Create a random vector of size 10 and sort it (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 40. Crea un vector aleatorio de tamaño 10 y ordénalo (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 40. Crea un vector aleatorio de tamaño 10 y ordénalo (★★☆)
>>>>>>> origin/main


```python
Z = np.random.random(10)
Z.sort()
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 41. ¿Cómo sumar un array pequeño más rápido que np.sum? (★★☆)


```python
# Autor: Evgeni Burovski
<<<<<<< HEAD
=======
#### 41. How to sum a small array faster than np.sum? (★★☆)


```python
# Author: Evgeni Burovski
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 41. ¿Cómo sumar un array pequeño más rápido que np.sum? (★★☆)


```python
# Autor: Evgeni Burovski
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

Z = np.arange(10)
np.add.reduce(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 42. Considera dos arrays aleatorios A y B, verifica si son iguales (★★☆)
=======
#### 42. Consider two random array A and B, check if they are equal (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 42. Considera dos arrays aleatorios A y B, verifica si son iguales (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 42. Considera dos arrays aleatorios A y B, verifica si son iguales (★★☆)
>>>>>>> origin/main


```python
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
# Asumiendo forma idéntica de los arrays y una tolerancia para la comparación de valores
equal = np.allclose(A,B)
print(equal)

# Verificando tanto la forma como los valores de los elementos, sin tolerancia (los valores deben ser exactamente iguales)
equal = np.array_equal(A,B)
print(equal)
```
#### 43. Haz un array inmutable (solo lectura) (★★☆)
<<<<<<< HEAD
=======
# Assuming identical shape of the arrays and a tolerance for the comparison of values
=======
# Asumiendo forma idéntica de los arrays y una tolerancia para la comparación de valores
>>>>>>> 847158e (06-02-2025_19:20)
equal = np.allclose(A,B)
print(equal)

# Verificando tanto la forma como los valores de los elementos, sin tolerancia (los valores deben ser exactamente iguales)
equal = np.array_equal(A,B)
print(equal)
```
<<<<<<< HEAD
#### 43. Make an array immutable (read-only) (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 43. Haz un array inmutable (solo lectura) (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main


```python
Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 44. Considera una matriz aleatoria 10x2 que representa coordenadas cartesianas, conviértelas a coordenadas polares (★★☆)
=======
#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 44. Considera una matriz aleatoria 10x2 que representa coordenadas cartesianas, conviértelas a coordenadas polares (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 44. Considera una matriz aleatoria 10x2 que representa coordenadas cartesianas, conviértelas a coordenadas polares (★★☆)
>>>>>>> origin/main


```python
Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 45. Crea un vector aleatorio de tamaño 10 y reemplaza el valor máximo por 0 (★★☆)
=======
#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 45. Crea un vector aleatorio de tamaño 10 y reemplaza el valor máximo por 0 (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 45. Crea un vector aleatorio de tamaño 10 y reemplaza el valor máximo por 0 (★★☆)
>>>>>>> origin/main


```python
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 46. Crea un array estructurado con coordenadas `x` y `y` que cubran el área [0,1]x[0,1] (★★☆)
=======
#### 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 46. Crea un array estructurado con coordenadas `x` y `y` que cubran el área [0,1]x[0,1] (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 46. Crea un array estructurado con coordenadas `x` y `y` que cubran el área [0,1]x[0,1] (★★☆)
>>>>>>> origin/main


```python
Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 47. Dado dos arrays, X e Y, construye la matriz de Cauchy C (Cij =1/(xi - yj)) (★★☆)


```python
# Autor: Evgeni Burovski
<<<<<<< HEAD
=======
#### 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (★★☆)


```python
# Author: Evgeni Burovski
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 47. Dado dos arrays, X e Y, construye la matriz de Cauchy C (Cij =1/(xi - yj)) (★★☆)


```python
# Autor: Evgeni Burovski
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 48. Imprime el valor mínimo y máximo representable para cada tipo escalar de numpy (★★☆)
=======
#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 48. Imprime el valor mínimo y máximo representable para cada tipo escalar de numpy (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 48. Imprime el valor mínimo y máximo representable para cada tipo escalar de numpy (★★☆)
>>>>>>> origin/main


```python
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 49. ¿Cómo imprimir todos los valores de un array? (★★☆)
=======
#### 49. How to print all the values of an array? (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 49. ¿Cómo imprimir todos los valores de un array? (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 49. ¿Cómo imprimir todos los valores de un array? (★★☆)
>>>>>>> origin/main


```python
np.set_printoptions(threshold=float("inf"))
Z = np.zeros((40,40))
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 50. ¿Cómo encontrar el valor más cercano (a un escalar dado) en un vector? (★★☆)
=======
#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 50. ¿Cómo encontrar el valor más cercano (a un escalar dado) en un vector? (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 50. ¿Cómo encontrar el valor más cercano (a un escalar dado) en un vector? (★★☆)
>>>>>>> origin/main


```python
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 51. Crea un array estructurado que represente una posición (x,y) y un color (r,g,b) (★★☆)
=======
#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 51. Crea un array estructurado que represente una posición (x,y) y un color (r,g,b) (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 51. Crea un array estructurado que represente una posición (x,y) y un color (r,g,b) (★★☆)
>>>>>>> origin/main


```python
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 52. Considera un vector aleatorio con forma (100,2) que representa coordenadas, encuentra las distancias punto a punto (★★☆)
=======
#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 52. Considera un vector aleatorio con forma (100,2) que representa coordenadas, encuentra las distancias punto a punto (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 52. Considera un vector aleatorio con forma (100,2) que representa coordenadas, encuentra las distancias punto a punto (★★☆)
>>>>>>> origin/main


```python
Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Mucho más rápido con scipy
import scipy
# Gracias Gavin Heverly-Coulson (#issue 1)
=======
# Much faster with scipy
import scipy
# Thanks Gavin Heverly-Coulson (#issue 1)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Mucho más rápido con scipy
import scipy
# Gracias Gavin Heverly-Coulson (#issue 1)
>>>>>>> 847158e (06-02-2025_19:20)
=======
# Mucho más rápido con scipy
import scipy
# Gracias Gavin Heverly-Coulson (#issue 1)
>>>>>>> origin/main
import scipy.spatial

Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)
print(D)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 53. ¿Cómo convertir un array de flotantes (32 bits) en un entero (32 bits) en su lugar?


```python
# Gracias Vikas (https://stackoverflow.com/a/10622758/5989906)
<<<<<<< HEAD
=======
#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?


```python
# Thanks Vikas (https://stackoverflow.com/a/10622758/5989906)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 53. ¿Cómo convertir un array de flotantes (32 bits) en un entero (32 bits) en su lugar?


```python
# Gracias Vikas (https://stackoverflow.com/a/10622758/5989906)
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
# & unutbu (https://stackoverflow.com/a/4396247/5989906)
Z = (np.random.rand(10)*100).astype(np.float32)
Y = Z.view(np.int32)
Y[:] = Z
print(Y)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 54. ¿Cómo leer el siguiente archivo? (★★☆)
=======
#### 54. How to read the following file? (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 54. ¿Cómo leer el siguiente archivo? (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 54. ¿Cómo leer el siguiente archivo? (★★☆)
>>>>>>> origin/main
```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```


```python
from io import StringIO

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Archivo falso
=======
# Fake file
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Archivo falso
>>>>>>> 847158e (06-02-2025_19:20)
=======
# Archivo falso
>>>>>>> origin/main
s = StringIO('''1, 2, 3, 4, 5

                6,  ,  , 7, 8

                 ,  , 9,10,11
''')
Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 55. ¿Cuál es el equivalente de enumerate para arrays de numpy? (★★☆)
=======
#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 55. ¿Cuál es el equivalente de enumerate para arrays de numpy? (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 55. ¿Cuál es el equivalente de enumerate para arrays de numpy? (★★☆)
>>>>>>> origin/main


```python
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)
for index in np.ndindex(Z.shape):
    print(index, Z[index])
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 56. Genera un array genérico 2D similar a una Gaussiana (★★☆)
=======
#### 56. Generate a generic 2D Gaussian-like array (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 56. Genera un array genérico 2D similar a una Gaussiana (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 56. Genera un array genérico 2D similar a una Gaussiana (★★☆)
>>>>>>> origin/main


```python
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 57. ¿Cómo colocar aleatoriamente p elementos en un array 2D? (★★☆)


```python
# Autor: Divakar
<<<<<<< HEAD
=======
#### 57. How to randomly place p elements in a 2D array? (★★☆)


```python
# Author: Divakar
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 57. ¿Cómo colocar aleatoriamente p elementos en un array 2D? (★★☆)


```python
# Autor: Divakar
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 58. Resta la media de cada fila de una matriz (★★☆)


```python
# Autor: Warren Weckesser

X = np.random.rand(5, 10)

# Versiones recientes de numpy
Y = X - X.mean(axis=1, keepdims=True)

# Versiones antiguas de numpy
<<<<<<< HEAD
=======
#### 58. Subtract the mean of each row of a matrix (★★☆)
=======
#### 58. Resta la media de cada fila de una matriz (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)


```python
# Autor: Warren Weckesser

X = np.random.rand(5, 10)

# Versiones recientes de numpy
Y = X - X.mean(axis=1, keepdims=True)

<<<<<<< HEAD
# Older versions of numpy
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Versiones antiguas de numpy
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
Y = X - X.mean(axis=1).reshape(-1, 1)

print(Y)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 59. ¿Cómo ordenar un array por la n-ésima columna? (★★☆)


```python
# Autor: Steve Tjoa
<<<<<<< HEAD
=======
#### 59. How to sort an array by the nth column? (★★☆)


```python
# Author: Steve Tjoa
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 59. ¿Cómo ordenar un array por la n-ésima columna? (★★☆)


```python
# Autor: Steve Tjoa
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()])
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 60. ¿Cómo saber si un array 2D dado tiene columnas nulas? (★★☆)


```python
# Autor: Warren Weckesser

# nulo : 0 
Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any())

# nulo : np.nan
<<<<<<< HEAD
=======
#### 60. How to tell if a given 2D array has null columns? (★★☆)
=======
#### 60. ¿Cómo saber si un array 2D dado tiene columnas nulas? (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)


```python
# Autor: Warren Weckesser

# nulo : 0 
Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any())

<<<<<<< HEAD
# null : np.nan
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# nulo : np.nan
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
Z=np.array([
    [0,1,np.nan],
    [1,2,np.nan],
    [4,5,np.nan]
])
print(np.isnan(Z).all(axis=0))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 61. Encuentra el valor más cercano a un valor dado en un array (★★☆)
=======
#### 61. Find the nearest value from a given value in an array (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 61. Encuentra el valor más cercano a un valor dado en un array (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 61. Encuentra el valor más cercano a un valor dado en un array (★★☆)
>>>>>>> origin/main


```python
Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 62. Considerando dos arrays con forma (1,3) y (3,1), ¿cómo calcular su suma usando un iterador? (★★☆)
=======
#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 62. Considerando dos arrays con forma (1,3) y (3,1), ¿cómo calcular su suma usando un iterador? (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 62. Considerando dos arrays con forma (1,3) y (3,1), ¿cómo calcular su suma usando un iterador? (★★☆)
>>>>>>> origin/main


```python
A = np.arange(3).reshape(3,1)
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None])
for x,y,z in it: z[...] = x + y
print(it.operands[2])
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 63. Crea una clase de array que tenga un atributo de nombre (★★☆)
=======
#### 63. Create an array class that has a name attribute (★★☆)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 63. Crea una clase de array que tenga un atributo de nombre (★★☆)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 63. Crea una clase de array que tenga un atributo de nombre (★★☆)
>>>>>>> origin/main


```python
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 64. Considera un vector dado, ¿cómo añadir 1 a cada elemento indexado por un segundo vector (ten cuidado con los índices repetidos)? (★★★)


```python
# Autor: Brett Olsen
<<<<<<< HEAD
=======
#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)


```python
# Author: Brett Olsen
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 64. Considera un vector dado, ¿cómo añadir 1 a cada elemento indexado por un segundo vector (ten cuidado con los índices repetidos)? (★★★)


```python
# Autor: Brett Olsen
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
# Otra solución
# Autor: Bartosz Telenczuk
np.add.at(Z, I, 1)
print(Z)
```
#### 65. ¿Cómo acumular elementos de un vector (X) en un array (F) basado en una lista de índices (I)? (★★★)


```python
# Autor: Alan G Isaac
<<<<<<< HEAD
=======
# Another solution
# Author: Bartosz Telenczuk
=======
# Otra solución
# Autor: Bartosz Telenczuk
>>>>>>> 847158e (06-02-2025_19:20)
np.add.at(Z, I, 1)
print(Z)
```
#### 65. ¿Cómo acumular elementos de un vector (X) en un array (F) basado en una lista de índices (I)? (★★★)


```python
<<<<<<< HEAD
# Author: Alan G Isaac
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Alan G Isaac
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 66. Considerando una imagen de (w,h,3) de (dtype=ubyte), calcula el número de colores únicos (★★☆)


```python
# Autor: Fisher Wang
<<<<<<< HEAD
=======
#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)


```python
# Author: Fisher Wang
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 66. Considerando una imagen de (w,h,3) de (dtype=ubyte), calcula el número de colores únicos (★★☆)


```python
# Autor: Fisher Wang
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

w, h = 256, 256
I = np.random.randint(0, 4, (h, w, 3)).astype(np.ubyte)
colors = np.unique(I.reshape(-1, 3), axis=0)
n = len(colors)
print(n)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Versión más rápida
# Autor: Mark Setchell
=======
# Faster version
# Author: Mark Setchell
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Versión más rápida
# Autor: Mark Setchell
>>>>>>> 847158e (06-02-2025_19:20)
=======
# Versión más rápida
# Autor: Mark Setchell
>>>>>>> origin/main
# https://stackoverflow.com/a/59671950/2836621

w, h = 256, 256
I = np.random.randint(0,4,(h,w,3), dtype=np.uint8)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
# Ver cada píxel como un solo entero de 24 bits, en lugar de tres bytes de 8 bits
I24 = np.dot(I.astype(np.uint32),[1,256,65536])

# Contar colores únicos
n = len(np.unique(I24))
print(n)
```
#### 67. Considerando un array de cuatro dimensiones, ¿cómo obtener la suma sobre los dos últimos ejes a la vez? (★★★)
<<<<<<< HEAD
=======
# View each pixel as a single 24-bit integer, rather than three 8-bit bytes
=======
# Ver cada píxel como un solo entero de 24 bits, en lugar de tres bytes de 8 bits
>>>>>>> 847158e (06-02-2025_19:20)
I24 = np.dot(I.astype(np.uint32),[1,256,65536])

# Contar colores únicos
n = len(np.unique(I24))
print(n)
```
<<<<<<< HEAD
#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 67. Considerando un array de cuatro dimensiones, ¿cómo obtener la suma sobre los dos últimos ejes a la vez? (★★★)
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main


```python
A = np.random.randint(0,10,(3,4,3,4))
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
# solución pasando una tupla de ejes (introducido en numpy 1.7.0)
sum = A.sum(axis=(-2,-1))
print(sum)
# solución aplanando las dos últimas dimensiones en una
# (útil para funciones que no aceptan tuplas para el argumento axis)
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
```
#### 68. Considerando un vector unidimensional D, ¿cómo calcular las medias de subconjuntos de D usando un vector S del mismo tamaño que describe los índices de los subconjuntos? (★★★)


```python
# Autor: Jaime Fernández del Río
<<<<<<< HEAD
=======
# solution by passing a tuple of axes (introduced in numpy 1.7.0)
=======
# solución pasando una tupla de ejes (introducido en numpy 1.7.0)
>>>>>>> 847158e (06-02-2025_19:20)
sum = A.sum(axis=(-2,-1))
print(sum)
# solución aplanando las dos últimas dimensiones en una
# (útil para funciones que no aceptan tuplas para el argumento axis)
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
```
#### 68. Considerando un vector unidimensional D, ¿cómo calcular las medias de subconjuntos de D usando un vector S del mismo tamaño que describe los índices de los subconjuntos? (★★★)


```python
<<<<<<< HEAD
# Author: Jaime Fernández del Río
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Jaime Fernández del Río
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
# Solución con Pandas como referencia debido a su código más intuitivo
import pandas as pd
print(pd.Series(D).groupby(S).mean())
```
#### 69. ¿Cómo obtener la diagonal de un producto punto? (★★★)


```python
# Autor: Mathieu Blondel
<<<<<<< HEAD
=======
# Pandas solution as a reference due to more intuitive code
=======
# Solución con Pandas como referencia debido a su código más intuitivo
>>>>>>> 847158e (06-02-2025_19:20)
import pandas as pd
print(pd.Series(D).groupby(S).mean())
```
#### 69. ¿Cómo obtener la diagonal de un producto punto? (★★★)


```python
<<<<<<< HEAD
# Author: Mathieu Blondel
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Mathieu Blondel
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
# Versión lenta
np.diag(np.dot(A, B))

# Versión rápida
np.sum(A * B.T, axis=1)

# Versión más rápida
np.einsum("ij,ji->i", A, B)
```
#### 70. Considera el vector [1, 2, 3, 4, 5], ¿cómo construir un nuevo vector con 3 ceros consecutivos intercalados entre cada valor? (★★★)


```python
# Autor: Warren Weckesser
<<<<<<< HEAD
=======
# Slow version
=======
# Versión lenta
>>>>>>> 847158e (06-02-2025_19:20)
np.diag(np.dot(A, B))

# Versión rápida
np.sum(A * B.T, axis=1)

# Versión más rápida
np.einsum("ij,ji->i", A, B)
```
#### 70. Considera el vector [1, 2, 3, 4, 5], ¿cómo construir un nuevo vector con 3 ceros consecutivos intercalados entre cada valor? (★★★)


```python
<<<<<<< HEAD
# Author: Warren Weckesser
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Warren Weckesser
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 71. Considera un array de dimensión (5,5,3), ¿cómo multiplicarlo por un array con dimensiones (5,5)? (★★★)
=======
#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 71. Considera un array de dimensión (5,5,3), ¿cómo multiplicarlo por un array con dimensiones (5,5)? (★★★)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 71. Considera un array de dimensión (5,5,3), ¿cómo multiplicarlo por un array con dimensiones (5,5)? (★★★)
>>>>>>> origin/main


```python
A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print(A * B[:,:,None])
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 72. ¿Cómo intercambiar dos filas de un array? (★★★)


```python
# Autor: Eelco Hoogendoorn
<<<<<<< HEAD
=======
#### 72. How to swap two rows of an array? (★★★)


```python
# Author: Eelco Hoogendoorn
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 72. ¿Cómo intercambiar dos filas de un array? (★★★)


```python
# Autor: Eelco Hoogendoorn
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 73. Considera un conjunto de 10 tríos que describen 10 triángulos (con vértices compartidos), encuentra el conjunto de segmentos de línea únicos que componen todos los triángulos (★★★)


```python
# Autor: Nicolas P. Rougier
<<<<<<< HEAD
=======
#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)


```python
# Author: Nicolas P. Rougier
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 73. Considera un conjunto de 10 tríos que describen 10 triángulos (con vértices compartidos), encuentra el conjunto de segmentos de línea únicos que componen todos los triángulos (★★★)


```python
# Autor: Nicolas P. Rougier
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 74. Dado un array ordenado C que corresponde a un bincount, ¿cómo producir un array A tal que np.bincount(A) == C? (★★★)


```python
# Autor: Jaime Fernández del Río
<<<<<<< HEAD
=======
#### 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)


```python
# Author: Jaime Fernández del Río
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 74. Dado un array ordenado C que corresponde a un bincount, ¿cómo producir un array A tal que np.bincount(A) == C? (★★★)


```python
# Autor: Jaime Fernández del Río
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 75. ¿Cómo calcular promedios usando una ventana deslizante sobre un array? (★★★)


```python
# Autor: Jaime Fernández del Río
<<<<<<< HEAD
=======
#### 75. How to compute averages using a sliding window over an array? (★★★)


```python
# Author: Jaime Fernández del Río
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 75. ¿Cómo calcular promedios usando una ventana deslizante sobre un array? (★★★)


```python
# Autor: Jaime Fernández del Río
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Autor: Jeff Luo (@Jeff1999)
# asegúrate de que tu NumPy >= 1.20.0
=======
# Author: Jeff Luo (@Jeff1999)
# make sure your NumPy >= 1.20.0
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Jeff Luo (@Jeff1999)
# asegúrate de que tu NumPy >= 1.20.0
>>>>>>> 847158e (06-02-2025_19:20)
=======
# Autor: Jeff Luo (@Jeff1999)
# asegúrate de que tu NumPy >= 1.20.0
>>>>>>> origin/main

from numpy.lib.stride_tricks import sliding_window_view

Z = np.arange(20)
print(sliding_window_view(Z, window_shape=3).mean(axis=-1))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 76. Considera un array unidimensional Z, construye un array bidimensional cuya primera fila sea (Z[0],Z[1],Z[2]) y cada fila subsiguiente esté desplazada en 1 (la última fila debe ser (Z[-3],Z[-2],Z[-1]) (★★★)


```python
# Autor: Joe Kington / Erik Rigtorp
<<<<<<< HEAD
=======
#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is  shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)


```python
# Author: Joe Kington / Erik Rigtorp
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 76. Considera un array unidimensional Z, construye un array bidimensional cuya primera fila sea (Z[0],Z[1],Z[2]) y cada fila subsiguiente esté desplazada en 1 (la última fila debe ser (Z[-3],Z[-2],Z[-1]) (★★★)


```python
# Autor: Joe Kington / Erik Rigtorp
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.strides[0], a.strides[0])
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Autor: Jeff Luo (@Jeff1999)
=======
# Author: Jeff Luo (@Jeff1999)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Jeff Luo (@Jeff1999)
>>>>>>> 847158e (06-02-2025_19:20)
=======
# Autor: Jeff Luo (@Jeff1999)
>>>>>>> origin/main

Z = np.arange(10)
print(sliding_window_view(Z, window_shape=3))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 77. ¿Cómo negar un booleano, o cambiar el signo de un flotante en su lugar? (★★★)


```python
# Autor: Nathaniel J. Smith
<<<<<<< HEAD
=======
#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)


```python
# Author: Nathaniel J. Smith
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 77. ¿Cómo negar un booleano, o cambiar el signo de un flotante en su lugar? (★★★)


```python
# Autor: Nathaniel J. Smith
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

Z = np.random.randint(0,2,100)
np.logical_not(Z, out=Z)

Z = np.random.uniform(-1.0,1.0,100)
np.negative(Z, out=Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 78. Considera 2 conjuntos de puntos P0,P1 que describen líneas (2d) y un punto p, ¿cómo calcular la distancia desde p a cada línea i (P0[i],P1[i])? (★★★)
=======
#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 78. Considera 2 conjuntos de puntos P0,P1 que describen líneas (2d) y un punto p, ¿cómo calcular la distancia desde p a cada línea i (P0[i],P1[i])? (★★★)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 78. Considera 2 conjuntos de puntos P0,P1 que describen líneas (2d) y un punto p, ¿cómo calcular la distancia desde p a cada línea i (P0[i],P1[i])? (★★★)
>>>>>>> origin/main


```python
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 79. Considera 2 conjuntos de puntos P0,P1 que describen líneas (2d) y un conjunto de puntos P, ¿cómo calcular la distancia desde cada punto j (P[j]) a cada línea i (P0[i],P1[i])? (★★★)


```python
# Autor: Italmassov Kuanysh

# basado en la función distance de la pregunta anterior
<<<<<<< HEAD
=======
#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)
=======
#### 79. Considera 2 conjuntos de puntos P0,P1 que describen líneas (2d) y un conjunto de puntos P, ¿cómo calcular la distancia desde cada punto j (P[j]) a cada línea i (P0[i],P1[i])? (★★★)
>>>>>>> 847158e (06-02-2025_19:20)


```python
# Autor: Italmassov Kuanysh

<<<<<<< HEAD
# based on distance function from previous question
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# basado en la función distance de la pregunta anterior
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print(np.array([distance(P0,P1,p_i) for p_i in p]))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 80. Considera un array arbitrario, escribe una función que extraiga una subparte con una forma fija y centrada en un elemento dado (rellena con un valor `fill` cuando sea necesario) (★★★)


```python
# Autor: Nicolas Rougier
<<<<<<< HEAD
=======
#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)


```python
# Author: Nicolas Rougier
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 80. Considera un array arbitrario, escribe una función que extraiga una subparte con una forma fija y centrada en un elemento dado (rellena con un valor `fill` cuando sea necesario) (★★★)


```python
# Autor: Nicolas Rougier
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 81. Considera un array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ¿cómo generar un array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)


```python
# Autor: Stefan van der Walt
<<<<<<< HEAD
=======
#### 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)


```python
# Author: Stefan van der Walt
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 81. Considera un array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ¿cómo generar un array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)


```python
# Autor: Stefan van der Walt
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

Z = np.arange(1,15,dtype=np.uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Autor: Jeff Luo (@Jeff1999)
=======
# Author: Jeff Luo (@Jeff1999)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Jeff Luo (@Jeff1999)
>>>>>>> 847158e (06-02-2025_19:20)
=======
# Autor: Jeff Luo (@Jeff1999)
>>>>>>> origin/main

Z = np.arange(1, 15, dtype=np.uint32)
print(sliding_window_view(Z, window_shape=4))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 82. Calcula el rango de una matriz (★★★)


```python
# Autor: Stefan van der Walt

Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Descomposición en valores singulares
rank = np.sum(S > 1e-10)
print(rank)

# solución alternativa:
# Autor: Jeff Luo (@Jeff1999)
<<<<<<< HEAD
=======
#### 82. Compute a matrix rank (★★★)
=======
#### 82. Calcula el rango de una matriz (★★★)
>>>>>>> 847158e (06-02-2025_19:20)


```python
# Autor: Stefan van der Walt

Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Descomposición en valores singulares
rank = np.sum(S > 1e-10)
print(rank)

<<<<<<< HEAD
# alternative solution:
# Author: Jeff Luo (@Jeff1999)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# solución alternativa:
# Autor: Jeff Luo (@Jeff1999)
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

rank = np.linalg.matrix_rank(Z)
print(rank)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 83. ¿Cómo encontrar el valor más frecuente en un array?
=======
#### 83. How to find the most frequent value in an array?
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 83. ¿Cómo encontrar el valor más frecuente en un array?
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 83. ¿Cómo encontrar el valor más frecuente en un array?
>>>>>>> origin/main


```python
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 84. Extrae todos los bloques contiguos de 3x3 de una matriz aleatoria 10x10 (★★★)


```python
# Autor: Chris Barker
<<<<<<< HEAD
=======
#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)


```python
# Author: Chris Barker
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 84. Extrae todos los bloques contiguos de 3x3 de una matriz aleatoria 10x10 (★★★)


```python
# Autor: Chris Barker
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Autor: Jeff Luo (@Jeff1999)
=======
# Author: Jeff Luo (@Jeff1999)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Jeff Luo (@Jeff1999)
>>>>>>> 847158e (06-02-2025_19:20)
=======
# Autor: Jeff Luo (@Jeff1999)
>>>>>>> origin/main

Z = np.random.randint(0,5,(10,10))
print(sliding_window_view(Z, window_shape=(3, 3)))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 85. Crea una subclase de array 2D tal que Z[i,j] == Z[j,i] (★★★)


```python
# Autor: Eric O. Lebigot
# Nota: solo funciona para arrays 2D y asignación de valores usando índices
<<<<<<< HEAD
=======
#### 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)


```python
# Author: Eric O. Lebigot
# Note: only works for 2d array and value setting using indices
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 85. Crea una subclase de array 2D tal que Z[i,j] == Z[j,i] (★★★)


```python
# Autor: Eric O. Lebigot
# Nota: solo funciona para arrays 2D y asignación de valores usando índices
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 86. Considera un conjunto de p matrices con forma (n,n) y un conjunto de p vectores con forma (n,1). ¿Cómo calcular la suma de los p productos de matrices a la vez? (el resultado tiene forma (n,1)) (★★★)


```python
# Autor: Stefan van der Walt
<<<<<<< HEAD
=======
#### 86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)


```python
# Author: Stefan van der Walt
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 86. Considera un conjunto de p matrices con forma (n,n) y un conjunto de p vectores con forma (n,1). ¿Cómo calcular la suma de los p productos de matrices a la vez? (el resultado tiene forma (n,1)) (★★★)


```python
# Autor: Stefan van der Walt
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
# Funciona, porque:
# M es (p,n,n)
# V es (p,n,1)
# Por lo tanto, sumando sobre los ejes emparejados 0 y 0 (de M y V independientemente),
# y 2 y 1, para quedar con un vector (n,1).
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
```
#### 87. Considera un array de 16x16, ¿cómo obtener la suma de bloques (el tamaño del bloque es 4x4)? (★★★)


```python
# Autor: Robert Kern
<<<<<<< HEAD
=======
# It works, because:
# M is (p,n,n)
# V is (p,n,1)
# Thus, summing over the paired axes 0 and 0 (of M and V independently),
# and 2 and 1, to remain with a (n,1) vector.
=======
>>>>>>> 847158e (06-02-2025_19:20)
```
#### 87. Considera un array de 16x16, ¿cómo obtener la suma de bloques (el tamaño del bloque es 4x4)? (★★★)


```python
<<<<<<< HEAD
# Author: Robert Kern
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Robert Kern
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# solución alternativa:
# Autor: Sebastian Wallkötter (@FirefoxMetzger)
=======
# alternative solution:
# Author: Sebastian Wallkötter (@FirefoxMetzger)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# solución alternativa:
# Autor: Sebastian Wallkötter (@FirefoxMetzger)
>>>>>>> 847158e (06-02-2025_19:20)
=======
# solución alternativa:
# Autor: Sebastian Wallkötter (@FirefoxMetzger)
>>>>>>> origin/main

Z = np.ones((16,16))
k = 4

windows = np.lib.stride_tricks.sliding_window_view(Z, (k, k))
S = windows[::k, ::k, ...].sum(axis=(-2, -1))

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Autor: Jeff Luo (@Jeff1999)
=======
# Author: Jeff Luo (@Jeff1999)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Jeff Luo (@Jeff1999)
>>>>>>> 847158e (06-02-2025_19:20)
=======
# Autor: Jeff Luo (@Jeff1999)
>>>>>>> origin/main

Z = np.ones((16, 16))
k = 4
print(sliding_window_view(Z, window_shape=(k, k))[::k, ::k].sum(axis=(-2, -1)))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 88. ¿Cómo implementar el Juego de la Vida usando arrays de numpy? (★★★)


```python
# Autor: Nicolas Rougier

def iterate(Z):
    # Contar vecinos
<<<<<<< HEAD
=======
#### 88. How to implement the Game of Life using numpy arrays? (★★★)
=======
#### 88. ¿Cómo implementar el Juego de la Vida usando arrays de numpy? (★★★)
>>>>>>> 847158e (06-02-2025_19:20)


```python
# Autor: Nicolas Rougier

def iterate(Z):
<<<<<<< HEAD
    # Count neighbours
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
    # Contar vecinos
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    # Aplicar reglas
=======
    # Apply rules
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
    # Aplicar reglas
>>>>>>> 847158e (06-02-2025_19:20)
=======
    # Aplicar reglas
>>>>>>> origin/main
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)
print(Z)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#### 89. ¿Cómo obtener los n valores más grandes de un array (★★★)
=======
#### 89. How to get the n largest values of an array (★★★)
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 89. ¿Cómo obtener los n valores más grandes de un array (★★★)
>>>>>>> 847158e (06-02-2025_19:20)
=======
#### 89. ¿Cómo obtener los n valores más grandes de un array (★★★)
>>>>>>> origin/main


```python
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
# Lento
print (Z[np.argsort(Z)[-n:]])

# Rápido
print (Z[np.argpartition(-Z,n)[:n]])
```
#### 90. Dado un número arbitrario de vectores, construye el producto cartesiano (todas las combinaciones de cada elemento) (★★★)


```python
# Autor: Stefan Van der Walt
<<<<<<< HEAD
=======
# Slow
=======
# Lento
>>>>>>> 847158e (06-02-2025_19:20)
print (Z[np.argsort(Z)[-n:]])

# Rápido
print (Z[np.argpartition(-Z,n)[:n]])
```
#### 90. Dado un número arbitrario de vectores, construye el producto cartesiano (todas las combinaciones de cada elemento) (★★★)


```python
<<<<<<< HEAD
# Author: Stefan Van der Walt
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Stefan Van der Walt
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 91. ¿Cómo crear un array de registros a partir de un array regular? (★★★)


```python
Z = np.array([("Hola", 2.5, 3),
              ("Mundo", 3.6, 2)])
<<<<<<< HEAD
=======
#### 91. How to create a record array from a regular array? (★★★)


```python
Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 91. ¿Cómo crear un array de registros a partir de un array regular? (★★★)


```python
Z = np.array([("Hola", 2.5, 3),
              ("Mundo", 3.6, 2)])
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
R = np.core.records.fromarrays(Z.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 92. Considera un vector grande Z, calcula Z a la potencia de 3 usando 3 métodos diferentes (★★★)


```python
# Autor: Ryan G.
<<<<<<< HEAD
=======
#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)


```python
# Author: Ryan G.
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 92. Considera un vector grande Z, calcula Z a la potencia de 3 usando 3 métodos diferentes (★★★)


```python
# Autor: Ryan G.
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

x = np.random.rand(int(5e7))

%timeit np.power(x,3)
%timeit x*x*x
%timeit np.einsum('i,i,i->i',x,x,x)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 93. Considera dos arrays A y B de forma (8,3) y (2,2). ¿Cómo encontrar filas de A que contengan elementos de cada fila de B sin importar el orden de los elementos en B? (★★★)


```python
# Autor: Gabe Schwartz
<<<<<<< HEAD
=======
#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)


```python
# Author: Gabe Schwartz
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 93. Considera dos arrays A y B de forma (8,3) y (2,2). ¿Cómo encontrar filas de A que contengan elementos de cada fila de B sin importar el orden de los elementos en B? (★★★)


```python
# Autor: Gabe Schwartz
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 94. Considerando una matriz 10x3, extrae filas con valores desiguales (por ejemplo, [2,2,3]) (★★★)


```python
# Autor: Robert Kern

Z = np.random.randint(0,5,(10,3))
print(Z)
# solución para arrays de todos los tipos (incluyendo arrays de cadenas y arrays de registros)
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(U)
# solución para arrays numéricos solamente, funcionará para cualquier número de columnas en Z
U = Z[Z.max(axis=1) != Z.min(axis=1),:]
print(U)
```
#### 95. Convierte un vector de enteros en una representación binaria de matriz (★★★)


```python
# Autor: Warren Weckesser
<<<<<<< HEAD
=======
#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)
=======
#### 94. Considerando una matriz 10x3, extrae filas con valores desiguales (por ejemplo, [2,2,3]) (★★★)
>>>>>>> 847158e (06-02-2025_19:20)


```python
# Autor: Robert Kern

Z = np.random.randint(0,5,(10,3))
print(Z)
# solución para arrays de todos los tipos (incluyendo arrays de cadenas y arrays de registros)
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(U)
# solución para arrays numéricos solamente, funcionará para cualquier número de columnas en Z
U = Z[Z.max(axis=1) != Z.min(axis=1),:]
print(U)
```
#### 95. Convierte un vector de enteros en una representación binaria de matriz (★★★)


```python
<<<<<<< HEAD
# Author: Warren Weckesser
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Warren Weckesser
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Autor: Daniel T. McDonald
=======
# Author: Daniel T. McDonald
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Daniel T. McDonald
>>>>>>> 847158e (06-02-2025_19:20)
=======
# Autor: Daniel T. McDonald
>>>>>>> origin/main

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 96. Dado un array bidimensional, ¿cómo extraer filas únicas? (★★★)


```python
# Autor: Jaime Fernández del Río
<<<<<<< HEAD
=======
#### 96. Given a two dimensional array, how to extract unique rows? (★★★)


```python
# Author: Jaime Fernández del Río
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 96. Dado un array bidimensional, ¿cómo extraer filas únicas? (★★★)


```python
# Autor: Jaime Fernández del Río
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

Z = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Autor: Andreas Kouzelis
=======
# Author: Andreas Kouzelis
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Andreas Kouzelis
>>>>>>> 847158e (06-02-2025_19:20)
=======
# Autor: Andreas Kouzelis
>>>>>>> origin/main
# NumPy >= 1.13
uZ = np.unique(Z, axis=0)
print(uZ)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 97. Considerando 2 vectores A y B, escribe el equivalente de einsum de inner, outer, sum, y mul function (★★★)


```python
# Autor: Alex Riley
# Asegúrate de leer: http://ajcr.net/Basic-guide-to-einsum/
<<<<<<< HEAD
=======
#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)


```python
# Author: Alex Riley
# Make sure to read: http://ajcr.net/Basic-guide-to-einsum/
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 97. Considerando 2 vectores A y B, escribe el equivalente de einsum de inner, outer, sum, y mul function (★★★)


```python
# Autor: Alex Riley
# Asegúrate de leer: http://ajcr.net/Basic-guide-to-einsum/
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B)
np.einsum('i,j->ij', A, B)    # np.outer(A, B)
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 98. Considerando un camino descrito por dos vectores (X,Y), ¿cómo muestrearlo usando muestras equidistantes (★★★)?


```python
# Autor: Bas Swinckels
<<<<<<< HEAD
=======
#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?


```python
# Author: Bas Swinckels
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
#### 98. Considerando un camino descrito por dos vectores (X,Y), ¿cómo muestrearlo usando muestras equidistantes (★★★)?


```python
# Autor: Bas Swinckels
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # longitudes de segmento
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrar camino
r_int = np.linspace(0, r.max(), 200) # camino espaciado regularmente
x_int = np.interp(r_int, r, x)       # integrar camino
y_int = np.interp(r_int, r, y)
```
#### 99. Dado un entero n y un array 2D X, selecciona de X las filas que pueden interpretarse como extracciones de una distribución multinomial con n grados, es decir, las filas que solo contienen enteros y que suman n. (★★★)


```python
# Autor: Evgeni Burovski
<<<<<<< HEAD
=======
dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
=======
dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # longitudes de segmento
>>>>>>> 847158e (06-02-2025_19:20)
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrar camino
r_int = np.linspace(0, r.max(), 200) # camino espaciado regularmente
x_int = np.interp(r_int, r, x)       # integrar camino
y_int = np.interp(r_int, r, y)
```
#### 99. Dado un entero n y un array 2D X, selecciona de X las filas que pueden interpretarse como extracciones de una distribución multinomial con n grados, es decir, las filas que solo contienen enteros y que suman n. (★★★)


```python
<<<<<<< HEAD
# Author: Evgeni Burovski
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
# Autor: Evgeni Burovski
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main

X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])
```
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/main
#### 100. Calcula intervalos de confianza del 95% bootstrap para la media de un array 1D X (es decir, re-muestrea los elementos de un array con reemplazo N veces, calcula la media de cada muestra, y luego calcula percentiles sobre las medias). (★★★)


```python
# Autor: Jessica B. Hamrick

X = np.random.randn(100) # array 1D aleatorio
N = 1000 # número de muestras bootstrap
<<<<<<< HEAD
=======
#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)
=======
#### 100. Calcula intervalos de confianza del 95% bootstrap para la media de un array 1D X (es decir, re-muestrea los elementos de un array con reemplazo N veces, calcula la media de cada muestra, y luego calcula percentiles sobre las medias). (★★★)
>>>>>>> 847158e (06-02-2025_19:20)


```python
# Autor: Jessica B. Hamrick

<<<<<<< HEAD
X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
>>>>>>> 67236fe (30-01-2025 - 19:20)
=======
X = np.random.randn(100) # array 1D aleatorio
N = 1000 # número de muestras bootstrap
>>>>>>> 847158e (06-02-2025_19:20)
=======
>>>>>>> origin/main
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)
```