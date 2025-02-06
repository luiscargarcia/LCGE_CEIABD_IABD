# 100 ejercicios de numpy

Esta es una colección de ejercicios que han sido recopilados de la lista de correo de numpy, 
Stack Overflow y la documentación de numpy. El objetivo de esta colección es ofrecer una referencia
rápida tanto para usuarios nuevos como experimentados, y también proporcionar un conjunto de 
ejercicios para quienes enseñan.

Si encuentras un error o crees que tienes una mejor manera de resolver alguno de ellos,
no dudes en abrir un issue en <https://github.com/rougier/numpy-100>.
Archivo generado automáticamente. Consulta la documentación para actualizar preguntas/respuestas/pistas programáticamente.

#### 1. Importar el paquete numpy bajo el nombre `np` (★☆☆)


```python
import numpy as np
```
#### 2. Imprimir la versión de numpy y la configuración (★☆☆)


```python
print(np.__version__)
np.show_config()
```
#### 3. Crear un vector nulo de tamaño 10 (★☆☆)


#### 4. ¿Cómo encontrar el tamaño de memoria de cualquier array? (★☆☆)


```python
Z = np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))
```
#### 5. ¿Cómo obtener la documentación de la función numpy add desde la línea de comandos? (★☆☆)


```python
%run `python -c "import numpy; numpy.info(numpy.add)"`
```
#### 6. Crea un vector nulo de tamaño 10 pero el quinto valor que sea 1 (★☆☆)


```python
Z = np.zeros(10)
Z[4] = 1
print(Z)
```
#### 7. Crea un vector con valores que van de 10 a 49 (★☆☆)


```python
Z = np.arange(10,50)
print(Z)
```
#### 8. Invierte un vector (el primer elemento se convierte en el último) (★☆☆)


```python
Z = np.arange(50)
Z = Z[::-1]
print(Z)
```
#### 9. Crea una matriz 3x3 con valores que van de 0 a 8 (★☆☆)


```python
Z = np.arange(9).reshape(3, 3)
print(Z)
```
```python
Z = np.arange(9).reshape(3, 3)
print(Z)
```
#### 10. Encuentra los índices de los elementos no nulos de [1,2,0,0,4,0] (★☆☆)


```python
nz = np.nonzero([1,2,0,0,4,0])
print(nz)
```
#### 11. Crea una matriz identidad de 3x3 (★☆☆)


```python
Z = np.eye(3)
print(Z)
```
#### 12. Crea un array 3x3x3 con valores aleatorios (★☆☆)


```python
Z = np.random.random((3,3,3))
print(Z)
```
#### 13. Crea un array 10x10 con valores aleatorios y encuentra los valores mínimo y máximo (★☆☆)


```python
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)
```
#### 14. Crea un vector aleatorio de tamaño 30 y encuentra el valor medio (★☆☆)


```python
Z = np.random.random(30)
m = Z.mean()
print(m)
```
#### 15. Crea un array 2D con 1 en el borde y 0 en el interior (★☆☆)


```python
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)
```
#### 16. ¿Cómo añadir un borde (relleno con 0's) alrededor de un array existente? (★☆☆)


```python
Z = np.ones((5,5))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
print(Z)

# Usando indexación avanzada
Z[:, [0, -1]] = 0
Z[[0, -1], :] = 0
print(Z)
```
#### 17. ¿Cuál es el resultado de la siguiente expresión? (★☆☆)
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
#### 18. Crea una matriz 5x5 con valores 1,2,3,4 justo debajo de la diagonal (★☆☆)


```python
Z = np.diag(1+np.arange(4),k=-1)
print(Z)
```
#### 19. Crea una matriz 8x8 y rellénala con un patrón de tablero de ajedrez (★☆☆)


```python
Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)
```
#### 20. Considera un array de forma (6,7,8), ¿cuál es el índice (x,y,z) del elemento 100? (★☆☆)


```python
print(np.unravel_index(99,(6,7,8)))
```
#### 21. Crea un tablero de ajedrez 8x8 usando la función tile (★☆☆)


```python
Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)
```
#### 22. Normaliza una matriz aleatoria 5x5 (★☆☆)


```python
Z = np.random.random((5,5))
Z = (Z - np.mean (Z)) / (np.std (Z))
print(Z)
```
#### 23. Crea un dtype personalizado que describa un color como cuatro bytes sin signo (RGBA) (★☆☆)


```python
color = np.dtype([("r", np.ubyte),
                  ("g", np.ubyte),
                  ("b", np.ubyte),
                  ("a", np.ubyte)])
```
#### 24. Multiplica una matriz 5x3 por una matriz 3x2 (producto de matrices reales) (★☆☆)


```python
Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)

# Solución alternativa, en Python 3.5 y superior
Z = np.ones((5,3)) @ np.ones((3,2))
print(Z)
```
#### 25. Dado un array 1D, niega todos los elementos que están entre 3 y 8, en su lugar. (★☆☆)


```python
# Autor: Evgeni Burovski

Z = np.arange(11)
Z[(3 < Z) & (Z < 8)] *= -1
print(Z)
```
#### 26. ¿Cuál es la salida del siguiente script? (★☆☆)
```python
# Autor: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```


```python
# Autor: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```
#### 27. Considera un vector entero Z, ¿cuál de estas expresiones son legales? (★☆☆)
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
#### 28. ¿Cuáles son los resultados de las siguientes expresiones? (★☆☆)
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
#### 29. ¿Cómo redondear un array de flotantes alejándose de cero? (★☆☆)


```python
# Autor: Charles R Harris

Z = np.random.uniform(-10,+10,10)
print(np.copysign(np.ceil(np.abs(Z)), Z))

# Más legible pero menos eficiente
print(np.where(Z>0, np.ceil(Z), np.floor(Z)))
```
#### 30. ¿Cómo encontrar valores comunes entre dos arrays? (★☆☆)


```python
Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))
```
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
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```


```python
np.sqrt(-1) == np.emath.sqrt(-1)
```
#### 33. ¿Cómo obtener las fechas de ayer, hoy y mañana? (★☆☆)


```python
yesterday = np.datetime64('today') - np.timedelta64(1)
today     = np.datetime64('today')
tomorrow  = np.datetime64('today') + np.timedelta64(1)
```
#### 34. ¿Cómo obtener todas las fechas correspondientes al mes de julio de 2016? (★★☆)


```python
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)
```
#### 35. ¿Cómo calcular ((A+B)*(-A/2)) en su lugar (sin copia)? (★★☆)


```python
A = np.ones(3)*1
B = np.ones(3)*2
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)
```
#### 36. Extrae la parte entera de un array aleatorio de números positivos usando 4 métodos diferentes (★★☆)


```python
Z = np.random.uniform(0,10,10)

print(Z - Z%1)
print(Z // 1)
print(np.floor(Z))
print(Z.astype(int))
print(np.trunc(Z))
```
#### 37. Crea una matriz 5x5 con valores de fila que van de 0 a 4 (★★☆)


```python
Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)

# sin broadcasting
Z = np.tile(np.arange(0, 5), (5,1))
print(Z)
```
#### 38. Considera una función generadora que genera 10 enteros y úsala para construir un array (★☆☆)


```python
def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)
```
#### 39. Crea un vector de tamaño 10 con valores que van de 0 a 1, ambos excluidos (★★☆)


```python
Z = np.linspace(0,1,11,endpoint=False)[1:]
print(Z)
```
#### 40. Crea un vector aleatorio de tamaño 10 y ordénalo (★★☆)


```python
Z = np.random.random(10)
Z.sort()
print(Z)
```
#### 41. ¿Cómo sumar un array pequeño más rápido que np.sum? (★★☆)


```python
# Autor: Evgeni Burovski

Z = np.arange(10)
np.add.reduce(Z)
```
#### 42. Considera dos arrays aleatorios A y B, verifica si son iguales (★★☆)


```python
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)

# Asumiendo forma idéntica de los arrays y una tolerancia para la comparación de valores
equal = np.allclose(A,B)
print(equal)

# Verificando tanto la forma como los valores de los elementos, sin tolerancia (los valores deben ser exactamente iguales)
equal = np.array_equal(A,B)
print(equal)
```
#### 43. Haz un array inmutable (solo lectura) (★★☆)


```python
Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1
```
#### 44. Considera una matriz aleatoria 10x2 que representa coordenadas cartesianas, conviértelas a coordenadas polares (★★☆)


```python
Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)
```
#### 45. Crea un vector aleatorio de tamaño 10 y reemplaza el valor máximo por 0 (★★☆)


```python
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)
```
#### 46. Crea un array estructurado con coordenadas `x` y `y` que cubran el área [0,1]x[0,1] (★★☆)


```python
Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z)
```
#### 47. Dado dos arrays, X e Y, construye la matriz de Cauchy C (Cij =1/(xi - yj)) (★★☆)


```python
# Autor: Evgeni Burovski

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))
```
#### 48. Imprime el valor mínimo y máximo representable para cada tipo escalar de numpy (★★☆)


```python
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
```
#### 49. ¿Cómo imprimir todos los valores de un array? (★★☆)


```python
np.set_printoptions(threshold=float("inf"))
Z = np.zeros((40,40))
print(Z)
```
#### 50. ¿Cómo encontrar el valor más cercano (a un escalar dado) en un vector? (★★☆)


```python
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])
```
#### 51. Crea un array estructurado que represente una posición (x,y) y un color (r,g,b) (★★☆)


```python
Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)
```
#### 52. Considera un vector aleatorio con forma (100,2) que representa coordenadas, encuentra las distancias punto a punto (★★☆)


```python
Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)

# Mucho más rápido con scipy
import scipy
# Gracias Gavin Heverly-Coulson (#issue 1)
import scipy.spatial

Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)
print(D)
```
#### 53. ¿Cómo convertir un array de flotantes (32 bits) en un entero (32 bits) en su lugar?


```python
# Gracias Vikas (https://stackoverflow.com/a/10622758/5989906)
# & unutbu (https://stackoverflow.com/a/4396247/5989906)
Z = (np.random.rand(10)*100).astype(np.float32)
Y = Z.view(np.int32)
Y[:] = Z
print(Y)
```
#### 54. ¿Cómo leer el siguiente archivo? (★★☆)
```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```


```python
from io import StringIO

# Archivo falso
s = StringIO('''1, 2, 3, 4, 5

                6,  ,  , 7, 8

                 ,  , 9,10,11
''')
Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
print(Z)
```
#### 55. ¿Cuál es el equivalente de enumerate para arrays de numpy? (★★☆)


```python
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)
for index in np.ndindex(Z.shape):
    print(index, Z[index])
```
#### 56. Genera un array genérico 2D similar a una Gaussiana (★★☆)


```python
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)
```
#### 57. ¿Cómo colocar aleatoriamente p elementos en un array 2D? (★★☆)


```python
# Autor: Divakar

n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print(Z)
```
#### 58. Resta la media de cada fila de una matriz (★★☆)


```python
# Autor: Warren Weckesser

X = np.random.rand(5, 10)

# Versiones recientes de numpy
Y = X - X.mean(axis=1, keepdims=True)

# Versiones antiguas de numpy
Y = X - X.mean(axis=1).reshape(-1, 1)

print(Y)
```
#### 59. ¿Cómo ordenar un array por la n-ésima columna? (★★☆)


```python
# Autor: Steve Tjoa

Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()])
```
#### 60. ¿Cómo saber si un array 2D dado tiene columnas nulas? (★★☆)


```python
# Autor: Warren Weckesser

# nulo : 0 
Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any())

# nulo : np.nan
Z=np.array([
    [0,1,np.nan],
    [1,2,np.nan],
    [4,5,np.nan]
])
print(np.isnan(Z).all(axis=0))
```
#### 61. Encuentra el valor más cercano a un valor dado en un array (★★☆)


```python
Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)
```
#### 62. Considerando dos arrays con forma (1,3) y (3,1), ¿cómo calcular su suma usando un iterador? (★★☆)


```python
A = np.arange(3).reshape(3,1)
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None])
for x,y,z in it: z[...] = x + y
print(it.operands[2])
```
#### 63. Crea una clase de array que tenga un atributo de nombre (★★☆)


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
#### 64. Considera un vector dado, ¿cómo añadir 1 a cada elemento indexado por un segundo vector (ten cuidado con los índices repetidos)? (★★★)


```python
# Autor: Brett Olsen

Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)

# Otra solución
# Autor: Bartosz Telenczuk
np.add.at(Z, I, 1)
print(Z)
```
#### 65. ¿Cómo acumular elementos de un vector (X) en un array (F) basado en una lista de índices (I)? (★★★)


```python
# Autor: Alan G Isaac

X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)
```
#### 66. Considerando una imagen de (w,h,3) de (dtype=ubyte), calcula el número de colores únicos (★★☆)


```python
# Autor: Fisher Wang

w, h = 256, 256
I = np.random.randint(0, 4, (h, w, 3)).astype(np.ubyte)
colors = np.unique(I.reshape(-1, 3), axis=0)
n = len(colors)
print(n)

# Versión más rápida
# Autor: Mark Setchell
# https://stackoverflow.com/a/59671950/2836621

w, h = 256, 256
I = np.random.randint(0,4,(h,w,3), dtype=np.uint8)

# Ver cada píxel como un solo entero de 24 bits, en lugar de tres bytes de 8 bits
I24 = np.dot(I.astype(np.uint32),[1,256,65536])

# Contar colores únicos
n = len(np.unique(I24))
print(n)
```
#### 67. Considerando un array de cuatro dimensiones, ¿cómo obtener la suma sobre los dos últimos ejes a la vez? (★★★)


```python
A = np.random.randint(0,10,(3,4,3,4))
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

D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

# Solución con Pandas como referencia debido a su código más intuitivo
import pandas as pd
print(pd.Series(D).groupby(S).mean())
```
#### 69. ¿Cómo obtener la diagonal de un producto punto? (★★★)


```python
# Autor: Mathieu Blondel

A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))

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

Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)
```
#### 71. Considera un array de dimensión (5,5,3), ¿cómo multiplicarlo por un array con dimensiones (5,5)? (★★★)


```python
A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print(A * B[:,:,None])
```
#### 72. ¿Cómo intercambiar dos filas de un array? (★★★)


```python
# Autor: Eelco Hoogendoorn

A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)
```
#### 73. Considera un conjunto de 10 tríos que describen 10 triángulos (con vértices compartidos), encuentra el conjunto de segmentos de línea únicos que componen todos los triángulos (★★★)


```python
# Autor: Nicolas P. Rougier

faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)
```
#### 74. Dado un array ordenado C que corresponde a un bincount, ¿cómo producir un array A tal que np.bincount(A) == C? (★★★)


```python
# Autor: Jaime Fernández del Río

C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)
```
#### 75. ¿Cómo calcular promedios usando una ventana deslizante sobre un array? (★★★)


```python
# Autor: Jaime Fernández del Río

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))

# Autor: Jeff Luo (@Jeff1999)
# asegúrate de que tu NumPy >= 1.20.0

from numpy.lib.stride_tricks import sliding_window_view

Z = np.arange(20)
print(sliding_window_view(Z, window_shape=3).mean(axis=-1))
```
#### 76. Considera un array unidimensional Z, construye un array bidimensional cuya primera fila sea (Z[0],Z[1],Z[2]) y cada fila subsiguiente esté desplazada en 1 (la última fila debe ser (Z[-3],Z[-2],Z[-1]) (★★★)


```python
# Autor: Joe Kington / Erik Rigtorp
from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.strides[0], a.strides[0])
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)

# Autor: Jeff Luo (@Jeff1999)

Z = np.arange(10)
print(sliding_window_view(Z, window_shape=3))
```
#### 77. ¿Cómo negar un booleano, o cambiar el signo de un flotante en su lugar? (★★★)


```python
# Autor: Nathaniel J. Smith

Z = np.random.randint(0,2,100)
np.logical_not(Z, out=Z)

Z = np.random.uniform(-1.0,1.0,100)
np.negative(Z, out=Z)
```
#### 78. Considera 2 conjuntos de puntos P0,P1 que describen líneas (2d) y un punto p, ¿cómo calcular la distancia desde p a cada línea i (P0[i],P1[i])? (★★★)


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
#### 79. Considera 2 conjuntos de puntos P0,P1 que describen líneas (2d) y un conjunto de puntos P, ¿cómo calcular la distancia desde cada punto j (P[j]) a cada línea i (P0[i],P1[i])? (★★★)


```python
# Autor: Italmassov Kuanysh

# basado en la función distance de la pregunta anterior
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print(np.array([distance(P0,P1,p_i) for p_i in p]))
```
#### 80. Considera un array arbitrario, escribe una función que extraiga una subparte con una forma fija y centrada en un elemento dado (rellena con un valor `fill` cuando sea necesario) (★★★)


```python
# Autor: Nicolas Rougier

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
#### 81. Considera un array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ¿cómo generar un array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)


```python
# Autor: Stefan van der Walt

Z = np.arange(1,15,dtype=np.uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)

# Autor: Jeff Luo (@Jeff1999)

Z = np.arange(1, 15, dtype=np.uint32)
print(sliding_window_view(Z, window_shape=4))
```
#### 82. Calcula el rango de una matriz (★★★)


```python
# Autor: Stefan van der Walt

Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Descomposición en valores singulares
rank = np.sum(S > 1e-10)
print(rank)

# solución alternativa:
# Autor: Jeff Luo (@Jeff1999)

rank = np.linalg.matrix_rank(Z)
print(rank)
```
#### 83. ¿Cómo encontrar el valor más frecuente en un array?


```python
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())
```
#### 84. Extrae todos los bloques contiguos de 3x3 de una matriz aleatoria 10x10 (★★★)


```python
# Autor: Chris Barker

Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)

# Autor: Jeff Luo (@Jeff1999)

Z = np.random.randint(0,5,(10,10))
print(sliding_window_view(Z, window_shape=(3, 3)))
```
#### 85. Crea una subclase de array 2D tal que Z[i,j] == Z[j,i] (★★★)


```python
# Autor: Eric O. Lebigot
# Nota: solo funciona para arrays 2D y asignación de valores usando índices

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
#### 86. Considera un conjunto de p matrices con forma (n,n) y un conjunto de p vectores con forma (n,1). ¿Cómo calcular la suma de los p productos de matrices a la vez? (el resultado tiene forma (n,1)) (★★★)


```python
# Autor: Stefan van der Walt

p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)

# Funciona, porque:
# M es (p,n,n)
# V es (p,n,1)
# Por lo tanto, sumando sobre los ejes emparejados 0 y 0 (de M y V independientemente),
# y 2 y 1, para quedar con un vector (n,1).
```
#### 87. Considera un array de 16x16, ¿cómo obtener la suma de bloques (el tamaño del bloque es 4x4)? (★★★)


```python
# Autor: Robert Kern

Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)

# solución alternativa:
# Autor: Sebastian Wallkötter (@FirefoxMetzger)

Z = np.ones((16,16))
k = 4

windows = np.lib.stride_tricks.sliding_window_view(Z, (k, k))
S = windows[::k, ::k, ...].sum(axis=(-2, -1))

# Autor: Jeff Luo (@Jeff1999)

Z = np.ones((16, 16))
k = 4
print(sliding_window_view(Z, window_shape=(k, k))[::k, ::k].sum(axis=(-2, -1)))
```
#### 88. ¿Cómo implementar el Juego de la Vida usando arrays de numpy? (★★★)


```python
# Autor: Nicolas Rougier

def iterate(Z):
    # Contar vecinos
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Aplicar reglas
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)
print(Z)
```
#### 89. ¿Cómo obtener los n valores más grandes de un array (★★★)


```python
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

# Lento
print (Z[np.argsort(Z)[-n:]])

# Rápido
print (Z[np.argpartition(-Z,n)[:n]])
```
#### 90. Dado un número arbitrario de vectores, construye el producto cartesiano (todas las combinaciones de cada elemento) (★★★)


```python
# Autor: Stefan Van der Walt

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
#### 91. ¿Cómo crear un array de registros a partir de un array regular? (★★★)


```python
Z = np.array([("Hola", 2.5, 3),
              ("Mundo", 3.6, 2)])
R = np.core.records.fromarrays(Z.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)
```
#### 92. Considera un vector grande Z, calcula Z a la potencia de 3 usando 3 métodos diferentes (★★★)


```python
# Autor: Ryan G.

x = np.random.rand(int(5e7))

%timeit np.power(x,3)
%timeit x*x*x
%timeit np.einsum('i,i,i->i',x,x,x)
```
#### 93. Considera dos arrays A y B de forma (8,3) y (2,2). ¿Cómo encontrar filas de A que contengan elementos de cada fila de B sin importar el orden de los elementos en B? (★★★)


```python
# Autor: Gabe Schwartz

A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)
```
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

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])

# Autor: Daniel T. McDonald

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))
```
#### 96. Dado un array bidimensional, ¿cómo extraer filas únicas? (★★★)


```python
# Autor: Jaime Fernández del Río

Z = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)

# Autor: Andreas Kouzelis
# NumPy >= 1.13
uZ = np.unique(Z, axis=0)
print(uZ)
```
#### 97. Considerando 2 vectores A y B, escribe el equivalente de einsum de inner, outer, sum, y mul function (★★★)


```python
# Autor: Alex Riley
# Asegúrate de leer: http://ajcr.net/Basic-guide-to-einsum/

A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B)
np.einsum('i,j->ij', A, B)    # np.outer(A, B)
```
#### 98. Considerando un camino descrito por dos vectores (X,Y), ¿cómo muestrearlo usando muestras equidistantes (★★★)?


```python
# Autor: Bas Swinckels

phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

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

X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])
```
#### 100. Calcula intervalos de confianza del 95% bootstrap para la media de un array 1D X (es decir, re-muestrea los elementos de un array con reemplazo N veces, calcula la media de cada muestra, y luego calcula percentiles sobre las medias). (★★★)


```python
# Autor: Jessica B. Hamrick

X = np.random.randn(100) # array 1D aleatorio
N = 1000 # número de muestras bootstrap
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)
```