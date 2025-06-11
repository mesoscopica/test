"""
Simulacion de cadenas polimericas por modelo de cadena libremente unida (Random Walk 3D). 
Este script:

1. Genera configuraciones de polimeros de diferentes longitudes (N) mediante caminante aleatorio en 3D.
2. Calcula para cada cadena el radio de giro (Rg) y la distancia extremo–extremo (Rn).
3. Repite la simulacion multiples veces para cada N, obteniendo medias e incertidumbres
   estadisticas de Rg y Rn.
4. Guarda los resultados en archivos de texto separados (valores_rg.txt y valores_rn.txt).
5. Grafica la ultima cadena simulada en un espacio 3D, mostrando el inicio (verde) y final (rojo).
"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simular_polimero(N, longitud_paso):
    posiciones = np.zeros((N + 1, 3))
    for i in range(1, N + 1):
        direccion = np.random.normal(size=3)
        direccion /= np.linalg.norm(direccion)
        posiciones[i] = posiciones[i - 1] + longitud_paso * direccion
    return posiciones

def calcular_radio_de_giro_cuadrado(posiciones):
    centro_de_masa = np.mean(posiciones, axis=0)
    return np.mean(np.sum((posiciones - centro_de_masa)**2, axis=1))

def calcular_radio_de_giro(posiciones):
    rg_cuadrado = calcular_radio_de_giro_cuadrado(posiciones)
    return sqrt(rg_cuadrado)

def calcular_distancia_extremo_a_extremo(posiciones):
    vector_extremo_a_extremo = posiciones[-1] - posiciones[0]
    return np.linalg.norm(vector_extremo_a_extremo)

def graficar_polimero(posiciones, ax=None, etiqueta="Polimero", color="blue"):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot(posiciones[:, 0], posiciones[:, 1], posiciones[:, 2], lw=0.5, label=etiqueta, color=color)
    ax.scatter(posiciones[0, 0], posiciones[0, 1], posiciones[0, 2], c='green', s=50, label="Inicio")
    ax.scatter(posiciones[-1, 0], posiciones[-1, 1], posiciones[-1, 2], c='red', s=50, label="Final")
    ax.set_xlabel("Eje X")
    ax.set_ylabel("Eje Y")
    ax.set_zlabel("Eje Z")
    ax.set_title("Simulacion 3D de un Polimero - Modelo Cadena Libremente Unida")
    ax.legend()
    return ax

def simular_cadenas(numero_cadenas, N, longitud_paso):
    lista_rg = []
    lista_rn = []
    for _ in range(numero_cadenas):
        posiciones = simular_polimero(N, longitud_paso)
        rg = calcular_radio_de_giro(posiciones)
        lista_rg.append(rg)
        rn = calcular_distancia_extremo_a_extremo(posiciones)
        lista_rn.append(rn)
    promedio_rg = np.mean(lista_rg)
    error_rg = np.std(lista_rg, ddof=1) / np.sqrt(numero_cadenas)
    promedio_rn = np.mean(lista_rn)
    error_rn = np.std(lista_rn, ddof=1) / np.sqrt(numero_cadenas)
    return promedio_rg, error_rg, promedio_rn, error_rn

def guardar_resultados(grado_polimerizacion, valores_rg, errores_rg, valores_rn, errores_rn):
    """
    Guarda los resultados en dos archivos separados:
    - valores_rg.txt contiene N, radio de giro promedio (Rg) y error de la media.
    - valores_rn.txt contiene N, distancia promedio de extremo a extremo (Rn) y error de la media.
    """
    with open("valores_rg.txt", "w") as archivo_rg:
        archivo_rg.write("N\tRg\tError_Rg\n")
        for N, rg, err_rg in zip(grado_polimerizacion, valores_rg, errores_rg):
            archivo_rg.write(f"{N}\t{rg:.4f}\t{err_rg:.4f}\n")
    
    with open("valores_rn.txt", "w") as archivo_rn:
        archivo_rn.write("N\tRn\tError_Rn\n")
        for N, rn, err_rn in zip(grado_polimerizacion, valores_rn, errores_rn):
            archivo_rn.write(f"{N}\t{rn:.4f}\t{err_rn:.4f}\n")

# Parametros de simulacion
longitud_paso = 1.0         
grado_polimerizacion = [i * 5000 for i in range(1, 21)]
num_simulaciones = 100 

# Simulacion de multiples cadenas para diferentes grados de polimerizacion
valores_rg = []
errores_rg = []
valores_rn = []
errores_rn = []

for N in grado_polimerizacion:
    promedio_rg, error_rg, promedio_rn, error_rn = simular_cadenas(num_simulaciones, N, longitud_paso)
    valores_rg.append(promedio_rg)
    errores_rg.append(error_rg)
    valores_rn.append(promedio_rn)
    errores_rn.append(error_rn)

# Guardar resultados en archivos
guardar_resultados(grado_polimerizacion, valores_rg, errores_rg, valores_rn, errores_rn)

# Simular y graficar la ultima cadena
posiciones_ultimo = simular_polimero(grado_polimerizacion[-1], longitud_paso)
ax = graficar_polimero(posiciones_ultimo)
plt.show()
