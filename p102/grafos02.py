#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string, random
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import gzip, pickle
import time
import queue as qe
from sklearn.linear_model import LinearRegression
import networkx as nx
from queue import PriorityQueue

def fit_plot(l, func_2_fit, size_ini, size_fin, step):
    ''' Funcion que dibuja la lista de puntos y la compara con la funcion
    func_2_fit, para la cual se obtienen valores desde size_ini hasta
    size_fin con incrementos de step
    :l: lista de puntos a comparar con func_2_fit
    :func_2_fit: funcion para la que conseguiremos valores
    :size_ini: primero de los puntos a evaluar
    :size_fin: ultimo de los puntos a evaluar
    :step: salto entre valores
    :type l: lista
    :type func_2_fit: funcion
    :type size_ini: float
    :type size_fin: float
    :type step: float
    '''
    l_func_values =[i*func_2_fit(i) for i in range(size_ini, size_fin+1, step)]

    lr_m = LinearRegression()
    X = np.array(l_func_values).reshape( len(l_func_values), -1 )
    lr_m.fit(X, l)
    y_pred = lr_m.predict(X)

    plt.plot(l, '*', y_pred, '-')

def n2_log_n(n):
    ''' Funcion 2*n log(n)
    :n: parametro de la funcion
    :type n: int
    :return: resultado de la operacion
    :return type: float
    '''
    return n**2. * np.log(n)

def print_m_g(m_g):
    ''' Funcion que imprime una representacion a partir de la matriz de adyacenciaself.
    :m_g: matriz de adyacencia
    :type m_g: np 2 dimentional array
    '''
    print("graph_from_matrix:\n")
    n_v = m_g.shape[0]
    for u in range(n_v):
        for v in range(n_v):
            if v != u and m_g[u, v] != np.inf:
                print("(", u, v, ")", m_g[u, v])

def print_d_g(d_g):
    ''' Funcion que imprime una representacion a partir del diccionario de adyacencia
    :d_g: grafo representado con un doble diccionario
    :type d_g:  doble diccionario {origen: {destino: valor}}
    '''
    print("\ngraph_from_dict:\n")
    for u in d_g.keys():
        for v in d_g[u].keys():
            print("(", u, v, ")", d_g[u][v])

def rand_matr_pos_graph(n_nodes, sparse_factor, max_weight=50., decimals=0):
    """ Genera una matriz aleatoria de tamaño n_nodes x n_nodes, con un factor
    de dispersion de sparse_factor. En la que el valor maximo viene especificado
    por max_weight y en la que los elementos tienen tantos decimales como decimales
    :n_nodes: numero de nodos del grafos
    :sparse_factor: factor de dispersion de la matriz
    :max_weight: valor maximo de los pesos de la matriz
    :decimals: numero de decimales que queremos que tengan los elementos de la matriz
    :type n_nodes: int
    :type sparse_factor: float
    :type max_weight: float
    :type decimals: int
    :return: matriz de adyacencia que representa un grafos
    :return type: np 2 dimentional array
    """
    m=np.random.random((n_nodes, n_nodes))
    np.place(m,m<1-sparse_factor,np.inf)
    m=m*max_weight # Reescalado de los datos
    np.fill_diagonal(m,0)
    return m.round(decimals)

def m_g_2_d_g(m_g):
    """ Transformamos la representacion matricial del grafo m_g a una version
    con diccionarios d_g
    :m_g: matriz de adyacencia que representa el grafo.
    :type m_g: np 2 dimentional array
    :return: grafo en formato de doble diccionario {origen{destino: peso}}.
    :return type: dictionario cuyos valores son otro diccionario
    """
    dim=m_g.shape

    d_g={}
    for i in range(dim[0]):
        d_g[i]={}
        for j in range(dim[1]):
            if j != i and m_g[i][j] != np.inf:
                d_g[i][j]=m_g[i][j]

    return d_g

def d_g_2_m_g(d_g):
    """ Transformamos la representacion en diccionarios del grafo m_g a una
    version con matrices de adyacencia m_g
    :d_g: doble diccionario con los enlaces del grafo a transformar en matriz
    de adyacencia.
    :type d_g: diccionario de diccionarios.
    :return: grafo representado por una matriz de adyacencia.
    :return type: np 2 dimentional array.
    """
    l = len(d_g)
    m_g = np.full((l,l), np.inf)
    for i in d_g:
        for j in d_g[i]:
            m_g[i][j] = d_g[i][j]
    np.fill_diagonal(m_g,0)
    return m_g

def cuenta_ramas(m_g):
    """ Funcion que cuenta el numero de ramas de el grafo representado por
    una matriz de adyacencia.
    :m_g: matriz de adyacencia del grafo.
    :type m_g: np 2 dimentional array.
    :return: numero de ramas del grafos.
    :return type: int.
    """
    dim = m_g.shape[0]
    # Al tamanno total de la matriz (dim**2) le restamos
    # el numero de inf.
    # Restamos los ceros de la diagonal
    return dim**2-len(np.where(m_g == np.inf)[0])-dim

def check_sparse_factor(n_grafos, n_nodes, sparse_factor):
    """ Comprueba el funcionamiento correcto del parametro sparse_factor.
    Para ello creamos n_grafos con n_nodes en los que comprobamos el valor
    de cada sparse_factor y despues promediamos
    :n_grafos: numero de grafos a realizar.
    :n_nodes: numero de nodos que tendran los grafos.
    :sparse_factor: factor de dispersion de los grafos.
    :type n_grafos: int
    :type n_nodes:int
    :type sparse_factor:int
    :return: sparse factor resultante de hacer el promedio de los grafos
    :return type: float
    """
    acum=0
    for n in range(n_grafos):
        m=rand_matr_pos_graph(n_nodes,sparse_factor)
        sf=cuenta_ramas(m)/n_nodes**2

        acum+=sf
    return acum/n_grafos

def save_object(obj, f_name="obj.pklz", save_path='.'):
    """Guarda en binario el objeto pasado por argumento, con nombre f_name en
    el directorio save_path.
    :obj: objeto a serializar.
    :f_name: nombre del fichero serializado.
    :save_path: nombre de la carpeta donde se guardara el binario
    :type obj: Objeto
    :type f_name: str
    :type save_path: str
    """
    # Apertura del fichero comprimido en modo escritura
    file_path = os.path.join(save_path, f_name)
    with gzip.open(file_path, mode="wb", compresslevel=9) as file:
        # Volcado del objeto
        pickle.dump(obj, file, protocol=None)

def read_object(f_name, save_path='.'):
    """Carga un objeto serializado en el fichero f_name que se encuentra
    en la carpeta save_path.
    :f_name: nombre del fichero a cargar
    :save_path: nombre de la carpeta donde se encuentra el fichero.
    :type f_name: str
    :type save_path: str
    :return: el fichero leido
    :return type: tipo del fichero leido
    """
    # Apertura del fichero comprimido en modo lectura
    file_path = os.path.join(save_path, f_name)
    with gzip.open(file_path, mode="rb", compresslevel=9) as file:
        # Lectura del objeto
        return pickle.load(file)

def d_g_2_TGF(d_g, f_name):
    """ Recibe la lista de adyacencia de un grafo ponderado en formato doble
    diccionario. Y genera en el fichero f_name que guarda el grafo en formato TGF
    :d_g: grafo representado por un doble diccionario.
    :f_name: fichero destino del grafo, donde se guardara en formato TGF.
    :type d_g: doble diccionario.
    :type f_name: str
    """
    with open(f_name,'w') as f:
        for k in d_g:
            print(str(k), file=f)
        print('#', file=f)

        for d in d_g:
            for dd in d_g[d]:
                print("{0}\t{1}\t{2}".format(d, dd, d_g[d][dd]), file=f)

def TGF_2_d_g(f_name):
    """Funcion que carga fichero f_name, donde se encuentra un grafo representado
    en formato TFG y devuelve un diccionario de diccionario.
    :f_name: nombre del fichero donde se encuentra el grafo codificado en TGF
    :type f_name: str
    :return: lista de adyacencia
    :return type: diccionario de diccionarios ;{origen:{destino: peso}}
    """
    d_g = {}
    with open(f_name, 'r') as file:
        ls= file.readlines()
        i=0
        for l in ls:
            if l == '#\n':
                break
            d_g[int(l[:-1])] = {}
            i+=1
        for line in ls[i+1:]:
            rama = line[:-1].split('\t')
            d_g[int(rama[0])][int(rama[1])] = float(rama[2])
    return d_g

def dijkstra_d(d_g, u):
    """ Implementacion del algoritmo de Dijkstra con listas de adyacencia
    :d_g: Lista de adyacencia que representa el grafo
    :u: Nodo inicial
    :type d_g: dictionary, keys are integers, values are dictionarys
    :type u: int
    :return: d_dist lista de distancias minimas encontradas
    :return type d_dist: diccionario {int : float}
    :return: d_prev camino
    :return type d_prev: diccionario {int: int}
    """
    # Inicializacion
    s = {node: False for node in d_g}
    d_prev = {}
    d_dist = {node: np.inf for node in d_g}
    Q = PriorityQueue()
    d_dist[u] = 0
    Q.put((d_dist[u],u))
    while not Q.empty():
        d_v, v = Q.get()
        if not s[v]:
            s[v] = True
        for z in d_g[v]:
            if d_dist[z] > d_v + d_g[v][z]:
                d_dist[z] = d_v + d_g[v][z]
                d_prev[z] = v
                Q.put((d_dist[z],z))
    return d_dist,d_prev

def dijkstra_m(m_g, u):
    """ Implementacion del algoritmo de Dijkstra con matrices
    :u: Nodo inicial
    :type m_g: np 2 dimentional array
    :type u: int
    :return: d_dist lista de distancias minimas encontradas
    :return type d_dist: diccionario {int : float}
    :return: d_prev camino
    :return type d_prev: diccionario {int: int}
    """
    # Inicializacion
    n = m_g.shape[0]
    s = {i: False for i in range(n)}
    d_prev = {}
    d_dist = {i: np.inf for i in range(n)}
    Q = PriorityQueue()
    d_dist[u] = 0
    Q.put((d_dist[u],u))
    while not Q.empty():
        d_v, v = Q.get()
        if not s[v]:
            s[v] = True
        for z in range(n):
            if d_dist[z] > d_v + m_g[v][z]:
                d_dist[z] = d_v + m_g[v][z]
                d_prev[z] = v
                Q.put((d_dist[z],z))
    return d_dist,d_prev

def min_paths(d_prev):
    """ TODO
    :d_prev: diccionario con el nodo previo a cada nodo
    :d_prev type: diccionario
    :return: diccionario de listas con el camino desde cada nodo inicial a cada nodo
    """
    d_path = {}
    for node in d_prev:
        node2 = d_prev[node]
        d_path[node] = [node2, node]
        while node2 in d_prev:
            node2 = d_prev[node2]
            d_path[node].insert(0,node2)
    d_path[node2] = [node2]
    return d_path


def time_dijkstra_m(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):
    """ Genera un conjunto de grafos con distintos numeros de nodos para devolver
    el coste teorico temporal del algoritmo de Dijkstra.
    :n_grafos: numero de grafos a generar
    :n_nodes_ini: numero inicial de nodos del grafo
    :n_nodes_fin: numero final de nodos del grafo
    :step: paso en cada iteracion de numero de nodos
    :sparse_factor: factor de dispersion de la matriz
    :type n_grafos: int
    :type n_nodes_ini: int
    :type n_nodes_fin: int
    :type step: int
    :type sparse_factor: float
    :return: time_l lista con los tiempos en cada paso
    """
    time_l = []
    n = n_nodes_ini
    while n <= n_nodes_fin:
        t = 0
        for _ in range(n_graphs):
            m_g = rand_matr_pos_graph(n, sparse_factor)
            for i in range(n):
                ini = time.time()
                dijkstra_m(m_g,i)
                fin = time.time()
                t+=(fin-ini)
        time_l.append(t/(n_graphs*n))
        n+=step
    return time_l

def time_dijkstra_d(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):
    """ Genera un conjunto de grafos con distintos numeros de nodos para devolver
    el coste teorico temporal del algoritmo de Dijkstra.
    :n_grafos: numero de grafos a generar
    :n_nodes_ini: numero inicial de nodos del grafo
    :n_nodes_fin: numero final de nodos del grafo
    :step: paso en cada iteracion de numero de nodos
    :sparse_factor: factor de dispersion de la matriz
    :type n_grafos: int
    :type n_nodes_ini: int
    :type n_nodes_fin: int
    :type step: int
    :type sparse_factor: float
    :return: time_l lista de los tiempos en cada paso
    """

    time_l = []
    n = n_nodes_ini
    while n <= n_nodes_fin:
        t = 0
        for _ in range(n_graphs):
            m_g = rand_matr_pos_graph(n, sparse_factor)
            d_g= m_g_2_d_g(m_g)
            for i in range(n):
                ini = time.time()
                dijkstra_d(d_g,i)
                fin = time.time()
                t+=(fin-ini)
        time_l.append(t/(n_graphs*n))
        n+=step
    return time_l

def d_g_2_nx_g(d_g):
    """ Transforma un grafo de la representacion de doble diccionario a la
    la usada por la biblioteca nx.
    :d_g: diccionario a transformar en el formato nx; tripe diccionario
    :d_g type: diccionario de diccionario, las claves del primer diccionario son
    los nodos origen y las del segundo los nodos destino. Los valores del
    diccionario mas interno son el peso de cada rama. Los nodos son valores enteros
    los pesos en float
    :return: grafo nx resultante
    :return type: NetworkX graph
    """
    d_g_nx = nx.DiGraph()
    d_g_nx.add_nodes_from(d_g.keys())
    d_g_nx.add_weighted_edges_from([(u, v, d_g[u][v]) for u in d_g for v in d_g[u]])
    return d_g_nx

def nx_g_2_d_g(nx_g):
    """ Transforma un grafo nx a la representacion de doble diccionario
    :nx_g: grafo de nx
    :nx_g type: NetworkX graph
    :return: grafo en nuestra implementacion de doble diccionario
    :return type: dict con claves int valor dict con claves int valor float
    """
    l_e = nx_g.edges(data = True)
    d_g = {}
    for u, v, w in l_e:
        try:
            d_g[u][v]=w['weight']
        except KeyError:
            d_g[u]={v: w['weight']}
    return d_g

def time_dijkstra_nx(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):
    """ Genera un conjunto de grafos con distintos numeros de nodos para comprobar
    el coste teorico del algoritmo de Dijkstra implementado en la biblioteca Networkx.
    :n_graphs: numero de grafos a generar
    :n_nodes_ini: numero inicial de nodos del grafo
    :n_nodes_fin: numero final de nodos del grafo
    :step: paso en cada iteracion de numero de nodos
    :sparse_factor: factor de dispersion del grafo
    :type n_grafos: int
    :type n_nodes_ini: int
    :type n_nodes_fin: int
    :type step: int
    :type sparse_factor: float
    :return: time_l lista con los tiempos en cada paso
    :return type: lista de float
    """
    time_l = []
    n = n_nodes_ini
    while n <= n_nodes_fin:
        t = 0
        for _ in range(n_graphs):
            m_g = rand_matr_pos_graph(n, sparse_factor)
            d_g= m_g_2_d_g(m_g)
            nx_g = d_g_2_nx_g(d_g)
            for i in range(n):
                ini = time.time()
                nx.single_source_dijkstra(nx_g, i)
                fin = time.time()
                t+=(fin-ini)
        time_l.append(t/(n_graphs*n))
        n+=step
    return time_l
