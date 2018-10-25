import numpy as np
import copy

def rand_matr_pos_multi_graph(n_nodes, prob, num_max_multiple_edges=3):
    """ Genera una matriz de adyacencia para un multigrafo aleatoria de tamaño
    n_nodes x n_nodes, con un factor de dispersion de sparse_factor. En la que
    el valor maximo viene especificado por max_weight y en la que los elementos
    tienen tantos decimales como decimales
    :n_nodes: numero de nodos del grafos
    :prob: factor de dispersion de la matriz
    :max_weight: valor maximo de los pesos de la matriz
    :decimals: numero de decimales que queremos que tengan los elementos de la matriz
    :type n_nodes: int
    :type prob: float in [0,1]
    :type max_weight: float
    :type decimals: int
    :return: matriz de adyacencia que representa un grafos
    :return type: np 2 dimentional array
    """
    return np.random.binomial(num_max_multiple_edges, prob, size=(n_nodes, n_nodes))

m_mg = rand_matr_pos_multi_graph(3, 0.5)
print(m_mg)

def m_mg_2_d_mg(m_mg):
    ''' Dada una matriz de adyacencia, la convierte en un diccionario de multigrafo.
    :m_mg: matriz de adyacencia
    :type m_mg: numpy array
    :return d_mg: diccionario de multigrafo
    :type return d_mg: dict of dict of dict
    '''
    d_mg = {}
    n = m_mg.shape[0]
    for i in range(n):
        for j in range(n):
            if m_mg[i][j]:
                try:
                    d_mg [i][j] = {x: 1 for x in range(m_mg[i][j])}
                except KeyError:
                    d_mg [i] = {j: {x: 1 for x in range(m_mg[i][j])}}
    return d_mg

d_mg = m_mg_2_d_mg(m_mg)
print(d_mg)

def rand_unweighted_multigraph(n_nodes, num_max_multiple_edges=3, prob=0.5):
    ''' Funcion que genera un multigrafo en forma de diccionario sin pesos con
    n_nodes node, en el que el numero maximo de ramas por nodo es num_max_multiple_edges
    y con una probabilidad prob de generar un solo enlace
    :n_nodes: numero de nodos del multigrafo
    :num_max_multiple_edges: numero maximo de ramas entre dos nodos
    :prob: probabilidad de tener una sola rama entre dos nodos
    :type n_nodes: int
    :type num_max_multiple_edges: int
    :type prob: float in [0,1]
    '''

    m_mg=rand_matr_pos_multi_graph(n_nodes, prob, num_max_multiple_edges)
    return m_mg_2_d_mg(m_mg)

def graph_2_multigraph(g):
    ''' Funcion que genera un multigrado con la implementacion de tripre diccionario
    a partir de un grafo con la antigua implementacion de doble diccionario
    :g: grafo
    :type g: {node origen: {node destino: valor}}
    :return: multigrafo
    :return type: {origen: {destino: {num_edges: num}}}
    '''
    d_mg={}
    for u in g:
        d_mg[u]={}
        for v in g[u]:
            d_mg[u][v]={}
            d_mg[u][v][0]=1

    return d_mg

def print_multi_graph(g):
    ''' Dado un multigrafo imprime cada una de sus ramas como (u,v) junto con las
    ramas del multigrafo.
    :g: multigrafo
    :type g: dict of dict of dict
    '''
    for u in g:
        for v in g[u]:
            s = "({0}, {1})".format(u,v)
            for e in g[u][v]:
                if g[u][v][e]:
                    s+=" {0}: {1}".format(e, g[u][v][e])
            print(s)

print_multi_graph(d_mg)

def adj_inc_directed_multigraph(d_mg):
    ''' Funcion que devuelve la lista de adyacencia adj y la lista de incidencia inc
    a partir de un multigrafo dado
    :d_mg: multigrafo
    :type d_mg: dict of dict of dict
    :return:
        adj: lista de adyacencia del multigrado
        inc: lista de incidencia del multigrafo
    :return types:
        adj: list
        inc: list
    '''
    n=len(d_mg)
    adj=[0 for _ in range(n)]
    inc=[0 for _ in range(n)]
    for u in d_mg.keys():
        for v in d_mg[u].keys():
            inc[v]+=len(d_mg[u][v])
            adj[u]+=len(d_mg[u][v])
    return adj, inc

print(adj_inc_directed_multigraph(d_mg))

prueba = np.array([[0,1,1],[0,0,1],[1,0,0]])
prueba_d = m_mg_2_d_mg(prueba)
print(prueba)

def is_there_euler_path_directed_multigraph(d_gm):
    ''' Comprueba si existen caminos eulerianos en un multigrafo dirigido
    :d_mg: multigrafo dirigido
    :type d_mg: dict of dict of dict
    :return: True si hay caminos eulerianos, False en caso contrario
    :type return: bool
    '''
    adj, inc = adj_inc_directed_multigraph(d_gm)
    cmp = [adj[i]!=inc[i] for i in range(len(adj))]

    print(cmp)

print(is_there_euler_path_directed_multigraph(prueba_d))

def first_last_euler_path_directed_multigraph(d_mg):
    ''' devuelve el punto inicial y el final de un camino euleriano del multigrafo
    d_mg EN CASO DE QUE EXISTA dicho camino. de no ser asi devuelve una tupla vacia
    :d_mg: multigrafo dirigido
    :type d_mg: dict of dict of dict
    :return: (ini,fin)
    :return type: tupla con el nodo inicial y final
    '''


    if is_there_euler_path_directed_multigraph(d_mg):
        adj, inc=adj_inc_directed_multigraph(d_mg)
        return adj.index(min(adj)), inc.index(max(inc))
    else:
        return tuple()

print(first_last_euler_path_directed_multigraph(d_mg))

def euler_walk_directed_multigraph(u, d_mg):
    '''

    :u:nodo inicial en el que se inicie el paseo
    :d_mg: multigrafo dirigido
    :type u: int
    :type d_mg: dict of dict of dict
    :return:
    :return type:
    '''
    dcopy= copy.deepcopy(d_mg)
    pass

def next_first_node(l_path, d_mg):
    '''

    :l_path:
    :d_mg:
    :return:
    :return type:
    '''
    pass
