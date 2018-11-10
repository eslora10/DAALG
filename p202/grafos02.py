import numpy as np
import copy
import random

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
    #n=len(d_mg) no funciona en caso de que el diccionario no tenga todos los nodos

    s=set()
    for u in d_mg:
        s.add(u)
        for e in d_mg[u].keys():
            s.add(e)
    n=max(s)+1
    adj=[0 for _ in range(n)]
    inc=[0 for _ in range(n)]
    for u in d_mg.keys():
        for v in d_mg[u].keys():
            inc[v]+=len(d_mg[u][v])
            adj[u]+=len(d_mg[u][v])
    return adj, inc

print(adj_inc_directed_multigraph(d_mg))

#prueba = np.array([[0,1,1],[0,0,1],[1,0,0]])
prueba = np.array([[0,1,1,1,0],[1,0,1,1,0],[1,1,0,1,1],[1,1,1,0,1],[0,0,1,1,0]])
prueba_d = m_mg_2_d_mg(prueba)
print(prueba)

def my_is_there_euler_path_directed_multigraph(d_gm):
    adj, inc = adj_inc_directed_multigraph(d_gm)
    cmp = [adj[i]==inc[i] for i in range(len(adj))]
    ini = False
    fin = False
    iniNode = list(filter(lambda x: x>0, adj))[0]
    finNode = iniNode
    for node in np.where(np.array(cmp)==False)[0]:
        if inc[node]==adj[node]-1 and not ini:
            ini = True
            iniNode=node
        elif inc[node]==adj[node]+1 and not fin:
            fin = True
            finNode = node
        else:
            return False, tuple()

    return True, (iniNode, finNode)

def is_there_euler_path_directed_multigraph(d_mg):
    ''' Comprueba si existen caminos eulerianos en un multigrafo dirigido
    :d_mg: multigrafo dirigido
    :type d_mg: dict of dict of dict
    :return: True si hay caminos eulerianos, False en caso contrario
    :type return: bool
    '''
    return my_is_there_euler_path_directed_multigraph(d_mg)[0]


print("RES:", is_there_euler_path_directed_multigraph(prueba_d))

def first_last_euler_path_directed_multigraph(d_mg):
    ''' devuelve el punto inicial y el final de un camino euleriano del multigrafo
    d_mg EN CASO DE QUE EXISTA dicho camino. de no ser asi devuelve una tupla vacia
    :d_mg: multigrafo dirigido
    :type d_mg: dict of dict of dict
    :return: (ini,fin)
    :return type: tupla con el nodo inicial y final
    '''
    return my_is_there_euler_path_directed_multigraph(d_mg)[1]


print(first_last_euler_path_directed_multigraph(prueba_d))

def euler_walk_directed_multigraph(u, d_mg):
    '''Antes de usar esta funcion debe comprobarse si existe un camino eulerianos
    y debe iniciarse en el nodo correcto u. De no ser asi, esta funcion de devolvera
    un resultado correcto.
    Ademas, esta funcion es DESTRUCTIVA. recomendamos utilizar una copia del diccionario
    con copy.deepcopy
    :u:nodo inicial en el que se inicie el paseo
    :d_mg: multigrafo dirigido
    :type u: int
    :type d_mg: dict of dict of dict
    :return:
    :return type:
    '''
    pi = [u]
    while u in d_mg:
        v = sorted(d_mg[u].keys())[0]
        pi.append(v)
        d_mg[u][v].pop(max(d_mg[u][v]))
        if not d_mg[u][v]:
            d_mg[u].pop(v)
        if not d_mg[u]:
            d_mg.pop(u)
        u = v
    return pi

def next_first_node(l_path, d_mg):
    ''' Devuelve el nodo en el que continuar el camino euleriano.
    :l_path: camino euleriano actual desde el que debemos continuar. Se escoge el
        nodo del que debemos encontrar un circuito euleriano.
    :d_mg: representacion del grafo resultante tras las anteriores iteraciones del
        algoritmo
    :type l_path: list
    :type d_mg: dict of dict of dict
    :return: nodo en el que se debe iniciar un nuevo camino.
    :return type: integer
    '''
    adj, inc = adj_inc_directed_multigraph(d_mg)
    for node in l_path:
        if adj[node]:
            return node

def path_stitch(path_1, path_2):
    ''' Pega path_2 en path_1 de manera correcta.
    :path_1: Camino inicial
    :path_2: Camino a adherir
    :type path_1: list of integers
    :type path_2: list of integers
    :return: camino resultante
    :return type: list of integers
    '''
    u = path_2[0]
    idx = path_1.index(u)
    return path_1[:idx]+path_2+path_1[idx+1:]

def euler_path_directed_multigraph(d_mg):
    ''' Busca un camino euleriano en un grafo
    :d_mg: multigrafo dirigido
    :type d_mg: dict of dict of dict
    :return: lista con el camino
    :return type: list
    '''
    t = first_last_euler_path_directed_multigraph(d_mg)
    if t:
        u = t[0]
        dcopy = copy.deepcopy(d_mg)
        path = euler_walk_directed_multigraph(u, dcopy)
        while dcopy:
            u = next_first_node(path, dcopy)
            path = path_stitch(path, euler_walk_directed_multigraph(u, dcopy))
    else:
        path = []
    return path

print(euler_path_directed_multigraph(prueba_d))

def isthere_euler_circuit_directed_multigraph(d_mg):
    '''Comprueba si existen circuitos eulerianos en un multigrafo dirigido
    :d_mg: multigrafo dirigido
    :type d_mg: dict of dict of dict
    :return: True si hay circuitos eulerianos, False en caso contrario
    :type return: bool
    '''
    adj, inc = adj_inc_directed_multigraph(d_mg)
    cmp = [adj[i]!=inc[i] for i in range(len(adj))]
    return sum(cmp)==0

def euler_circuit_directed_multigraph(d_mg, u=0):
    '''
    '''
    if isthere_euler_circuit_directed_multigraph(d_mg):
        dcopy = copy.deepcopy(d_mg)
        v = sorted(d_mg[u].keys())[0]
        dcopy[u][v].pop(max(dcopy[u][v]))
        if not dcopy[u][v]:
            dcopy[u].pop(v)
        if not dcopy[u]:
            dcopy.pop(u)
        path = euler_path_directed_multigraph(dcopy)
        return [u]+path
    else:
        return []

print(euler_circuit_directed_multigraph(prueba_d))


###########################SECUENCIACION DE LECTURAS###########################

def random_sequence(len_seq):
    '''Genera una secuencia aleatoria con los caracteres (A, C, G, T) de longitud len_seq
    :len_seq: longitud de la secuencia
    :type len_seq: int
    :return: secuencia
    :return type: str
    '''
    d={0:'A',1:'C',2:'G',3:'T'}

    l=np.random.randint(0,4,len_seq)
    return ''.join([d[x] for x in l])

def spectrum(sequence, len_read):
    '''Genera el espectro de la cadena sequence de longitud len_seq.
    Devuelve una lista con las lecturas de longitud len_read desordenadas
    :sequence: secuencia
    :len_read: tamaño de las lecturas
    :type sequence: str
    :type len_read: int
    :return:lista con las lecturas de longitud len_read desordenadas
    :return type: list
    '''
    l = {sequence[i:i+len_read] for i in range(len(sequence)-len_read+1)}
    """
    l=set([])
    for i in range(len(sequence)-len_read+1):
        l.add(sequence[i:i+len_read])
    """
    l = list(l)
    random.shuffle(l)
    return l

def spectrum_2(spectr):
    '''Genera el (l-1)-espectro a partir del espectro pasado por argumento
    :spectr: espectro
    :type spectr: list
    :return: lista con el (l-1) espectro
    :return type: lista de cadenas de longitud l-1
    '''
    long=len(spectr[0])
    l=set()

    for s in spectr:
        for i in range(long-1):
            l.add(s[i:i+long-1])
    l = list(l)
    random.shuffle(l)
    return l

def spectrum_2_undirected_graph(spectr):
    ''' Devuelve el multigrafo dirigido formado a partir del espectro
    :spectr: espectro
    :type spectr: list
    :return: multigrafo dirigido
    :return type: dict of dict of dict
    '''
    # Generamos el (l-1)-espectro a partir de spectr
    l_spectr = spectrum_2(spectr)
    d_mg = {}
    i = 0
    for seq in l_spectr:
        # Conseguir todas las palabras que empiezan por seq
        # Quedarnos con el final de la palabra
        next = [l_spectr.index(s[1:]) for s in spectr if s[:len(seq)]==seq]
        print(i, next)
        if next:
            nodes, nedges = np.unique(next, return_counts=True)
            d_mg[i] = {j: {n: 1 for n in nedges} for j in nodes}
        i+=1
    return d_mg

#seq=random_sequence(9)
seq='TAAAGTGTC'
print(seq)
spec=spectrum(seq, 3)
print(spec)
spec2=spectrum_2(spec)
print(spec2)
print('generamos grafo a partir de spec:')
d_mg = spectrum_2_undirected_graph(spec)
print_multi_graph(d_mg)
