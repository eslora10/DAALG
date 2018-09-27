import numpy as np

def m_g_2_d_g(m_g):
    """

    TODO: ¿Hay que eliminar los 0's de lso dict?

    TODO: ¿Mejor forma de hacerlo?¿pythonic way?
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
    """
    """
    l = len(d_g)
    m_g = np.full((l,l), np.inf)
    for i in d_g:
        for j in d_g[i]:
            m_g[i][j] = d_g[i][j]
    np.fill_diagonal(m_g,0)
    return m_g

def rand_matr_pos_graph(n_nodes, sparse_factor, max_weight=50., decimals=0):
    """
    """
    m=np.random.random((n_nodes, n_nodes))
    np.place(m,m<1-sparse_factor,np.inf)
    m=m*100-(100-max_weight)
    np.fill_diagonal(m,0)
    return m


def cuenta_ramas(m_g):
    """
    """
    dim = m_g.shape[0]
    print(m_g)
    print (np.where(m_g == np.inf))
    # Al tamanno total de la matriz (dim**2) le restamos
    # el numero de inf.
    # Restamos los ceros de la diagonal
    return dim**2-len(np.where(m_g == np.inf)[0])-dim


def check_sparse_factor(n_grafos, n_nodes, sparse_factor):
    """
     @param n_grafos: numero de grafos a crear
     @param n_nodes: numero de nodos de los grafos
     @param sparse_factor: factor de dispersion de la matriz

     @return sparse_factor real medio

    """
    acum=0
    for n in range(n_grafos):
        m=rand_matr_pos_graph(n_nodes,sparse_factor)
        sf=cuenta_ramas(m)/n_nodes**2

        acum+=sf
    return acum/n_grafos

def save_object(obj, f_name="obj.pklz", save_path='.'):
    """"""
    # Apertura del fichero comprimido en modo escritura
    file_path = os.path.join(save_path, f_name)
    file = gzip.open(file_path, mode="wb", compresslevel=9)
    # Volcado del objeto
    pickle.dump(obj, file, protocol=None)


def read_object(f_name, save_path='.'):
    """"""
    # Apertura del fichero comprimido en modo lectura
    file_path = os.path.join(save_path, f_name)
    file = gzip.open(file_path, mode="rb", compresslevel=9)
    # Lectura del objeto
    pickle.load(file)

def d_g_2_TGF(d_g, f_name):
    """
    """

    with open(f_name,w ) as f:
        for k in d_g:
            f.write(k)
        f.write('#')

        for d in d_g:

            for dd in d:
                f.write(d+'\t'+dd+'\t'+d_g[d][dd])


def TGF_2_d_g(f_name):
    """
    """
    d_g = {}
    with open(f_name) as file:
        while (l = file.readline()[:-1]) != '#':
            d_g[int(l)] = {}
        for line in file:
            rama = line[:-1].split('\t')
            d_g[int(rama[0])][int(rama[1])] = float(rama[2])
    return d_g

if __name__ == "__main__":
    print(rand_matr_pos_graph(3, 0.3))


    print(check_sparse_factor(3,3,0.5))


    m=rand_matr_pos_graph(10, .5)

    d_g_2_TGF(m_g_2_d_g(m),'prueba.txt')
