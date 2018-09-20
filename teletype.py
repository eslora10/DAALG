import numpy as np

def rand_matr_pos_graph(n_nodes, sparse_factor, max_weight=50., decimals=0):
    """
    """
    m=np.random.random((n_nodes, n_nodes))
    np.place(m,m<1-sparse_factor,np.inf)
    m=m*100-(100-max_weight)
    np.fill_diagonal(m,0)
    return m

def m_g_2_d_g(m_g):
    """
    """
    pass


def d_g_2_m_g(d_g):
    """
    """
    pass

if __name__ == "__main__":
    print(rand_matr_pos_graph(3, 0.3))
