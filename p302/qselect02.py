def split(t, ini, fin):
    """Divide los elementos de la tabla t entre indices ini y fin usando como
    pivote el primer elemento

    :t : tabla
    :type t: list
    :ini : indice inicial de la tabla
    :type ini: int
    :fin : indice final de la tabla
    :type fin: int

    :return: indice por el que dividimos
    :return type: int
    """
    assert (fin >= ini),"El indice final debe ser mayor que el inicial"
    assert (ini >= 0),"El inicio debe ser como minimo 0"
    assert (fin < len(t)), "No se puede buscar por encima del final de la tabla"

    pivot = t[ini]
    m = ini
    for i in range(ini+1, fin+1):
        if t[i] <= pivot:
            m+=1
            t[i], t[m] = t[m], t[i]

    t[ini], t[m] = t[m], t[ini]
    return m


def split_pivot(t, ini, fin, pivot=None):
    """Divide los elementos de la tabla t entre indices ini y fin usando como
    pivote el primer elemento

    :t : tabla
    :type t: list
    :ini : indice inicial de la tabla
    :type ini: int
    :fin : indice final de la tabla
    :type fin: int
    :pivot : indice del pivote, si es none, se utiliza como pivote el primer elemento
    :type pivot: int

    :return: el indice del pivote, la tabla t se modifica
    :return type: int
    """
    assert (fin >= ini),"El indice final debe ser mayor que el inicial"
    assert (ini >= 0),"El inicio debe ser como minimo 0"
    assert (fin < len(t)), "No se puede buscar por encima del final de la tabla"

    if pivot is None:
        return split(t, ini, fin)

    m = ini
    i_pivot = -1

    for i in range(ini, fin+1):
        if t[i] < pivot:
            t[i], t[m] = t[m], t[i]
            m+=1
        elif t[i]==pivot:
            t[i], t[m] = t[m], t[i]
            i_pivot = m
            m+=1

    assert (i_pivot >= 0), "El pivote no esta en la tabla"
    t[i_pivot], t[m-1] = t[m-1], t[i_pivot]

    return m-1


def qselect(t, ini, fin, ind, pivot=None):
    """Funcion que aplica el algoritmo de qselect para encontrar el valor del
    elemento en la posicion ind si la tabla estuviese ordenada y devuelve dicho
    valor y la posicion donde se encuentra tras aplicar el algoritmo.

    :t : tabla
    :type t: list
    :ini : indice inicial de la tabla
    :type ini: int
    :fin : indice final de la tabla
    :type fin: int
    :ind : indice del elemento buscado
    :type ind: int
    :pivot : funcion que calcula el pivote
    :type pivot: function

    :return: elemento de la posicion ind si la tabla estuviese ordenada
    :return type: int
    :return: posicion donde se enuentra dicho valor tras aplicar qselect
    :return type: int

    """
    assert (ind <= fin-ini+1 or ind >= 0), "Los indices no son correctos"

    m = split(t, ini, fin)
    if ind == m-ini:
        return t[m]
    elif ind < m-ini:
        return qselect(t, ini, m-1, ind, pivot)
    else:
        return qselect(t, m+1, fin, ind-(m-ini+1), pivot)


def qselect_sr(t, ini, fin, ind, pivot=None):
    """Implementacion de qselect sin recursion de cola

    :t : tabla
    :type t: list
    :ini : indice inicial de la tabla
    :type ini: int
    :fin : indice final de la tabla
    :type fin: int
    :ind : indice del elemento buscado
    :type ind: int
    :pivot : indice del pivote
    :type pivot: int

    :return: elemento de la posicion ind si la tabla estuviese ordenada
    :return type: int
    :return: posicion donde se enuentra dicho valor tras aplicar qselect
    :return type: int

    """
    assert (ind <= fin-ini+1 or ind >= 0), "Los indices no son correctos"

    m = split(t, ini, fin)
    while not (ind == m-ini):
        if ind < m-ini:
            return qselect_sr(t, ini, m-1, ind, pivot)
        elif ind > m-ini:
            ind = ind-(m-ini+1)
            ini = m+1
            m = split(t, ini, fin)
    return t[m]


def _mediana(t):
    sortT=sorted(t)
    l=len(sortT)
    return sortT[(l-1)//2]

def pivot_5(t, ini, fin):
    """Funcion que devuelve el 'pivote 5' de una tabla mediante el procedimiento
    de mediana de medias.

    :t : tabla
    :type t: list
    :ini : indice inicial de la tabla
    :type ini: int
    :fin : indice final de la tabla
    :type fin: int

    :return: devuelve el indice del 'pivote 5'
    :return type: int

    """
    medianas=[]
    for i in range(ini,fin+1,5):

        if fin-i < 5:
            m = _mediana(t[i:fin+1])
        else:
            m = _mediana(t[i:i+5])

        medianas.append(m)

    return _mediana(medianas)


def qselect_5(t, ini, fin, pos):
    """Devuelve el elemento de la tabla t que se encuentre en la poscion pos
    mendiante la 'variante 5' del algoritmo qselect

    :t : tabla
    :type t: list
    :ini : indice inicial de la tabla
    :type ini: int
    :fin : indice final de la tabla
    :type fin: int
    :pos : posicion del elemento a devolver
    :type pos: ind

    :return: Devuelve el elemento de la posicion pos
    :return type: int

    """
    pivot = pivot_5(t, ini, fin)
    m = split_pivot(t, ini, fin, pivot=pivot)
    while not (pos == m-ini):
        if pos < m-ini:
            return qselect_5(t, ini, m-1, pos)
        elif pos > m-ini:
            pos = pos-(m-ini+1)
            ini = m+1
            pivot = pivot_5(t, ini, fin)
            m = split_pivot(t, ini, fin, pivot=pivot)
    return t[m]


def qsort_5(t, ini, fin, verbose=False):
    """Ordena la tabla t entre las posiciones ini y fin ( inclusive)
    Es una version de Quicksort con caso peor O(N lg(N))

    :t : tabla
    :type t: list
    :ini : indice inicial a ordenar
    :type ini: int
    :fin : indice final a ordenar
    :type fin: int

    :return: tabla ordenada entre ini y fin
    :return type: list

    """
    if ini == fin:
        return t[ini:fin+1]

    pivot = pivot_5(t, ini, fin)
    m = split_pivot(t, ini, fin, pivot=pivot)
    if ini < m-1:
        qsort_5(t, ini, m-1)
    if m+1 < fin:
        qsort_5(t, m+1, fin)

    return t[ini:fin+1]
