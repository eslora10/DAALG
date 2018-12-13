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
    pivot = t[ini]
    m = ini
    for i in range(ini, fin+1):
        if t[i] < pivot:
            m+=1
            t[i], t[m] = t[m], t[i]

    t[ini], t[m] = t[m], t[ini]
    return m

t = [3,2,1,4,5]
print(split(t, 0, 4))
print(t)

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
    if not pivot:
        return split(t, ini, fin)
    m = ini

    for i in range(ini, fin+1):
        if t[i] < pivot:
            m+=1
            print(t[i], i, m , pivot)
            print(t)
            t[i], t[m] = t[m], t[i]

    t[ini], t[m] = t[m], t[ini]
    assert (t[m] == pivot), "El pivote no esta en la tabla"

    return m

t = [3,2,1,4,5]
print(split_pivot(t, 0, 4, 4))
print(t)

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

    m = split_pivot(t, ini, fin)

    if ind == fin-ini+1:
        return (t[m], m)
    elif ind < m-ini+1:
        return qselect(t, ini, m-1, ind, pivot)
    else:
        return qselect(t, m+1, fin, ind-(m-ini), pivot)

t = [3,4,5,1,8,10,15,24,12]
print(sorted(t))
print(qselect(t,0,6,4))

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
    pass

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
    pass



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
    pass

def qsort_5(t, ini, fin):
    """Ordena la tabla t entre las posiciones ini y fin ( inclusive)

    :t : tabla
    :type t: list
    :ini : indice inicial a ordenar
    :type ini: int
    :fin : indice final a ordenar
    :type fin: int

    :return: tabla ordenada entre ini y fin
    :return type: list

    """
    pass
