import numpy as np

import time


# TODO divisiones con // para que sean de enteros


################### I-A ###################
def _len2k(t,n):

    K = int(np.ceil(np.log2(n)))
    N = 2**K
    return np.concatenate((t, [0 for _ in range(N-len(t))]))

def fft(t):
    ''' Calcula la transformada de Fourier discreta de la tabla T

    :t: tabla sobre la que se aplica fft
    :type t: array de enteros

    :return: la transformada de Fourier de t
    :return type: array de numeros complejos
    '''
    K = int(np.ceil(np.log2(len(t))))
    N = 2**K
    # Caso base recursion
    if K==0:
        return t
    # Raices del polinommio ciclotomico
    r = [np.cos(2*np.pi*i/N)+1j*np.sin(2*np.pi*i/N) for i in range(N)]

    # Rellenar con 0 para obtener una potencia de 2
    #t = t + [0 for _ in range(N-len(t))]
    t = _len2k(t, len(t))

    # Particion
    t_even = []
    t_odd = []
    for i in range(N):
        if i%2==0:
            t_even.append(t[i])
        else:
            t_odd.append(t[i])

    # FFT de las subtablas (llamada recursiva)
    t_even = fft(t_even)
    t_odd = fft(t_odd)

    # Combinar subtablas
    t_ret = []
    n2=int(N/2)
    for j in range(N):
        t_ret.append(t_even[j%n2]+r[j]*t_odd[j%n2])

    return t_ret


def invert_fft(t, fft_func=fft):
    ''' Calcula la inversa de una transformada de Fourier

    :t: tabla sobre la que se aplica la inversa
    :type t: array de numeros complejos
    :fft_func: funcion que aplica para calcular la DFT
    :type fft_func: callable

    :return: inversa de t
    :return type: array de numeros complejos
    '''

    tconj=[x.conjugate() for x in t]

    t=fft_func(tconj)
    N = len(t)
    tinv=[1/N *x.conjugate() for x in t]

    return tinv

################### I-B ###################

def rand_polinomio(long=2**10, base=10):
    ''' Genera una lista de enteros que representa los coeficientes de un polinomio

    :long: longitud de la lista a devolver
    :type long: int
    :base: los coeficientes van de 0 a base-1; base in [2,10] inclusive
    :type base: int

    :return: coeficientes del polinomio
    :return type: lista de ints de python
    '''

    return np.random.randint(0, base, long)


def poli_2_num(l_pol, base=10):
    ''' Transforma un polinomio en un numero (en base base) evaluandolo los coeficientes del
    del polinomio expresado en la lista l_pol. Para ello se utiliza la la regla de Horner

    :l_pol: coeficientes del polinomio
    :type l_pol: lista
    :base: numero en el que se evalua el polinomio
    :type base: int

    :return: numero resultante de evaluar
    :return type: int
    '''
    act=0
    for coef in reversed(l_pol):
        res=int(coef)+act
        act=res*base
    return res

def rand_numero(num_digits, base=10):
    ''' Genera un numero aleatorio de num_digits cifras

    :num_digits: numero de cifras
    :type num_digits: int
    :base: base en la que se representa el numero, por defecto decimal
    :type base: int

    :return: numero aleatorio de num_digits cifras
    :return type: int
    '''
    # Generamos un polinomio con num_digits coeficientes

    pol=rand_polinomio(long=num_digits, base=base)

    # Evaluamos en la base
    return poli_2_num(pol, base=base)

def num_2_poli(num, base=10):
    ''' Devuelve un polinomio (en formato lista) con los coeficientes de
    correspondientes al numero expresado en base base

    :num: numero a transdormar en polinomio
    :type num: int
    :base: base en la que representamos el numero
    :type base: int

    :return: polinomio (lista de coeficientes x^0..... x^n)
    :return type: list
    '''
    pol = []
    while num > 0:
        num, res = divmod(num, base)
        pol.append(res)
    return pol

def mult_polinomios(l_pol_1, l_pol_2):
    ''' Algoritmo de miltiplicacion habitual. Como los polinomios entan
    ordenados por potencias crecientes, la multiplicacion es de izq a der

    :l_pol_1: polinomio 1
    :type l_pol_1: list
    :l_pol_2: polinomio 2
    :type l_pol_2: list

    :return: lista de coeficientes resultantes de la multiplicacion
    :return type: list
    '''
    n = len(l_pol_1) # grado pol_1 +1
    m = len(l_pol_2) # grado pol_2 +1
    res = [0 for _ in range(n+m-1)]

    for i in range(n):
        for j in range(m):
            res[i+j] += l_pol_1[i]*l_pol_2[j]

    return np.array(res)

def mult_polinomios_fft(l_pol_1, l_pol_2, fft_func=fft):
    ''' Multiplicacion de polinomios con fft

    :l_pol_1: polinomio 1
    :type l_pol_1: list
    :l_pol_2: polinomio 2
    :type l_pol_2: list
    :fft_func: funcion fft
    :type fft_func: funcion

    :return: polinomio resultado de multiplicar
    :return type: lista de ints
    '''
    # DFT de ambos polinomios
    dg = len(l_pol_1)+len(l_pol_2)-1
    pol1 = _len2k(l_pol_1, dg)
    pol2 = _len2k(l_pol_2, dg)

    coef1=fft_func(pol1)
    coef2=fft_func(pol2)

    # Multiplicacion de coeficientes
    for i in range(len(coef1)):
        coef1[i] *= coef2[i]

    # Inversa de la DFT
    l = invert_fft(coef1, fft_func=fft_func)
    return np.rint(l)

def mult_numeros(num1, num2):
    ''' multiplica los numeros num1 y num2  llevamdolos a mult_polinomios
    y multplicandolos con la funcion mult_polinomios. Despues recupera el numero
    resultante

    :num1: numero 1
    :type num1: int
    :num2: numero 2
    :type num2: int

    :return: numero resultante de multiplicar polinomios
    :return type: int
    '''
    return poli_2_num(mult_polinomios(num_2_poli(num1), num_2_poli(num2)))

def mult_numeros_fft(num1, num2, fft_func=fft):
    ''' Multiplica dos numeros mediante fft

    :num1: numero 1
    :type num1: int
    :num2: numero 2
    :type num2: int
    :fft_func: funcion fft
    :type fft_func: function

    :return: numero resultante
    :return type: int
    '''
    return int(poli_2_num(mult_polinomios_fft(num_2_poli(num1), num_2_poli(num2), fft_func=fft_func)))

################### I-C ###################

def time_mult_numeros(n_pairs, num_digits_ini, num_digits_fin, step):
    ''' Mide el tiempo medio de multiplicar varias parejas de numeros con la el
    algoritmo de multiplicacion estandar

    ::n_pairs: numero de parejas a generar
    :type n_pairs: int
    :num_digits_ini: digitos iniciales de la pareja
    :type num_digits_ini: int
    :num_digits_fin:digitos finale de la pareja
    :type num_digits_fin: int
    :step: incremento en el numero de digitos
    :type step: int

    :return: lista con los tiempos
    :return type: list
    '''
    l=[]
    while num_digits_ini <= num_digits_fin:
        t=0
        for n in range(n_pairs):
            num1 = rand_numero(num_digits_ini)
            num2 = rand_numero(num_digits_ini)
            t_ini= time.time()
            mult_numeros(num1,num2)
            t_fin = time.time()
            t+=t_fin-t_ini
        l.append(t/n_pairs)
        num_digits_ini+=step
    return l

def time_mult_numeros_fft(n_pairs, num_digits_ini, num_digits_fin, step, fft_func=fft):
    ''' Mide el tiempo medio de multiplicar varias parejas de numeros con la el
    algoritmo de la fft

    :n_pairs: numero de parejas a generar
    :type n_pairs: int
    :num_digits_ini: digitos iniciales de la pareja
    :type num_digits_ini: int
    :num_digits_fin:digitos finale de la pareja
    :type num_digits_fin: int
    :step: incremento en el numero de digitos
    :type step: int
    :fft_func: funcion para la fft
    :type fft_func: function

    :return: lista con los tiempos
    :return type: list
    '''
    l = []
    while num_digits_ini <= num_digits_fin:
        t=0
        for _ in range(n_pairs):
            num1 = rand_numero(num_digits_ini)
            num2 = rand_numero(num_digits_ini)
            t_ini= time.time()
            mult_numeros_fft(num1,num2, fft_func=fft_func)
            t_fin = time.time()
            t+=t_fin-t_ini
        l.append(t/n_pairs)
        num_digits_ini+=step
    return l
    #comprobar q va bien pork dio value error
