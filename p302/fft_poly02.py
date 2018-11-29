import numpy as np

################### I-A ###################
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
    t = t + [0 for _ in range(N-len(t))]

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

#print(np.rint((invert_fft(fft([1,2,1,0,4])))))


################### I-B ###################

def rand_polinomio(long=2**10, base=10):
    ''' Genera una lista de enteros que representa los coeficientes de un polinommio
    :long: longitud de la lista a devolver
    :type long: int
    :base: los coeficientes van de 0 a base-1; base in [2,10] inclusive
    :type base: int
    :return: coeficientes del polinomio
    :return type: lista de ints de python
    '''

    return [int(np.random.randint(0, base)) for i in range(long)]


#print( type(rand_polinomio()[0]))

def poli_2_num(l_pol, base=10):
    ''' Transforma un polinomio en un numero (en base base) evaluandolo los coeficientes del
    del polinomio expresado en la lista l_pol. Para ello se utiliza la la regla de Horner
    :l_pol: coeficientes del polinomio
    :type l_pol: lista
    :base: los coeficientes van de 0 a base-1; base in [2,10] inclusive
    :type base: int
    :return: numero resultante de evaluar
    :return type: int
    '''
    pass
    #[1,2,3] # 3*X**2+ 2*X+ 1    =34 (rvaluando en 3)

def rand_numero(num_digits, base=10):
    '''
    :num_digits:
    :type num_digits:
    :base:
    :type base:
    :return:
    :return type:
    '''
    pass

def num_2_poli(num, base=10):
    '''
    :num:
    :type num:
    :base:
    :type base:
    :return:
    :return type:
    '''
    pass

def mult_polinomios(l_pol_1, l_pol_2):
    '''
    :l_pol_1:
    :type l_pol_1:
    :base:
    :type l_pol_2:
    :return:
    :return type:
    '''
    pass

def mult_polinomios_fft(l_pol_1, l_pol_2, fft_func=fft):
    '''
    :l_pol_1:
    :type l_pol_1:
    :l_pol_2:
    :type l_pol_2:
    :fft_func:
    :type fft_func:
    :return:
    :return type:
    '''
    pass

def mult_numeros(num1, num2):
    '''
    :num1:
    :type num1:
    :num2:
    :type num2:
    :return:
    :return type:
    '''
    pass

def mult_numeros_fft(num1, num2, fft_func=fft):
    '''
    :num1:
    :type num1:
    :num2:
    :type num2:
    :fft_func:
    :type fft_func:
    :return:
    :return type:
    '''
    pass


################### I-C ###################
def time_mult_numeros(n_pairs, num_digits_ini, num_digits_fin, step):
    '''
    :n_pairs:
    :type n_pairs:
    :num_digits_ini:
    :type num_digits_ini:
    :num_digits_fin:
    :type num_digits_fin:
    :step:
    :type step:
    :return:
    :return type:
    '''
    pass

def time_mult_numeros_fft(n_pairs, num_digits_ini, num_digits_fin, step, fft_func=fft):
    '''
    :n_pairs:
    :type n_pairs:
    :num_digits_ini:
    :type num_digits_ini:
    :num_digits_fin:
    :type num_digits_fin:
    :step:
    :type step:
    :fft_func:
    :type fft_func:
    :return:
    :return type:
    '''
    pass

################### II ###################
