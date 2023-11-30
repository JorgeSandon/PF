import math
import scipy.integrate as spi
import numpy as np
import random
import pandas as pd

global Sn
global Max_HorasUsoMaq
global Prob


"""def Crear_Demanda_Aleatoria():
    i=0
    DemandaPorSemana=[]
    while i<=104:
        
        DemandaPorSemana.append(random.randint(30, 60))
        i+=1
    return DemandaPorSemana"""


def cargar_demanda(file_path='pronostico.csv'):
    try:
        # Cargar el archivo CSV
        df = pd.read_csv(file_path, sep=';', parse_dates=['Fecha'])

        # Asegurarse de que la columna 'Fecha' sea el índice
        df.set_index('Fecha', inplace=True)

        # Obtener la columna 'Demanda'
        demanda = df['Demanda']

        # Convertir la columna 'Demanda' a una lista y almacenarla en Demanda_n
        Demanda_n = demanda.tolist()

        return Demanda_n
    except Exception as e:
        print(f"Error al cargar la demanda: {e}")
        return None

def Estados(CantidadEstados): #mientras tanto se deja uniforme partido entre el maximo y el minimo

    resultados_Sn=[]
    Min=0
    Max=Max_HorasUsoMaq
    rangos=round(((Max-Min)/CantidadEstados)+0.5)
    cuenta= 0
    
    for i in range(CantidadEstados):
        LimInf=cuenta
        LimSup=(cuenta+rangos-1)
        
        Intervalo=[LimInf,LimSup]
        cuenta+=rangos   
        resultados_Sn.append(Intervalo)
        
    return resultados_Sn
'''
def weibull_pdf(x, lambda_param, k_param):
    return (k_param / lambda_param) * (x / lambda_param)**(k_param - 1) * np.exp(-(x / lambda_param)**k_param)

def weibull_cdf(a=2, b=10, lambda_param=8, k_param=1.2):
    integrand = lambda x: weibull_pdf(x, lambda_param, k_param)
    result, _ = spi.quad(integrand, a, b)
    return result
'''
def linear_increasing_cdf(x, a, b): 
    if x < a:
        return 0
    elif a <= x <= b:
        return ((x - a)**2) / ((b - a)**2)
    else:
        return 1
def probability_of_failure(x1, x2, a=0, b=360):
    cdf_x1 = linear_increasing_cdf(x1, a, b)
    cdf_x2 = linear_increasing_cdf(x2, a, b)
    resultados_probability = cdf_x2 - cdf_x1
    return resultados_probability

def probabilidadesSn(listaSn):
    
    Probabilidades_Sn={}
    
    for rango in listaSn:
        probabilidad=probability_of_failure(rango[0]-0.5,rango[1]+0.5)
        Probabilidades_Sn[tuple(rango)]=probabilidad
        
    return Probabilidades_Sn

def probabilidad_en_n(Sn,D):

    
    suma_probabilidades=0
    
    for rango, valor in Prob.items():
        inicio, fin = rango
        
        if (inicio<=Sn<=fin):
            suma_probabilidades+=valor
        elif (inicio>Sn):
            if inicio>(Sn+D):
                break
            suma_probabilidades+=valor           
    
    return suma_probabilidades

def NextSn(Sn_prom):

   for sublista in Sn:
        if Sn_prom<=sublista[1] and Sn_prom>=sublista[0]:
            return tuple(sublista)
        elif Sn_prom>Max_HorasUsoMaq:
            return tuple(Sn[-1])
   

def EstadoActual(estado):
    i=1
    if estado>Max_HorasUsoMaq:
        estado=Max_HorasUsoMaq
    if estado==-1:# cuando ya el codigo este lindo y hermoso cambiar condicional a <0
        estado=0
    for x in Sn:
        if estado<=x[1] and estado>=x[0]:
            return i
            
        i+=1
    return None


########## Parametros ##########

#Uso de Demanda_n
Demanda_n = cargar_demanda()

### Inputs

Max_HorasUsoMaq = 350  #Horas maximas que se permite la maquina trabajar antes de Mant

N = len(Demanda_n)

SN = 0#Estado inicial

HPlaneadas_MantPrev = 300#Horas planeadas para mantenimiento preventivo

HDurac_MantPrev = 4#Horas que demora el mantenimiento preventivo

HDurac_MantCorr = 15#Horas que demora el mantenimiento correctivo

###### Costos

COportXHora_Penalizada = 1000*4200 #Costo de penalización por no atender el cliente por hora ($/h) 

COportXHora_HMant = 80000#Costo de oportunidad horas máquina no usadas hasta mantenimiento prev programado ($/h) 

C_MantPrev = 14000000#Costo de realizar mantenimiento preventivo ($) 

C_MantCorr = 40000000#Costo de realizar mantenimiento correctivo ($) 

C_UsoMaq = 150000#Costo de uso de maquina ($/h) 

Ganancia = -2000000#Ganancia por uso de maquina (-$/h) 

##################################################################################################

### No Inputs
CantidadEstados = 24 #Cantidad de estados que se van a discretizar

Sn = Estados(CantidadEstados) #Generar lista con los estados

Prob = probabilidadesSn(Sn) #Generar probabilidades para cada rango en cada estado

DecisionMant = [0,1]#Si hago o no hago mantenimiento, variable de decision

##################################################################################################


F = {}
X_opt ={}

for n in range(N,-1,-1):
    
    #Fin/Inicio PDE
    if n==N:
        for s in range(len(Sn)):
            F[n,tuple(Sn[s])]=0
    
    #Iteraciones en el resto de n
    else:
        for s in range(min(EstadoActual(n*60),len(Sn))):
            
            F[n,tuple(Sn[s])]=9999999999999999999999999
            
            Sn_prom= (Sn[s][1]+Sn[s][0])/2
            
            for x in range(len(DecisionMant)):
                
                if (Sn_prom+Demanda_n[n])*(1-x)>=Max_HorasUsoMaq: # si al no querer hacer mantenimiento se pasa de 350, se hace.
                    f1=99999999999999999999
                    
                else:
                    
                    f1=((1-probabilidad_en_n(Sn_prom, Demanda_n[n]))*(DecisionMant[x]*(COportXHora_HMant*(max(HPlaneadas_MantPrev-Sn_prom,0))+
                        (COportXHora_Penalizada-Ganancia)*HDurac_MantPrev+C_MantPrev)+(1-DecisionMant[x])*(Demanda_n[n])*(Ganancia+C_UsoMaq)+
                        F[n+1,NextSn((Sn_prom+Demanda_n[n])*(1-DecisionMant[x]))]))+(
                        probabilidad_en_n(Sn_prom, Demanda_n[n])*(((COportXHora_Penalizada-Ganancia)*(HDurac_MantCorr)+C_MantCorr)*(1-DecisionMant[x])+F[n+1,NextSn(0)]))
                    '''
                    f=probabilidad_en_n(Sn_prom, Demanda_n[n]) * (
                        DecisionMant[x] * (
                            COportXHora_HMant * (max(HPlaneadas_MantPrev - Sn_prom, 0) + COportXHora_Penalizada * HDurac_MantPrev + C_MantPrev) +
                            (1 - DecisionMant[n]) * (Demanda_n[n]) * (Ganancia + C_UsoMaq)
                        ) +
                        ((1 - probabilidad_en_n(Sn_prom, Demanda_n[n])) * ((COportXHora_Penalizada + Ganancia) * HDurac_MantCorr + C_MantCorr) * (1 - DecisionMant[n]))
                        )    
                    '''
                    
                if f1<F[n,tuple(Sn[s])]:
                    
                    F[n,tuple(Sn[s])]=f1
                    
                    X_opt[n,tuple(Sn[s])]=DecisionMant[x]
                    
data_list = [
    [key[0], key[1], X_opt[key], F[key]] for key in X_opt.keys()
]

# Crear el DataFrame directamente desde la lista de listas
DataF = pd.DataFrame(data_list, columns=["Etapa", "Rango Estado Actual", "Mejor Decision", "Valor esperado(en Mill)"])
print(DataF)

