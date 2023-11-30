import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
from plotly import graph_objects as go
import math
import scipy.integrate as spi
import numpy as np
import random


global COportXHora_Penalizada
global COportXHora_HMant
global C_MantPrev
global C_MantCorr
global C_UsoMaq
global Ganancia
global Max_HorasUsoMaq
global Prob
global Sn

global L
global BI
global BC
global BS


global HDurac_MantPrev
global HDurac_MantCorr
global HPlaneadas_MantPrev


def main():
    

    def Hagase_La_Luz():


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

        N = len(Demanda_n)

        ###### Costos

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

                        if f1<F[n,tuple(Sn[s])]:
                            
                            F[n,tuple(Sn[s])]=f1
                            
                            X_opt[n,tuple(Sn[s])]=DecisionMant[x]
                            
        data_list = [
            [key[0], key[1], X_opt[key], F[key]] for key in X_opt.keys()
        ]

        # Crear el DataFrame directamente desde la lista de listas
        DataF = pd.DataFrame(data_list, columns=["Etapa", "Rango Estado Actual", "Mejor Decision", "Valor esperado(en Mill)"])
        
        return(DataF)
    
    def entrenar_modelo_arima(data):
        modelo_arima = ARIMA(data['Demanda'], order=(3, 1, 2))
        resultados = modelo_arima.fit()
        return resultados

    def calcular_pronostico(data, semanas_hacia_delante):
        modelo_arima = entrenar_modelo_arima(data)
        predicciones_futuras = modelo_arima.predict(start=len(data), end=len(data) + semanas_hacia_delante - 1,
                                                    typ='levels')

        # Añadir las predicciones al DataFrame original debajo de la columna de demanda
        fecha_inicio = data.index[-1] + pd.DateOffset(days=7)
        index_predicciones = pd.date_range(start=fecha_inicio, periods=semanas_hacia_delante,
                                           freq='W').date  # Solo la fecha sin la hora
        data = data.append(pd.DataFrame({'Demanda': predicciones_futuras}, index=index_predicciones))

        # Renombrar el índice a "Fecha"
        data.index.name = "Fecha"

        return data

    st.title("Modelo de programación dinámica estocástica")
    
    st.sidebar.title("Parametros")
    
    ######################### PARAMETROS######################
    COportXHora_Penalizada = st.sidebar.number_input("Costo de penalización por no atender al cliente por hora ($/h)",min_value=0, step=1000,value=1000*4200, format= '%d')
    COportXHora_HMant = st.sidebar.number_input("Costo de oportunidad horas máquina perdidas ($/h)",min_value=0, step=1000,value=80000)
    C_MantPrev = st.sidebar.number_input("Costo de realizar mantenimiento preventivo ($)",min_value=0, step=1000,value=14000000)
    C_MantCorr = st.sidebar.number_input("Costo de realizar mantenimiento correctivo ($)",min_value=0, step=1000,value=40000000)
    C_UsoMaq = st.sidebar.number_input("Costo de uso de maquina ($/h)", min_value=0,step=1000,value=150000)
    Ganancia = st.sidebar.number_input("Ganancia por uso de maquina (-$/h)", step=1000,value=2000000)
    Ganancia=Ganancia*-1
    
    L = st.sidebar.number_input("Parámetro Lambda de la distribución Weibull")
    BI = st.sidebar.number_input("Parámetro beta, fallas tempranas 1 parte de la curva de la bañera")
    BC = st.sidebar.number_input("Parámetro beta, fallas aleatorias 2 parte de la curva de la bañera")
    BS = st.sidebar.number_input("Parámetro beta, fallas por desgaste 3 parte de la curva de la bañera")

    HDurac_MantPrev = st.sidebar.number_input("Horas duracion mantenimiento preventivo", min_value=0, step=5,value=4)
    HDurac_MantCorr = st.sidebar.number_input("Horas duracion mantenimiento correctivo", min_value=0, step=10,value=15)
    
    Max_HorasUsoMaq = st.sidebar.number_input("Horas maximas de uso maquina antes de mantenimiento preventivo", min_value=0, step=50,value=350)
    HPlaneadas_MantPrev = st.sidebar.number_input("Horas-Uso para mantenimiento preventivo por manual", min_value=0, step=50,value=300)#Horas planeadas para mantenimiento preventivo 
    
    ##########################################################
    
    tab1,tab2=st.tabs(["Pronóstico","Politica de Mantenimiento"])

    with tab1:
        st.title("Pronóstico")
    
        uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
    
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, delimiter=";")
            data['Fecha'] = pd.to_datetime(data['Fecha']).dt.date  # Quitar la hora de la fecha
            data.set_index('Fecha', inplace=True)
            
            col1,col2 = st.columns([0.3,0.7])
            
            with col1 : semanas_hacia_delante = st.number_input("Número de semanas hacia delante para pronosticar", min_value=1, step=1, value=1)
    
            if st.button("Calcular Pronóstico"):
                data = calcular_pronostico(data, semanas_hacia_delante)
    
                # Crear gráfico con Plotly Express
                fig = px.line(data, x=data.index, y='Demanda', labels={'Demanda': 'Demanda'}, title='Demanda y Predicción ARIMA')
                fig.update_traces(line=dict(color='blue'), name='Demanda')
    
                # Destacar los datos más recientes en rojo
                fig.add_trace(
                    go.Scatter(
                        x=data.tail(semanas_hacia_delante).index,
                        y=data.tail(semanas_hacia_delante)['Demanda'],
                        mode='lines+markers',
                        marker=dict(color='red'),
                        name='Pronostico'
                    )
                )

                # Agregar el slider al eje x
                fig.update_xaxes(rangeslider_visible=True)

                # Mostrar el gráfico
                st.plotly_chart(fig, use_container_width=True)


                # Guardar el DataFrame con las predicciones en un archivo CSV
                data.tail(semanas_hacia_delante).to_csv('pronostico.csv', sep=';'
                                                        )
    
                # Mostrar el DataFrame con las predicciones añadidas
                st.write("Datos de predicciones:")
                st.write(data.tail(semanas_hacia_delante))
    
        st.image(
            "https://www.memecreator.org/static/images/memes/5384617.jpg",
            width=400,
        )


        with tab2:
            
            st.title("Y Zeus dijo, hagase pf ...")
            
            if st.sidebar.button("Atender el llamado de Zeus"):
                st.header("Y se hizo el pf : ")
                DataF2=Hagase_La_Luz()
                st.dataframe(DataF2)
                
            
if __name__ == "__main__":
    main()



