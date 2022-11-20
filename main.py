import numpy
import matplotlib.pyplot as plt
from datetime import datetime as dt
from pandas_datareader import data as get_data
from numpy import linalg as LA

lista_de_acoes = ['PETR4','ABEV3','USIM5','BVSP', 'VALE3']
lista_de_acoes = [acao + '.SA' for acao in lista_de_acoes]

data_inicial = dt(2000,2,1)
data_final = dt.now()

precos = get_data.get_data_yahoo(lista_de_acoes, data_inicial, data_final)

retornos = precos.pct_change().dropna()
matriz_covatiancia = retornos.cov()
pesos_carteira = numpy.full(len(lista_de_acoes), 1/len(lista_de_acoes))
numero_acoes = len(lista_de_acoes)

numero_de_simulacoes = 10000
dias_projetados = 252 * 3
capital_inicial = 1000

retorno_medio = retornos.mean(axis = 0).to_numpy()
n = (dias_projetados, numero_acoes) 
matriz_retorno_medio = retorno_medio * numpy.ones(n)

# retorno_medio

L = LA.cholesky(matriz_covatiancia)

retornos_carteira = numpy.zeros([dias_projetados, numero_de_simulacoes])
montante_final = numpy.zeros(numero_de_simulacoes)

for s in range(numero_de_simulacoes):
    Rpdf = numpy.random.normal(size=(dias_projetados, numero_acoes))
    
    retornos_sinteticos = matriz_retorno_medio + numpy.inner(Rpdf, L)
    retornos_sinteticos = matriz_retorno_medio + numpy.inner(Rpdf, 1)

    retornos_carteira[:, s] = numpy.cumprod(numpy.inner(pesos_carteira, retornos_sinteticos) + 1) * capital_inicial

    montante_final[s] = retornos_carteira[-1,s]


plt.plot(retornos_carteira, linewidth=1)
plt.ylabel('Dinheiro')
plt.xlabel('Dias')
plt.show()

montante_99 = str(numpy.percentile(montante_final, 1))
montante_95 = str(numpy.percentile(montante_final, 5))
montante_mediano = str(numpy.percentile(montante_final, 50))

cenarios_com_lucro = str((len(montante_final[montante_final > 1000])/len(montante_final)) * 100) + "%"

print(f'''Ao investir R$ 1000,00 na carteira {lista_de_acoes}, podemos esperar esses resultados para os proximos 3 anos, utilizando o método de Monte Carlo com 10 mil simulações: Com 50% de probabilidade, o montante será maior que R$ {montante_mediano}. Com 95% de probabilidade, o montante será maior que R$ {montante_95}. Com 99% de probabilidade, o montante será maior que R$ {montante_99}. Em {cenarios_com_lucro} dos cenarios, foi possivel obter lucro nos proximos 3 anos.''')

config = dict(histtype = "stepfilled", alpha = 0.8, density = False, bins = 150)
fig,ax = plt.subplots()
ax.hist(montante_final, ** config)
ax.xaxis.set_major_formatter('R${x:.0f}')
plt.title('Distribuição montantes finais com simulação MC')
plt.xlabel('Montante final (R$)')
plt.ylabel("Frequencia")
plt.show()