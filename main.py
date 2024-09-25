import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv('dataset.csv')

colunas_independentes_x = ["categoria","n_passageiros","cap_porta_malas","ar_condicionado","tipo_cambio"]
colunas_dependentes_y = ["aluguel_diario"]

dados_x = df[colunas_independentes_x]
dados_y = df[colunas_dependentes_y]

modelo = LinearRegression().fit(dados_x, dados_y)

num_categ_test = 1
num_passageiros_test = 4
num_cap_porta_malas_test = 6
num_ar_condicionado_test = 0
num_cambio_test = 1

valores_test = np.array([[num_categ_test, num_passageiros_test, num_cap_porta_malas_test, num_ar_condicionado_test, num_cambio_test]])

predicao = modelo.predict(valores_test)
print(predicao)