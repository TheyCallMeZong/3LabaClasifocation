import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro
from sklearn.decomposition import PCA
from statsmodels.multivariate.factor import Factor

# данные по вину (не опять а снова). Данные являются результатом химического анализа вин 3 различных сортов
# происходящих из одного региона.
# предварительно разделили на 3 фактора
# Химический состав:

#Alcohol (Алкоголь)
#Malic_acid (Яблочная кислота)
#Ash (Зола)
#Magnesium (Магний)
#Total_phenols (Общие фенолы)
#Flavanoids (Флавоноиды)
#Nonflavanoid_phenols (Нефлаваноидные фенолы)
#Proanthocyanins (Проантоцианы)
#Цвет и вкус:

#Color_intensity (Интенсивность цвета)
#Hue (Оттенок)
#OD280/OD315_of_diluted_wines (Оптическая плотность разбавленного вина)
#Производственные и химические параметры:
#Proline (Пролин)

# Class - target
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
# признаки
columns = ["Class", "Alcohol", "Malic_acid", "Ash", "Magnesium", "Total_phenols", "Flavanoids",
           "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280/OD315_of_diluted_wines", "Proline"]
wine_data = pd.read_csv(url, names=columns)
x_wine_data = wine_data.drop('Class', axis=1)
y_wine_data = wine_data['Class'].values


# стандартизация данный
scaler = StandardScaler()
x_wine_data_standardized = scaler.fit_transform(x_wine_data)
x_wine_data_df = pd.DataFrame(x_wine_data_standardized, columns=x_wine_data.columns)

# проверка на нормальность
normality_tests = {}
for column in x_wine_data_df.columns:
    stat, p_value = shapiro(x_wine_data_df[column])
    normality_tests[column] = p_value

# Вывод уровней значимости
alpha = 0.05
print("Тест на нормальность распределения:")
for column, p_value in normality_tests.items():
    if p_value > alpha:
        print(f"{column}: не подходит на нормальное распределние")
    else:
        print(f"{column}: подходит на нормальное распределние")
# Корреляция
# ура все подходит под нормальное распределение по тесту шапиро уилка (или как там его)
# значит делаем корреляцию пирсона
correl = x_wine_data_df.corr()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#print(correl)
#print(cov_matrix)

# PCA
pca = PCA(n_components=3)
principalComponents = pca.fit(x_wine_data_df)
print(pca.components_)
