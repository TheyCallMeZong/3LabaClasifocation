import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

use_columns = ["Goal Scored", "Ball Possession %", "Attempts", "Corners",
                        "Free Kicks", "Saves", "Pass Accuracy %",
                       "Passes", "Fouls Committed", "Yellow Card", "Yellow & Red", "Red"]

fifa = pd.read_csv("/Users/semensavcenko/Downloads/FIFA 2018 Statistics.csv",
                            usecols=use_columns)
df = fifa.copy()
scaler = StandardScaler()

# Применение стандартизации к данным
standardized_data = scaler.fit_transform(df)
df = pd.DataFrame(standardized_data, columns=df.columns)

# make correlation matrix
correlation_matrix = df.corr()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# print(correlation_matrix)

# по матрице корреляций походу 4 фактора

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Матрица корреляции')
plt.show()

# let's determine the factors
# n_componets - желаемое количесвто факторов, насчитали 4 поэтому будет 4
pca = PCA(n_components=4)

# Производим PCA
pca.fit(df)

# Получаем факторные нагрузки
loadings = pca.components_.T
# Выводим факторные нагрузки в таблицу
loadings_df = pd.DataFrame(loadings, index=df.columns, columns=[f'Factor {i+1}' for i in range(pca.n_components)])
plt.figure(figsize=(10, 6))
sns.heatmap(loadings_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Factor Loadings')
plt.show()

# вращение
pipeline = make_pipeline(StandardScaler(), FactorAnalysis(n_components=4, rotation="varimax"))
X_rotated = pipeline.fit_transform(df)

factor_loadings = pipeline.named_steps['factoranalysis'].components_

loadings_df_1 = pd.DataFrame(factor_loadings.T, index=df.columns, columns=[f'Factor {i+1}' for i in range(4)])
plt.figure(figsize=(10, 6))
sns.heatmap(loadings_df_1, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('factor loadings after rotation')
plt.show()
# Получаем собственные значения (важный показатель в определении факторов)
# собственные значения указывают на количество важных факторов, способных объяснить дисперсию в данных
eigenvalues = pca.explained_variance_
# Определяем количество компонент с собственными значениями больше 1
num_components = sum(eigenvalues > 1)

print("Количество компонент по критерию Кайзера:", num_components) #4
# каменистая осыпь
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.title('Scree Plot')
plt.xlabel('Количество компонент')
plt.ylabel('Собственные значения')
plt.show()

# в выоде можно бахнуть что то по типу мы определили 4 фактора и оказлось 4 по критерию Кайзера и каменистой осыпи
# и вставить таблицу с 3 нагрузочными факторами и 4, емае