import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import skew, kurtosis, mode
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro
import seaborn as sns
from sklearn import metrics

# Загрузка датасета Iris
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
# проверка пропусков
# print(iris_df.isnull().sum())

# Расчет статистических параметров
statistics = pd.DataFrame(index=['Skewness', 'Mean', 'Median', 'Mode', 'Kurtosis'])

for feature in iris_df.columns:
    # Расчет статистических параметров
    skewness = skew(iris_df[feature])
    mean_value = np.mean(iris_df[feature])
    median_value = np.median(iris_df[feature])
    mode_value = mode(iris_df[feature])[0]
    kurtosis_value = kurtosis(iris_df[feature])

    # Заполнение DataFrame
    statistics[feature] = [skewness, mean_value, median_value, mode_value, kurtosis_value]

# Вывод статистических параметров в таблицу с использованием plot
table_statistics = statistics.T
table_statistics.columns = ['Skewness', 'Mean', 'Median', 'Mode', 'Kurtosis']

# Удаление столбцов из iris_df.describe()
iris_summary = iris_df.describe().T.drop(['count', '25%', '50%', '75%', 'mean', 'std', 'min', 'max'],
                                         axis=1)

# Вывод таблицы
table_with_statistics = pd.concat([iris_summary, table_statistics], axis=1)
print(table_with_statistics)

shapiro_results = pd.DataFrame(index=['W statistic', 'p-value'])
for feature in iris_df.columns:
    # Тест Шапиро-Уилка
    w_stat, p_value_shapiro = shapiro(iris_df[feature])
    shapiro_results[feature] = [w_stat, p_value_shapiro]

# Проверка на уровень значимости
alpha = 0.05
significant_features = shapiro_results.columns[shapiro_results.loc['p-value'] < alpha]
non_significant_features = shapiro_results.columns[shapiro_results.loc['p-value'] >= alpha]

print("\nПризнаки с значимым отклонением от нормальности:")
print(significant_features)
print("\nПризнаки с незначимым отклонением от нормальности:")
print(non_significant_features)
# Построение гистограмм
for feature in iris_df.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(iris_df[feature], kde=True)
    plt.title(f'Histogram for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# не норм - sepal length (cm), petal length (cm), petal width (cm)
# норм - sepal width (cm)
# выбрал следующие методы кластерного анализа -
# 1. k-средних
# 2. DBSCAN

# станартизируем данные
standard_scaler = StandardScaler()
iris_scaler = standard_scaler.fit_transform(iris_df)
iris_scaler_df = pd.DataFrame(iris_scaler, columns=iris_df.columns)


# для метода к-средних определим количесвто кластеров методом локтя
wcss = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)

    kmeans.fit(iris_scaler_df)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), wcss)
plt.title('Выбор количества кластеров методом локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS') # сумма квадратов внутрикластерных расстояний (WCSS)
plt.show()
# получили оптимально кол-во кластеров - 3

kmeans = KMeans(n_clusters=3, n_init='auto')
y_pred = kmeans.fit_predict(iris_scaler_df)
labels = kmeans.labels_

# Расстояния между кластерами (Silhouette Score)
silhouette_score = metrics.silhouette_score(iris_scaler_df, labels)
print(f"Silhouette Score: {silhouette_score}")

# Внутрикластерные расстояния (Inertia)
inertia = kmeans.inertia_
print(f"Inertia: {inertia}")

# Компактность кластеров (Davies-Bouldin Index)
davies_bouldin_score = metrics.davies_bouldin_score(iris_scaler_df, labels)
print(f"Davies-Bouldin Index: {davies_bouldin_score}")

# Центры кластеров
cluster_centers = kmeans.cluster_centers_
print(f"Cluster Centers:\n{cluster_centers}")

plt.figure(figsize=(10, 6))
plt.scatter(iris_scaler_df.iloc[:, 0], iris_scaler_df.iloc[:, 1], c=y_pred, cmap='Paired')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=150, c='red', marker='^', label='Centroids')
plt.legend(loc='upper right')
plt.show()

# Silhouette Score: 0.45663380641237455 от -1 до 1, 0,45 - разумная степень разделения кластеров
# Inertia: 140.0820210962167
# Davies-Bouldin Index: 0.8322767767014573 - разумное качество кластеризации
# Cluster Centers:
# [[-0.07723421 -0.93062132  0.32313817  0.23727821]
#  [-1.01457897  0.85326268 -1.30498732 -1.25489349]
#  [ 1.06889068  0.05759433  0.96893325  1.00231456]]

# DBSCAN
agg_cluster = AgglomerativeClustering(n_clusters=3)
labels = agg_cluster.fit_predict(iris_scaler_df)

# Silhouette Score
silhouette_score = metrics.silhouette_score(iris_scaler_df, labels)
print(f"Silhouette Score: {silhouette_score}")

# Davies-Bouldin Index
davies_bouldin_score = metrics.davies_bouldin_score(iris_scaler_df, labels)
print(f"Davies-Bouldin Index: {davies_bouldin_score}")

# Визуализация дендрограммы
cluster_centers = np.array([iris_scaler_df[labels == i].mean(axis=0) for i in range(3)])
plt.figure(figsize=(10, 6))
plt.scatter(iris_scaler_df.iloc[:, 0], iris_scaler_df.iloc[:, 1], c=labels, cmap='Paired')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=150, c='red', marker='^',
            label='Centroids')
plt.legend(loc='upper right')
plt.show()