#задача классификации
import pandas as pd
from scipy.stats import shapiro
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

# 1 - выжил 0 - умер
use_columns = ["Sex", "Age", "Survived", "Pclass", "SibSp", "Parch"]
titanic_data = pd.read_csv("train.csv", usecols=use_columns)

# 50 на тест 840 на тренировку
msk = (titanic_data.index < len(titanic_data) - 50)
data_set_train = titanic_data[msk].copy()
data_set_test = titanic_data[~msk].copy()

# конвертим мужиков в 1 женщин в 0
data_set_train['Sex'] = data_set_train['Sex'].map({'male': 1, 'female': 0})
data_set_test['Sex'] = data_set_test['Sex'].map({'male': 1, 'female': 0})

# Заполнение пропущенных значений в возрасте медианным значением
data_set_train['Age'].fillna(data_set_train['Age'].median(), inplace=True)
data_set_test['Age'].fillna(data_set_test['Age'].median(), inplace=True)

data_surv = data_set_train['Survived']
data_sex = data_set_train['Sex']
data_age = data_set_train['Age']

# Проверяем нормальность с помощью теста Шапиро-Уилка
statistic, p_value = shapiro(data_surv)
statistic_sex, p_value_sex = shapiro(data_sex)
statistic_age, p_value_age = shapiro(data_age)

# Выводим результаты теста
print(f"Статистика теста для выживания: {statistic}, p-значение: {p_value}")
print(f"Статистика теста для возраста: {statistic_sex}, p-значение: {p_value_sex}")
print(f"Статистика теста для пола: {statistic_age}, p-значение: {p_value_age}")

# Проверяем уровень значимости
alpha = 0.05
if p_value > alpha:
    print("Выборка выживания похожа на нормальное распределение")
else:
    print("Выборка выживания не похожа на нормальное распределение")

if p_value_age > alpha:
    print("Выборка возраста похожа на нормальное распределение")
else:
    print("Выборка возраста не похожа на нормальное распределение")

# хотя это и так понятно но пусть будет
if p_value_sex > alpha:
    print("Выборка пола похожа на нормальное распределение")
else:
    print("Выборка пола не похожа на нормальное распределение")

# решим 2 методами, а именно - случайный лес и нейронки
# подготовка данных

#лесок
x_train = data_set_train.drop(['Survived'], axis=1)
x_test = data_set_test.drop(['Survived'], axis=1)
y_train = data_set_train['Survived']
y_test = data_set_test['Survived']
#`n_estimators`: Количество деревьев в лесу. Большее количество деревьев может повысить точность, но также увеличивает вычислительную сложность.
#`random_state`: Устанавливает случайное начальное состояние для воспроизводимости результатов. Если этот параметр не установлен, модель будет обучаться на разных данных при каждом запуске.
random_forest = RandomForestClassifier(n_estimators=100, random_state=10)
random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность: {accuracy:.2f}')
print(f'Полнота: {recall_score(y_test, y_pred):.2f}')


# логистическая регрессия
model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=100)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(f'Полнота: {recall_score(y_test, y_pred):.2f}')

