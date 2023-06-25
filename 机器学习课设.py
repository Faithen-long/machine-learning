import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
data = pd.read_csv('WorldCupsSummary.csv')

# 特征选择
features = ['Year', 'HostCountry', 'QualifiedTeams', 'MatchesPlayed', 'Attendance', 'HostContinent']
target = 'WinnerContinent'

# 数据预处理
X = data[features]
y = data[target]

# 将类别特征转换为数字编码
X_encoded = pd.get_dummies(X)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 对齐特征列
X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

# 创建逻辑回归模型
model = LogisticRegression()

# 拟合模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率：", accuracy)

# 使用模型进行冠军预测
future_data = pd.DataFrame({
    'Year': [2022],
    'HostCountry': ['Qatar'],
    'QualifiedTeams': [32],
    'MatchesPlayed': [0],
    'Attendance': [0],
    'HostContinent': ['Asia']
})

# 将输入数据转换为数字编码
future_data_encoded = pd.get_dummies(future_data)

# 对齐特征列
future_data_encoded = future_data_encoded.reindex(columns=X_train.columns, fill_value=0)

# 进行冠军预测
predicted_champion = model.predict(future_data_encoded)
print("预测冠军：", predicted_champion)

# 可视化特征重要性
feature_importance = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': model.coef_[0]})
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
