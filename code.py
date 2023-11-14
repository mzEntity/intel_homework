import pandas as pd
import numpy as np
import time

# 使用Intel版本的sklearn
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression # 逻辑回归
from sklearn.tree import DecisionTreeClassifier # 决策树
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.svm import SVC # 支持向量机
from sklearn.ensemble import AdaBoostClassifier # AdaBoost

filePath = "./creditcard.csv"

data = pd.read_csv(filePath)

# X需要舍弃Time列以及目标列（Class）
X = data.drop(['Class', 'Time'], axis=1)
# 目标列
y = data['Class']

# 分割测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 开始过采样，过采样就是增加少数类的样本个数，此处欺诈样本太少，所以增加至和非欺诈数量相等（n_samples = non_fraud_data.shape[0]）
fraud_data = X_train[y_train == 1]
non_fraud_data = X_train[y_train == 0]

# 测试集也要去掉Amount
X_test = X_test.drop("Amount", axis=1).values

fraud_upsampled = resample(fraud_data, replace=True, n_samples=non_fraud_data.shape[0] // 75, random_state=42)

# 重新拼接训练集
X_upsampled = np.vstack((non_fraud_data, fraud_upsampled))
y_upsampled = np.hstack((np.zeros(len(non_fraud_data)), np.ones(len(fraud_upsampled))))

# X的最后一列是金额Amount，用作权重而不是训练的属性
amount = X_upsampled[:, -1]
X_train_upsampled = X_upsampled[:, :-1]

# 成本敏感分析，权重和金额成正比
sample_weights = np.log(amount + 1) + 1

# 记录开始时间
start_time = time.time()

# 逻辑回归模型
# model = LogisticRegression()

# 决策树模型
# model = DecisionTreeClassifier()

# 随机森林模型
model = RandomForestClassifier(n_estimators=50, random_state=42)  # You can adjust hyperparameters

# AdaBoost模型
# base_model = DecisionTreeClassifier(max_depth=1)  # Weak learner, e.g., decision stump
# model = AdaBoostClassifier(base_model, n_estimators=10, random_state=42)

model.fit(X_train_upsampled, y_upsampled, sample_weight=sample_weights)


y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)

# 获得预测概率
y_scores = model.predict_proba(X_test)[:, 1]

# 计算AUPRC值
precision, recall, _ = precision_recall_curve(y_test, y_scores)
auprc = auc(recall, precision)

# 记录结束时间
end_time = time.time()

print("F1分数: ", f1)
print("auprc分数: ", auprc)
print(f"代码运行时间：{end_time - start_time:.2f} 秒")