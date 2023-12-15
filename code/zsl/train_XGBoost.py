import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt

# 提取数字特征
def extract_features(data):
    features = []
    labels = []
    for item in data:
        user = item["user"]
        feature = [
            user["followers_count"],
            user["friends_count"],
            user["listed_count"],
            user["favourites_count"],
            user["statuses_count"],
            int(user["protected"]),
            int(user["geo_enabled"]),
            int(user["verified"]),
            int(user["contributors_enabled"]),
            int(user["is_translator"]),
            int(user["is_translation_enabled"]),
            int(user["profile_background_tile"]),
            int(user["profile_use_background_image"]),
            int(user["has_extended_profile"]),
            int(user["default_profile"]),
            int(user["default_profile_image"]),
            int(user["following"]),
            int(user["follow_request_sent"]),
            int(user["notifications"])
        ]
        features.append(feature)
        if "label" in item:
            labels.append(item["label"])
        else:
            labels.append(None)
    return features, labels

# 读取json文件
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 加载数据
train_data = load_data("dataset/train.json")
dev_data = load_data("dataset/dev.json")
test_data = load_data("dataset/test.json")

# 提取训练集和验证集的特征
X_train, y_train = extract_features(train_data)
X_dev, y_dev = extract_features(dev_data)
X_test, y_test = extract_features(test_data)  # y_test 中的标签应该全部为 None

# 将标签转换为二进制（0为bot，1为human）
y_train_binary = [0 if y == 'bot' else 1 for y in y_train]
y_dev_binary = [0 if y == 'bot' else 1 for y in y_dev]

# 训练XGBoost模型
dtrain = xgb.DMatrix(X_train, label=y_train_binary)
ddev = xgb.DMatrix(X_dev, label=y_dev_binary)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 4,  # 减少树的深度
    'min_child_weight': 6,  # 增加孩子节点中所需的最小权重
    'gamma': 0.5,  # 增加gamma值
    'subsample': 0.8,  # 使用80%的样本来训练每棵树
    'colsample_bytree': 0.8  # 使用80%的特征来训练每棵树
}

num_rounds = 100
watchlist = [(dtrain, 'train'), (ddev, 'eval')]
bst = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=10)
# 验证模型
y_dev_pred = bst.predict(ddev)
y_dev_pred_binary = [0 if prob < 0.5 else 1 for prob in y_dev_pred]
print("Accuracy on dev set:", accuracy_score(y_dev_binary, y_dev_pred_binary))
print(classification_report(y_dev_binary, y_dev_pred_binary))


# 计算使用macro和weighted方法的F1指标
#f1_macro = f1_score(y_dev_binary, y_dev_pred_binary, average='macro')
f1_weighted = f1_score(y_dev_binary, y_dev_pred_binary, average='weighted')
#print(f"Macro F1 Score on dev set: {f1_macro:.2f}")
print(f"Weighted F1 Score on dev set: {f1_weighted:.2f}")

# 使用模型进行预测
dtest = xgb.DMatrix(X_test)
y_test_pred_probs = bst.predict(dtest)
y_test_pred = ['bot' if prob < 0.5 else 'human' for prob in y_test_pred_probs]
print("Predictions on test set:", y_test_pred)

# Update the test_data with the predicted labels
for i, item in enumerate(test_data):
    item['label'] = y_test_pred[i]

# Write the updated test_data back to a JSON file
with open('dataset/test_with_predictions.json', 'w', encoding='utf-8') as file:
    json.dump(test_data, file, ensure_ascii=False, indent=4)

print("Predictions have been written to test_with_predictions.json")