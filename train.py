import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import pickle

# Đọc dữ liệu từ file wdbc.data và chỉ định các cột
column_names = ['ID', 'Diagnosis', 'Radius_Mean', 'Texture_Mean', 'Perimeter_Mean', 'Area_Mean',
                'Smoothness_Mean', 'Compactness_Mean', 'Concavity_Mean', 'ConcavePoints_Mean',
                'Symmetry_Mean', 'FractalDimension_Mean', 'Radius_SE', 'Texture_SE', 'Perimeter_SE',
                'Area_SE', 'Smoothness_SE', 'Compactness_SE', 'Concavity_SE', 'ConcavePoints_SE',
                'Symmetry_SE', 'FractalDimension_SE', 'Radius_Worst', 'Texture_Worst', 'Perimeter_Worst',
                'Area_Worst', 'Smoothness_Worst', 'Compactness_Worst', 'Concavity_Worst',
                'ConcavePoints_Worst', 'Symmetry_Worst', 'FractalDimension_Worst']

# Đọc dữ liệu và tạo DataFrame
data = pd.read_csv("content/wdbc.data", names=column_names)

# Hiển thị 5 dòng đầu tiên của DataFrame để kiểm tra
print(data.tail(10))
print(data.dtypes)

class PerceptronCustom:
    def __init__(self, learning_rate=0.05, n_iterations=601):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])

        train_accuracies = []
        valid_accuracies = []
        train_f1_scores = []
        valid_f1_scores = []

        for i in range(self.n_iterations):
            for xi, yi in zip(X, y):
                update = self.learning_rate * (yi - self.predict(xi))
                self.weights += update * xi

            if i % 50 == 0:
                train_pred = self.predict(X_train)
                valid_pred = self.predict(X_valid)
                train_accuracy = np.mean(train_pred == y_train)
                valid_accuracy = np.mean(valid_pred == y_valid)
                train_f1 = 2 * precision_score(y_train, train_pred) * recall_score(y_train, train_pred) / \
                            (precision_score(y_train, train_pred) + recall_score(y_train, train_pred))
                valid_f1 = 2 * precision_score(y_valid, valid_pred) * recall_score(y_valid, valid_pred) / \
                            (precision_score(y_valid, valid_pred) + recall_score(y_valid, valid_pred))
                train_accuracies.append(train_accuracy)
                valid_accuracies.append(valid_accuracy)
                train_f1_scores.append(train_f1)
                valid_f1_scores.append(valid_f1)

        # Vẽ biểu đồ cột
        epochs = range(0, self.n_iterations + 1, 50)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
        plt.plot(epochs, valid_accuracies, label='Validation Accuracy', marker='o')
        plt.plot(epochs, train_f1_scores, label='Train F1 Score', marker='o')
        plt.plot(epochs, valid_f1_scores, label='Validation F1 Score', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Scores')
        plt.title('Accuracy and F1 Score Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict(self, xi):
        linear_output = np.dot(xi, self.weights)
        return np.where(linear_output >= 0, 1, 0)

# Tách tập train, valid và test sử dụng stratified sampling
X = data.loc[:, 'Radius_Mean':'FractalDimension_Worst'].values
X = np.c_[X, np.ones(X.shape[0])]  # Thêm cột bias vào X
# Xác định nhãn (labels) là cột 'Diagnosis'
y = data['Diagnosis']
y = np.where(y == 'M', 1, 0)
# Chia thành tập train, tập validation và tập test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=25)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.333, stratify=y_temp, random_state=25)

# Khởi tạo mô hình Perceptron và huấn luyện trên tập train
perceptron_model = PerceptronCustom()
perceptron_model.fit(X_train, y_train)

# Đánh giá mô hình trên tập train
train_pred = perceptron_model.predict(X_train)
train_accuracy = np.mean(train_pred == y_train)
print("Độ chính xác trên tập train:", train_accuracy)

# Đánh giá mô hình trên tập valid
valid_pred = perceptron_model.predict(X_valid)
valid_accuracy = np.mean(valid_pred == y_valid)
print("Độ chính xác trên tập valid:", valid_accuracy)

# Đánh giá trên tập train
train_precision = precision_score(y_train, train_pred)
train_recall = recall_score(y_train, train_pred)
train_specificity = recall_score(y_train, train_pred, pos_label=0)
train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0

print("Precision trên tập train:", train_precision)
print("Recall trên tập train:", train_recall)
print("Specificity trên tập train:", train_specificity)
print("F1 trên tập train:", train_f1)

# Đánh giá trên tập valid
valid_pred = perceptron_model.predict(X_valid)
valid_precision = precision_score(y_valid, valid_pred)
valid_recall = recall_score(y_valid, valid_pred)
valid_specificity = recall_score(y_valid, valid_pred, pos_label=0)
valid_f1 = 2 * (valid_precision * valid_recall) / (valid_precision + valid_recall) if (valid_precision + valid_recall) > 0 else 0

print("Precision trên tập valid:", valid_precision)
print("Recall trên tập valid:", valid_recall)
print("Specificity trên tập valid:", valid_specificity)
print("F1 trên tập valid:", valid_f1)

# Đánh giá trên tập test
test_pred = perceptron_model.predict(X_test)
test_accuracy = np.mean(test_pred == y_test)
test_precision = precision_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)
test_specificity = recall_score(y_test, test_pred, pos_label=0)
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0

print("\nKết quả trên tập test:")
print("Accuracy trên tập test:", test_accuracy)
print("Precision trên tập test:", test_precision)
print("Recall trên tập test:", test_recall)
print("Specificity trên tập test:", test_specificity)
print("F1 trên tập test:", test_f1)

# Lưu model
with open('perceptron_model.pkl', 'wb') as f:
    pickle.dump(perceptron_model, f)
print("Model đã được lưu vào file 'perceptron_model.pkl'")