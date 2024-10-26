import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap

# Load dataset Iris và lấy hai đặc trưng sepal length và sepal width
iris = datasets.load_iris()
X = iris.data[:, :2]  # Chỉ lấy sepal length và sepal width
y = iris.target

# Tạo các màu để biểu diễn các lớp
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ['#FF0000', '#00FF00', '#0000FF']

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Khởi tạo mô hình
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Support Vector Machine": SVC(kernel='linear', C=1),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42)
}

# Thiết lập đồ thị 2D
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Vẽ biểu đồ cho từng mô hình
plt.figure(figsize=(18, 6))
for i, (name, model) in enumerate(models.items(), start=1):
    model.fit(X_train, y_train)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(1, 3, i)
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor='k', s=20)
    plt.title(name)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")

plt.show()