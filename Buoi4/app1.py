from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ========== 1. ĐỌC DỮ LIỆU ==========
# Lưu ý: trong đường dẫn Windows phải dùng \\ hoặc r"" để tránh lỗi escape
df = pd.read_csv(r"C:\Dich Vu Ket Noi\Buoi4\iris.csv")

# Xác định X và y
X = df.iloc[:, :-1]   # 4 cột đầu
y = df.iloc[:, -1]    # cột species

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Huấn luyện KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Tính accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# ========== 2. ROUTES ==========
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Lấy dữ liệu từ form
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            # Dự đoán
            pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = pred[0]
        except Exception as e:
            prediction = f"Lỗi: {e}"

    return render_template("index.html", prediction=prediction, accuracy=accuracy)


if __name__ == "__main__":
    app.run(debug=True)
