import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# بارگیری داده از فایل اکسل
file_path = 'RawData21.xlsx'
df = pd.read_excel(file_path)

# تبدیل ستون تاریخ به datetime
df["Time"] = pd.to_datetime(df["Time"])

# نمایش چند ردیف از داده‌ها برای بررسی
print(df.head())

# نمایش آماری توصیفی از داده‌ها
print(df.describe())

# نمودار همبستگی بین ویژگی‌ها
plt.figure(figsize=(10, 6))
correlation_matrix = df.corr()
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title('Correlation Matrix')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.show()

# تفکیک ویژگی‌ها و هدف
X = df[["Open", "High", "Low", "Volume"]]
y = df["Target"]

# تقسیم داده‌ها به مجموعه آموزش و آزمایش
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# نرمال‌سازی داده‌ها
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ایجاد و آموزش مدل رگرسیون خطی
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# پیش‌بینی با استفاده از مدل
y_pred = model.predict(X_test_scaled)

# ارزیابی مدل
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# نمودار رگرسیون برای هر ویژگی
plt.figure(figsize=(15, 10))
for i, feature in enumerate(X.columns):
    plt.subplot(2, 2, i + 1)
    plt.scatter(X_test[feature], y_test, color='blue', label='Actual Data')
    plt.scatter(X_test[feature], y_pred, color='red', label='Predicted Data')
    plt.xlabel(feature)
    plt.ylabel('Target')
    plt.title(f'{feature} vs Target')
    plt.legend()
plt.tight_layout()
plt.show()

# نمایش ضرایب مدل
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# پیش‌بینی برای داده‌های جدید
new_data = {
    "Open": [126.00],  # مثال
    "High": [127.00],  # مثال
    "Low": [125.50],   # مثال
    "Volume": [8000000]  # مثال
}

new_df = pd.DataFrame(new_data)
new_df_scaled = scaler.transform(new_df)

# پیش‌بینی
predictions = model.predict(new_df_scaled)
print(predictions)
