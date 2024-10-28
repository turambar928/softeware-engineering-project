import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 读取CSV数据
data = pd.read_csv('D:\pythonProject20\Flight_Meals_Data.csv')

# 假设航班人数接近于总餐食份数，所以我们将总餐食份数作为独立变量
X = data['Total Meals'].values.reshape(-1, 1)  # 独立变量，总餐食份数
y = data['Unused Meals'].values  # 目标变量，未使用餐食份数

# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 构建神经网络模型
model = Sequential([
    Dense(64, input_dim=1, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # 输出层，预测未使用餐食份数
])



# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 定义预测函数，输入航班人数后，预测未使用的餐食份数
def predict_wasted_meals(flight_passengers):
    flight_passengers_scaled = scaler_X.transform(np.array([[flight_passengers]]))
    predicted_unused_meals_scaled = model.predict(flight_passengers_scaled)
    predicted_unused_meals = scaler_y.inverse_transform(predicted_unused_meals_scaled)[0][0]
    total_meals_needed = flight_passengers - predicted_unused_meals
    return predicted_unused_meals, total_meals_needed

passengers = 200
#需要修改
#todo 输入作为接口用于小程序调用


unused_meals, meals_needed = predict_wasted_meals(passengers)
print(f"Predicted Unused Meals: {unused_meals}")
print(f"Meals Needed: {meals_needed}")

