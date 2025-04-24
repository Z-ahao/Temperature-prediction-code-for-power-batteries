import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
plt.rc('font',family='DengXian')

# 1. 数据准备函数
def prepare_data(file_path):
    """加载并分割充放电数据

    Args:
        file_path (str): Excel文件路径

    Returns:
        tuple: (充电数据DataFrame, 放电数据DataFrame)
    """
    try:
        # 读取Excel文件（无标题行）
        data = pd.read_excel(file_path, header=None)

        # 提取充电数据 (前4列)
        charge_data = data.iloc[:, :4].copy()
        charge_data.columns = ['current', 'voltage', 'time', 'temperature']

        # 提取放电数据 (后4列)
        discharge_data = data.iloc[:, 4:8].copy()
        discharge_data.columns = ['current', 'voltage', 'time', 'temperature']

        # 数据质量检查
        assert not charge_data.isnull().values.any(), "充电数据包含空值"
        assert not discharge_data.isnull().values.any(), "放电数据包含空值"

        return charge_data, discharge_data

    except Exception as e:
        print(f"数据加载错误: {e}")
        exit(1)


# 2. 数据预处理函数
def preprocess_data(data):
    """数据标准化和分割

    Args:
        data (DataFrame): 输入数据

    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # 分离特征和目标
    X = data[['current', 'voltage', 'time']].values
    y = data['temperature'].values.reshape(-1, 1)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled,
        test_size=0.2,
        random_state=42,
        shuffle=False  # 保持时间序列顺序
    )

    return X_train, X_test, y_train, y_test, scaler


# 3. 模型训练函数
def train_model(X_train, y_train):
    """训练弹性网络回归模型

    Args:
        X_train (ndarray): 训练特征
        y_train (ndarray): 训练目标

    Returns:
        ElasticNet: 训练好的模型
    """
    model = ElasticNet(
        alpha=0.1,  # 正则化强度
        l1_ratio=0.5,  # L1/L2混合比例
        max_iter=10000,  # 增加迭代次数确保收敛
        random_state=42,
        selection='random'  # 更快的收敛
    )
    model.fit(X_train, y_train.ravel())
    return model


# 4. 评估函数
def evaluate_model(model, X_test, y_test, scaler):
    """模型评估和可视化

    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试目标
        scaler: 标准化器
    """
    # 预测并反标准化
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test)

    # 计算指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\n模型性能:")
    print(f"- MAE: {mae:.4f} °C")
    print(f"- MSE: {mse:.4f} °C²")

    # 可视化
    plt.figure(figsize=(12, 5))

    # 真实值 vs 预测值
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('真实温度 (°C)')
    plt.ylabel('预测温度 (°C)')
    plt.title('预测结果对比')

    # 残差分析
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测温度 (°C)')
    plt.ylabel('残差')
    plt.title('残差分析')

    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 数据准备
    file_path = "C:/Users/ZH/Desktop/四月zh/battery_data.xlsx"  # 文件路径
    charge_data, discharge_data = prepare_data(file_path)

    print("充电数据统计:")
    print(charge_data.describe())
    print("\n放电数据统计:")
    print(discharge_data.describe())

    # 处理充电数据
    print("\n=== 充电数据建模 ===")
    X_train_c, X_test_c, y_train_c, y_test_c, scaler_c = preprocess_data(charge_data)
    model_c = train_model(X_train_c, y_train_c)
    evaluate_model(model_c, X_test_c, y_test_c, scaler_c)

    # 处理放电数据
    print("\n=== 放电数据建模 ===")
    X_train_d, X_test_d, y_train_d, y_test_d, scaler_d = preprocess_data(discharge_data)
    model_d = train_model(X_train_d, y_train_d)
    evaluate_model(model_d, X_test_d, y_test_d, scaler_d)

    # 保存模型
    joblib.dump(model_c, 'charge_model.pkl')
    joblib.dump(model_d, 'discharge_model.pkl')
    print("\n模型已保存为 charge_model.pkl 和 discharge_model.pkl")

