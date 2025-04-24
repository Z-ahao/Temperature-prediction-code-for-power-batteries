import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from deap import base, creator, tools, algorithms
import random
import tensorflow as tf
import matplotlib.pyplot as plt  # 导入绘图库
plt.rc('font',family='DengXian')

# 1. 数据准备函数
def prepare_data(file_path):
    try:
        # 读取Excel文件
        data = pd.read_excel(file_path, header=None)  # 假设没有标题行

        # 提取充电数据 (前4列)
        charge_data = data.iloc[:, :4]
        charge_data.columns = ['current', 'voltage', 'time', 'temperature']

        # 提取放电数据 (后4列)
        discharge_data = data.iloc[:, 4:8]
        discharge_data.columns = ['current', 'voltage', 'time', 'temperature']

        return charge_data, discharge_data

    except Exception as e:
        print(f"数据加载错误: {e}")
        exit(1)


# 2. 数据预处理函数
def preprocess_data(data):
    # 分离特征和目标
    X = data[['current', 'voltage', 'time']].values
    y = data['temperature'].values.reshape(-1, 1)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler


# 3. 神经网络模型构建
def build_model(hidden_units=32):
    model = Sequential([
        Dense(hidden_units, input_dim=3, activation='relu'),
        Dense(hidden_units // 2, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


# 4. 遗传算法优化
def optimize_with_ga(X_train, y_train):
    # 遗传算法设置
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=2)  # 优化学习率和隐藏单元数
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        lr = max(0.0001, individual[0] * 0.01)  # 学习率范围: 0.0001-0.01
        units = int(individual[1] * 50 + 10)  # 隐藏单元范围: 10-60

        model = build_model(units)
        model.optimizer.lr.assign(lr)

        history = model.fit(X_train, y_train,
                            epochs=30,
                            batch_size=32,
                            validation_split=0.2,
                            verbose=0)

        return min(history.history['val_loss']),

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 运行遗传算法
    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)

    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=0.7, mutpb=0.2,
                                   ngen=15, stats=stats,
                                   halloffame=hof, verbose=True)

    return hof[0]


# 5. 主程序
if __name__ == "__main__":
    # 文件路径设置
    file_path = "C:/Users/ZH/Desktop/四月zh/battery_data.xlsx"  # 文件路径

    # 加载数据
    charge_data, discharge_data = prepare_data(file_path)

    # 处理充电数据
    print("\n处理充电数据...")
    X_train_c, X_test_c, y_train_c, y_test_c, _ = preprocess_data(charge_data)

    # 处理放电数据
    print("\n处理放电数据...")
    X_train_d, X_test_d, y_train_d, y_test_d, _ = preprocess_data(discharge_data)

    # 优化充电模型
    print("\n优化充电模型...")
    best_charge = optimize_with_ga(X_train_c, y_train_c)
    lr_c = max(0.0001, best_charge[0] * 0.01)
    units_c = int(best_charge[1] * 50 + 10)

    charge_model = build_model(units_c)
    charge_model.optimizer.lr.assign(lr_c)
    history_c = charge_model.fit(X_train_c, y_train_c, epochs=100, batch_size=32, verbose=1)

    # 优化放电模型
    print("\n优化放电模型...")
    best_discharge = optimize_with_ga(X_train_d, y_train_d)
    lr_d = max(0.0001, best_discharge[0] * 0.01)
    units_d = int(best_discharge[1] * 50 + 10)

    discharge_model = build_model(units_d)
    discharge_model.optimizer.lr.assign(lr_d)
    history_d = discharge_model.fit(X_train_d, y_train_d, epochs=100, batch_size=32, verbose=1)


    # 评估模型
    def evaluate_model(model, X_test, y_test, name):
        y_pred = model.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))
        print(f"\n{name}模型性能:")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

        # 绘制预测值与真实值对比图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'{name}模型预测值与真实值对比')
        plt.show()

    evaluate_model(charge_model, X_test_c, y_test_c, "充电")
    evaluate_model(discharge_model, X_test_d, y_test_d, "放电")

    # 绘制训练过程中的损失值变化图
    plt.figure(figsize=(10, 6))
    plt.plot(history_c.history['loss'], label='充电模型训练损失')
    plt.plot(history_c.history['val_loss'], label='充电模型验证损失')
    plt.plot(history_d.history['loss'], label='放电模型训练损失')
    plt.plot(history_d.history['val_loss'], label='放电模型验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.title('模型训练过程中的损失值变化')
    plt.legend()
    plt.show()

    # 保存模型
    charge_model.save("charge_model.h5")
    discharge_model.save("discharge_model.h5")
    print("\n模型已保存为 charge_model.h5 和 discharge_model.h5")