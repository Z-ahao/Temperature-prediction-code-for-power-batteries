import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from deap import base, creator, tools, algorithms
import random
import tensorflow as tf

if __name__ == "__main__":
    # 修改为Excel文件路径
    file_path = 'C:\\Users\\ZH\\Desktop\\四月zh\\date.xlsx'  # 修改为您的Excel文件路径
    
    try:
        # 读取Excel文件
        excel_data = pd.ExcelFile(file_path)
        print("可用工作表:", excel_data.sheet_names)
        
        # 假设数据在第一个工作表（可根据实际情况调整）
        data = pd.read_excel(file_path, sheet_name=excel_data.sheet_names[0])
        
        # 打印原始列名检查
        print("\n原始列名:", data.columns.tolist())
        
        # 提取充电数据 (第1-4列)
        charge_data = data.iloc[:, :4]
        charge_data.columns = ['电流_charge', '电压_charge', '时间_charge', '温度_charge']
        
        # 提取放电数据 (第5-8列，假设Excel中连续存储)
        discharge_data = data.iloc[:, 4:8]
        discharge_data.columns = ['电流_discharge', '电压_discharge', '时间_discharge', '温度_discharge']
        
        # 合并数据
        processed_data = pd.concat([charge_data, discharge_data], axis=1)
        
        # 删除包含非数值的行
        processed_data = processed_data.apply(pd.to_numeric, errors='coerce').dropna()
        
        print("\n处理后的数据预览:")
        print(processed_data.head())
        
    except Exception as e:
        print(f"读取文件时出错：{e}")
        exit(1)
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split


    def load_battery_data(file_path):
        """专业级电池数据加载函数"""
        try:
            # 方法一：精确读取（处理合并单元格）
            df = pd.read_excel(file_path, header=None, skiprows=1, usecols="A:H")
            df.columns = [
                '电流_charge', '电压_charge', '时间_charge', '温度_charge',
                '电流_discharge', '电压_discharge', '时间_discharge', '温度_discharge'
            ]

            # 处理特殊格式（如"0-00097"→-0.00097）
            def clean_current(val):
                if isinstance(val, str):
                    if '-' in val[1:]:  # 处理中间带负号的情况
                        return float(val[0] + val[2:]) * -1
                    return float(val.replace('charge', '').replace('discharge', ''))
                return val

            df['电流_discharge'] = df['电流_discharge'].apply(clean_current)

            # 删除包含文本的行（如第2行的"discharge"）
            df = df[~df.apply(lambda x: x.astype(str).str.contains('discharge')).any(axis=1)]

            # 最终清洗
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            return df

    # 数据质量检查
    def validate_data(df):
        """数据验证函数"""
        print("\n=== 数据质量验证 ===")
        print("1. 温度范围:")
        print(f"充电温度: {df['温度_charge'].min():.2f}~{df['温度_charge'].max():.2f}℃")
        print(f"放电温度: {df['温度_discharge'].min():.2f}~{df['温度_discharge'].max():.2f}℃")

        print("\n2. 数据统计:")
        print(df.describe())


    # 主程序
    if __name__ == "__main__":
        file_path = 'C:/Users/ZH/Desktop/四月zh/date.xlsx'

        # 1. 加载数据
        battery_data = load_battery_data(file_path)
        print("\n成功加载数据:")
        print(battery_data.head())

        # 2. 数据验证
        validate_data(battery_data)


        # 3. 数据预处理
        def prepare_data(df, mode='charge'):
            features = [f'电流_{mode}', f'电压_{mode}', f'时间_{mode}']
            target = f'温度_{mode}'

            # 标准化处理
            scaler = StandardScaler()
            X = scaler.fit_transform(df[features])
            y = df[target].values.reshape(-1, 1)

            return train_test_split(X, y, test_size=0.2, random_state=42), scaler


        # 充电数据处理
        (X_train_c, X_test_c, y_train_c, y_test_c), scaler_c = prepare_data(battery_data, 'charge')

        # 放电数据处理
        (X_train_d, X_test_d, y_train_d, y_test_d), scaler_d = prepare_data(battery_data, 'discharge')

        print("\n=== 预处理结果 ===")
        print(f"充电数据 - 训练集: {X_train_c.shape}, 测试集: {X_test_c.shape}")
        print(f"放电数据 - 训练集: {X_train_d.shape}, 测试集: {X_test_d.shape}")

    # 2. 神经网络模型构建
    def create_model(hidden_units=10, input_dim=3):
        model = Sequential([
            Dense(hidden_units, input_dim=input_dim, activation='relu'),
            Dense(hidden_units//2, activation='relu'),  # 添加第二隐藏层
            Dense(1, activation='linear')
        ])
        return model

    # 3. 遗传算法优化
    def genetic_optimization(X_train, y_train, process_name):
        print(f"\n正在优化{process_name}过程模型...")
        
        # 重置随机种子确保可重复性
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        # 创建遗传算法类型
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # 设置工具箱
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0, 1)
        
        # 优化三个参数：隐藏单元数、学习率和批量大小
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                         toolbox.attr_float, n=3)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # 评估函数
        def evaluate(individual):
            hidden_units = int(individual[0] * 90 + 10)  # 10-100
            learning_rate = max(0.0001, individual[1] * 0.1)  # 0.0001-0.1
            batch_size = int(individual[2] * 128 + 32)  # 32-160
            
            model = create_model(hidden_units)
            model.compile(optimizer=Adam(learning_rate=learning_rate),
                          loss='mean_squared_error',
                          metrics=['mae'])
            
            # 添加早停和模型检查点
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            try:
                history = model.fit(X_train, y_train, 
                                   validation_split=0.2,
                                   epochs=100, 
                                   batch_size=batch_size, 
                                   verbose=0,
                                   callbacks=callbacks)
                
                # 返回最小验证损失
                return min(history.history['val_loss']),
            except:
                return float('inf'),  # 返回无限大表示无效解

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=5)

        # 遗传算法参数
        population_size = 30
        num_generations = 50
        population = toolbox.population(n=population_size)

        # 统计和名人堂
        hof = tools.HallOfFame(3)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # 运行遗传算法
        population, logbook = algorithms.eaSimple(
            population, toolbox, cxpb=0.8, mutpb=0.3,
            ngen=num_generations, stats=stats, halloffame=hof, verbose=True
        )

        # 解码最优个体
        best_individual = hof[0]
        best_hidden = int(best_individual[0] * 90 + 10)
        best_lr = max(0.0001, best_individual[1] * 0.1)
        best_batch = int(best_individual[2] * 128 + 32)
        
        print(f"\n{process_name}过程最优参数:")
        print(f"- 隐藏层单元数: {best_hidden}")
        print(f"- 学习率: {best_lr:.6f}")
        print(f"- 批量大小: {best_batch}")
        
        return best_hidden, best_lr, best_batch

    # 4. 模型训练和评估函数
    def train_and_evaluate(X_train, y_train, X_test, y_test, 
                          hidden_units, learning_rate, batch_size, process_name):
        print(f"\n训练{process_name}最优模型...")
        
        model = create_model(hidden_units)
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='mean_squared_error',
                      metrics=['mae'])
        
        history = model.fit(X_train, y_train,
                           epochs=150,
                           batch_size=batch_size,
                           validation_split=0.2,
                           verbose=1,
                           callbacks=[
                               tf.keras.callbacks.EarlyStopping(patience=15),
                               tf.keras.callbacks.ModelCheckpoint(
                                   f'best_{process_name}_model.h5', 
                                   save_best_only=True)
                           ])
        
        # 评估测试集
        y_pred = model.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))
        
        print(f"\n{process_name}模型测试集性能:")
        print(f"- MSE: {mse:.4f}")
        print(f"- MAE: {mae:.4f}")
        
        return model, history

    # 优化和训练充电模型
    best_hidden_c, best_lr_c, best_batch_c = genetic_optimization(X_train_c, y_train_c, "充电")
    charge_model, _ = train_and_evaluate(X_train_c, y_train_c, X_test_c, y_test_c,
                                        best_hidden_c, best_lr_c, best_batch_c, "充电")

    # 优化和训练放电模型
    best_hidden_d, best_lr_d, best_batch_d = genetic_optimization(X_train_d, y_train_d, "放电")
    discharge_model, _ = train_and_evaluate(X_train_d, y_train_d, X_test_d, y_test_d,
                                           best_hidden_d, best_lr_d, best_batch_d, "放电")

    # 5. 标准BP模型对比
    def standard_model_comparison(X_train, y_train, X_test, y_test, process_name):
        print(f"\n训练标准BP模型({process_name})...")
        model = create_model(32)  # 默认32个隐藏单元
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='mean_squared_error',
                      metrics=['mae'])
        
        history = model.fit(X_train, y_train,
                           epochs=100,
                           batch_size=64,
                           validation_split=0.2,
                           verbose=0)
        
        y_pred = model.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))
        
        print(f"标准BP模型({process_name})性能:")
        print(f"- MSE: {mse:.4f}")
        print(f"- MAE: {mae:.4f}")
        
        return mse, mae

    # 对比充电模型
    std_mse_c, std_mae_c = standard_model_comparison(X_train_c, y_train_c, X_test_c, y_test_c, "充电")
    
    # 对比放电模型
    std_mse_d, std_mae_d = standard_model_comparison(X_train_d, y_train_d, X_test_d, y_test_d, "放电")