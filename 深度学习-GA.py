import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
plt.rc('font', family='DengXian')

# 初始化参数
POPULATION_SIZE = 20  # 种群大小
GENERATIONS = 15      # 迭代代数
CX_PROB = 0.7         # 交叉概率
MUT_PROB = 0.2        # 变异概率
GENE_LENGTH = 3       # 基因长度（对应电流、电压、时间三个特征）

# 定义适应度函数和个体
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化适应度
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)  # 基因值范围 [0, 1]
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=GENE_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 定义适应度函数
def evaluate(individual, X, y):
    """
    适应度函数：计算个体的适应度值。
    这里假设目标是优化某个目标函数，例如最小化温度预测误差。
    """
    # 将基因值映射到实际特征范围
    current = individual[0] * 10  # 假设电流范围 [0, 10]
    voltage = individual[1] * 5   # 假设电压范围 [0, 5]
    time = individual[2] * 100    # 假设时间范围 [0, 100]

    # 假设目标函数是温度预测误差（这里用简单的线性模型作为示例）
    predicted_temperature = 0.5 * current + 0.3 * voltage + 0.2 * time
    error = np.mean(np.abs(predicted_temperature - y))  # 计算平均绝对误差

    return (error,)  # 返回适应度值（误差越小越好）

# 运行遗传算法
def run_ga(X, y):
    toolbox.register("evaluate", evaluate, X=X, y=y)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 交叉操作
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)  # 变异操作
    toolbox.register("select", tools.selTournament, tournsize=3)  # 选择操作

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)  # 保存最优个体
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB,
        ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True
    )
    return pop, log, hof

# 可视化结果
def plot_results(log, name):
    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_avgs = log.select("avg")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax.plot(gen, fit_avgs, "r-", label="Average Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"{name} - Genetic Algorithm Performance")
    ax.legend()
    ax.grid(True)
    plt.show()

# 主程序
if __name__ == "__main__":
    # 文件路径
    file_path = "C:/Users/ZH/Desktop/四月zh/battery_data.xlsx"

    # 加载数据
    try:
        data = pd.read_excel(file_path, header=None)
        # 充电数据 (前4列)
        charge_data = data.iloc[:, :4]
        charge_data.columns = ['current', 'voltage', 'time', 'temperature']
        # 放电数据 (后4列)
        discharge_data = data.iloc[:, 4:8]
        discharge_data.columns = ['current', 'voltage', 'time', 'temperature']
    except Exception as e:
        print(f"数据加载错误: {e}")
        exit(1)

    # 处理充电数据
    print("\n处理充电数据...")
    X_c = charge_data[['current', 'voltage', 'time']].values
    y_c = charge_data['temperature'].values

    # 运行遗传算法优化充电模型
    pop_c, log_c, hof_c = run_ga(X_c, y_c)
    print(f"充电模型最优个体：{hof_c[0]}")
    print(f"充电模型最优适应度值：{hof_c[0].fitness.values[0]}")

    # 可视化充电模型结果
    plot_results(log_c, "充电模型")

    # 处理放电数据
    print("\n处理放电数据...")
    X_d = discharge_data[['current', 'voltage', 'time']].values
    y_d = discharge_data['temperature'].values

    # 运行遗传算法优化放电模型
    pop_d, log_d, hof_d = run_ga(X_d, y_d)
    print(f"放电模型最优个体：{hof_d[0]}")
    print(f"放电模型最优适应度值：{hof_d[0].fitness.values[0]}")

    # 可视化放电模型结果
    plot_results(log_d, "放电模型")
