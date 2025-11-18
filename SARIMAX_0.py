# 主要库的导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from itertools import product
from tqdm import tqdm_notebook






# 数据加载与检查
# 示例：加载时间序列数据
data = pd.read_csv('data.csv', index_col=0, parse_dates=True)
# 设置时间索引的频率为分钟
data = data.asfreq('min')  # 'T' 表示分钟频率
data.head()







# 检验平稳性
result = adfuller(data['VALUE'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
# 如果数据不平稳，则通过差分使其平稳
data_diff = data['VALUE'].diff().dropna()
adf_diff_result = adfuller(data_diff)
print(f'Differenced ADF Statistic: {adf_diff_result[0]}')
print(f'Differenced p-value: {adf_diff_result[1]}')







# 设置超参数范围
# p: 非季节性自回归项数的范围
p = range(0, 3)
# d: 非季节性差分次数，通常通过平稳性检验确定
# 在此处设置为 1，表示进行一次差分
d = 1
# q: 非季节性移动平均项数的范围
q = range(0, 3)
# P: 季节性自回归项数的范围
P = range(0, 3)
# D: 季节性差分次数，通常设置为 1 表示存在季节性差分
D = 1
# Q: 季节性移动平均项数的范围
Q = range(0, 3)
# s: 季节性周期，表示数据的季节性周期长度
# 例如，季度数据 s=4，月度数据 s=12
s = 4
# 使用 itertools.product 构造所有可能的参数组合
parameters = product(p, q, P, Q)
parameters_list = list(parameters)
# 打印所有参数组合的总数
print(f'Total parameter combinations: {len(parameters_list)}')






def optimize_SARIMAX(endog, exog, order_list, d, D, s):
    """
    优化 SARIMAX 模型的参数以最小化 AIC。

    参数：
    - endog: 时间序列的目标变量。
    - exog: 外生变量（可选）。
    - order_list: 参数组合列表，包含 (p, q, P, Q)。
    - d: 非季节性差分次数。
    - D: 季节性差分次数。
    - s: 季节性周期。

    返回：
    - result_df: 包含参数组合及对应 AIC 的 DataFrame。
    """
    results = []

    for order in tqdm_notebook(order_list):
        try:
            # 构建 SARIMAX 模型
            model = SARIMAX(
                endog,
                exog,
                order=(order[0], d, order[1]),  # 设置非季节性参数
                seasonal_order=(order[2], D, order[3], s)  # 设置季节性参数
            ).fit(disp=False)  # disp=False 禁止输出拟合过程的详细信息

            # 保存参数组合及对应的 AIC 值
            results.append((order, model.aic))
        except:
            # 忽略拟合失败的参数组合
            continue

    # 将结果转换为 DataFrame 并按 AIC 升序排序
    result_df = pd.DataFrame(results, columns=['params', 'AIC'])
    result_df = result_df.sort_values(by='AIC')

    return result_df

# 在训练集上调用优化函数
train = data[:-200]  # 使用前 200 条数据作为训练集
best_result = optimize_SARIMAX(train['VALUE'], None, parameters_list, d, D, s)
# 输出最优参数组合及对应的 AIC 值
print(best_result.head())









# 设置最佳参数
order = (2, 1, 2)  # 替换为最佳非季节性参数
seasonal_order = (1, 0, 0, 4)  # 替换为最佳季节性参数
# 拟合最佳模型，增加最大迭代次数
best_model = SARIMAX(data['VALUE'], order=order, seasonal_order=seasonal_order)
result = best_model.fit(maxiter=500, method='powell')  # 设置最大迭代次数为 500
print(result.summary())






from sklearn.metrics import mean_squared_error

# 测试集预测
forecast = result.get_forecast(steps=10).predicted_mean
# 计算均方误差 (MSE)
mse = mean_squared_error(data['VALUE'][-10:], forecast)
print(f'Mean Squared Error: {mse}')
