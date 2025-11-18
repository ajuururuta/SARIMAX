# -*- coding: utf-8 -*-

# ===============================================
# SARIMAX 自动参数优化 + 可视化 + 终端进度条示例
# ===============================================
# 改动说明：
# 1. 使用 tqdm 终端进度条 (替换 tqdm_notebook)。
# 2. 自动根据 ADF 检验选择 d。
# 3. 支持候选季节周期，自动搜索最优 (AIC 最小)。
# 4. 自动挑选 AIC 最小的模型并输出详细 summary。
# 5. 增加：原始序列图、差分图、ACF/PACF、季节分解、残差诊断、拟合 vs 实际、预测图。
# 6. 数据质量检查：缺失、重复索引、频率设置。
# 7. 新增：CPU 并行网格搜索 (joblib)，通过环境变量 SARIMAX_N_JOBS 控制并发数，默认使用全部核心。
# ===============================================

import warnings
warnings.filterwarnings('ignore')

# 基础库
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 统计 / 时序库
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# 评估指标
from sklearn.metrics import mean_squared_error

# 进度条 (终端)
from tqdm import tqdm

# 并行库
from joblib import Parallel, delayed

# 控制并行度：优先读环境变量
def _get_n_jobs():
    try:
        return int(os.environ.get('SARIMAX_N_JOBS', '-1'))  # -1 表示使用全部核心
    except Exception:
        return -1

# -----------------------------------------------
# 读取数据并检查是否适用于 SARIMAX
# -----------------------------------------------
# data.csv 已包含 timestamp(日期时间) 与 VALUE(数值)，为逐分钟数据。
# SARIMAX 需要：时间索引为 DateTimeIndex、频率稳定、无大量缺失。
# 若存在缺失将插值补齐；若存在重复将聚合为均值。

raw = pd.read_csv('data.csv')
# 兼容用户可能的不同列命名情况
if 'timestamp' in raw.columns:
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    raw = raw.set_index('timestamp')
else:
    # 如果第一列就是索引列
    raw.iloc[:, 0] = pd.to_datetime(raw.iloc[:, 0])
    raw = raw.set_index(raw.columns[0])

# 排序确保单调
raw = raw.sort_index()

# 设置频率为分钟
raw = raw.asfreq('T')  # 等价于 'min'

# 检查缺失
missing_count = raw['VALUE'].isna().sum()
if missing_count > 0:
    print(f'发现缺失值 {missing_count} 个，使用线性插值修复。')
    raw['VALUE'] = raw['VALUE'].interpolate()

# 检查重复索引
if raw.index.duplicated().any():
    print('发现重复时间戳，按均值聚合。')
    raw = raw.groupby(raw.index).mean()
    raw = raw.asfreq('T')

# 简单数据合理性检查
print('数据起止时间:', raw.index.min(), '->', raw.index.max())
print('总样本数:', len(raw))
print('是否存在缺失:', raw['VALUE'].isna().any())

# 若数据过短会影响季节性周期选择，这里给出长度提示
if len(raw) < 500:
    print('样本少于 500，建议谨慎选择较大的季节长度。')

# 赋值给 data 变量以保持后续兼容
data = raw.copy()

# -----------------------------------------------
# 平稳性检验 (ADF) & 差分
# -----------------------------------------------
adf_stat, adf_p, _, _, critical_values, _ = adfuller(data['VALUE'])
print(f'原始序列 ADF Statistic: {adf_stat:.4f}, p-value: {adf_p:.4f}')
if adf_p < 0.05:
    d = 0
    print('原始序列已平稳，d=0')
else:
    d = 1
    print('原始序列不平稳，采用一次差分 d=1')
    data['VALUE_DIFF'] = data['VALUE'].diff()

# 再次对差分序列做 ADF (若进行了差分)
if d == 1:
    diff_adf_stat, diff_adf_p, *_ = adfuller(data['VALUE'].dropna().diff().dropna())
    print(f'差分后 ADF Statistic: {diff_adf_stat:.4f}, p-value: {diff_adf_p:.4f}')

# -----------------------------------------------
# 参数搜索空间设置
# -----------------------------------------------
# 非季节性 (p,d,q) 范围适度控制避免组合爆炸
p_range = range(0, 3)
q_range = range(0, 3)
# 季节性 (P,D,Q,s) - 这里给出候选季节长度
# 对逐分钟数据，常见季节长度：
#   60  -> 一小时循环
#   120 -> 两小时循环
# 可根据业务需要扩展，如 1440 (一天)，但需更长序列支持。
s_candidates = [60, 120] if len(data) >= 120 else [60]
P_range = range(0, 2)
Q_range = range(0, 2)
D_values = [0, 1]  # 同时尝试是否做季节性差分

print(f'季节候选周期: {s_candidates}')

# 构建所有参数组合
param_grid = []
for p in p_range:
    for q in q_range:
        for P in P_range:
            for Q in Q_range:
                for D_ in D_values:
                    for s_ in s_candidates:
                        param_grid.append((p, d, q, P, D_, Q, s_))

print(f'总参数组合数: {len(param_grid)}')

# -----------------------------------------------
# 并行优化函数：遍历所有组合，返回 AIC 排序结果
# -----------------------------------------------

def _fit_one(endog, order, seasonal_order, maxiter=200):
    try:
        model = SARIMAX(endog, order=order, seasonal_order=seasonal_order)
        fitted = model.fit(disp=False, maxiter=maxiter)
        return {
            'p': order[0], 'd': order[1], 'q': order[2],
            'P': seasonal_order[0], 'D': seasonal_order[1], 'Q': seasonal_order[2], 's': seasonal_order[3],
            'AIC': fitted.aic
        }
    except Exception:
        return None


def optimize_sarimax_parallel(endog, param_grid, n_jobs=-1, maxiter=200):
    # 任务列表
    tasks = [((p, d_, q), (P, D_, Q, s_)) for (p, d_, q, P, D_, Q, s_) in param_grid]
    print(f'使用 CPU 并行搜索，n_jobs={n_jobs} (可通过环境变量 SARIMAX_N_JOBS 覆盖)')
    # 注意：为避免与 BLAS 多线程争抢，建议运行时设置 \n'
    #       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \n'
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_fit_one)(endog, order, seasonal, maxiter) for order, seasonal in tasks
    )
    results = [r for r in results if r is not None]
    if not results:
        return pd.DataFrame(columns=['p','d','q','P','D','Q','s','AIC'])
    return pd.DataFrame(results).sort_values('AIC').reset_index(drop=True)

# 划分训练 / 测试集 (最后 N 条作为测试)
TEST_STEPS = 100
train = data.iloc[:-TEST_STEPS]  # 留出最后 100 点做预测评估

print('开始参数搜索...')
N_JOBS = _get_n_jobs()
best_df = optimize_sarimax_parallel(train['VALUE'], param_grid, n_jobs=N_JOBS)
print('搜索完成，前 5 个最优结果:')
print(best_df.head())

if best_df.empty:
    raise RuntimeError('参数搜索失败，没有可用的模型。')

# 选取 AIC 最小的参数
best_params = best_df.iloc[0]
print('最优参数:')
print(best_params)

final_order = (int(best_params.p), int(best_params.d), int(best_params.q))
final_seasonal = (int(best_params.P), int(best_params.D), int(best_params.Q), int(best_params.s))
print(f'最终使用 order={final_order}, seasonal_order={final_seasonal}')

# -----------------------------------------------
# 拟合最终模型
# -----------------------------------------------
final_model = SARIMAX(data['VALUE'], order=final_order, seasonal_order=final_seasonal)
final_result = final_model.fit(disp=False, maxiter=500)
print(final_result.summary())

# -----------------------------------------------
# 预测评估 (使用留出的测试集长度)
# -----------------------------------------------
forecast = final_result.get_forecast(steps=TEST_STEPS)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int(alpha=0.05)
actual_test = data['VALUE'].iloc[-TEST_STEPS:]

mse = mean_squared_error(actual_test, forecast_mean)
print(f'测试集 {TEST_STEPS} 步预测 MSE: {mse:.4f}')

# -----------------------------------------------
# 可视化
# -----------------------------------------------
plt.figure(figsize=(12,4))
plt.plot(data['VALUE'], label='原始序列')
plt.title('原始时间序列')
plt.legend()
plt.tight_layout()

if d == 1:
    plt.figure(figsize=(12,4))
    plt.plot(data['VALUE'].diff(), label='一次差分')
    plt.title('差分后序列 (d=1)')
    plt.legend()
    plt.tight_layout()

# ACF / PACF
fig, ax = plt.subplots(1,2, figsize=(14,4))
plot_acf(data['VALUE'].diff().dropna() if d==1 else data['VALUE'], ax=ax[0], lags=60)
ax[0].set_title('ACF')
plot_pacf(data['VALUE'].diff().dropna() if d==1 else data['VALUE'], ax=ax[1], lags=60, method='ywm')
ax[1].set_title('PACF')
plt.tight_layout()

# 季节分解 (选取最优季节周期)
try:
    decomp = seasonal_decompose(data['VALUE'], period=final_seasonal[3], model='additive', extrapolate_trend='freq')
    decomp.plot()
    plt.suptitle('季节分解 (period={})'.format(final_seasonal[3]))
    plt.tight_layout()
except Exception as e:
    print('季节分解失败:', e)

# 拟合 vs 实际
plt.figure(figsize=(12,4))
plt.plot(data['VALUE'], label='实际')
plt.plot(final_result.fittedvalues, label='拟合', alpha=0.7)
plt.title('拟合值 vs 实际值')
plt.legend()
plt.tight_layout()

# 预测图 (包含置信区间)
plt.figure(figsize=(12,4))
plt.plot(data.index, data['VALUE'], label='历史')
future_index = pd.date_range(start=data.index[-1] + pd.Timedelta(minutes=1), periods=TEST_STEPS, freq='T')
plt.plot(future_index, forecast_mean, label='预测')
plt.fill_between(future_index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1], color='lightgray', alpha=0.5, label='95% CI')
plt.title(f'{TEST_STEPS} 步预测')
plt.legend()
plt.tight_layout()

# 残差诊断
try:
    final_result.plot_diagnostics(figsize=(12,8))
    plt.suptitle('残差诊断图')
    plt.tight_layout()
except Exception as e:
    print('残差诊断绘图失败:', e)

plt.show()

# -----------------------------------------------
# 小结 (打印)
# -----------------------------------------------
print('===== 模型选择与结果小结 =====')
print(f'最优非季节参数 (p,d,q): {final_order}')
print(f'最优季节参数 (P,D,Q,s): {final_seasonal}')
print(f'最优模型 AIC: {best_df.iloc[0].AIC:.2f}')
print(f'测试集 MSE: {mse:.4f}')
print('数据是否适合 SARIMAX: 已满足时间索引、频率稳定、缺失处理。')
print('如需进一步提升，可尝试加入外生变量 (exog) 或扩大参数搜索空间。')
