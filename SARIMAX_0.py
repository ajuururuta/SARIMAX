# -*- coding: utf-8 -*-

# ===============================================
# SARIMAX 自动参数优化 + 可视化 + 两阶段并行搜索 + 可调参数
# ===============================================
# 新增/增强点：
# - 参数可调（命令行参数优先，环境变量次之，最后使用默认值）：
#   --coarse-maxiter / COARSE_MAXITER (默认 50)
#   --fine-maxiter   / FINE_MAXITER   (默认 300)
#   --top-k          / TOP_K          (默认 10)
#   --small-gap      / SMALL_GAP      (默认 0.5)
#   --no-fine-expand（可选）关闭精调局部扩展
# - 两阶段搜索：粗筛（低 maxiter）→ 精调（高 maxiter），根据 small-gap 判定是否跳过精调。
# - 鲁棒拟合：多优化器回退链 + simple_differencing 回退，未收敛模型剔除。
# - 屏蔽起始参数告警 + 屏蔽 ConvergenceWarning（仅显示最终筛选后的结果）。
# ===============================================

import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_squared_error

from tqdm import tqdm
from joblib import Parallel, delayed

# 屏蔽收敛告警（在剔除未收敛模型逻辑下安全）
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# ---------------- 参数解析与默认 ----------------
def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

def _get_n_jobs():
    try:
        return int(os.environ.get('SARIMAX_N_JOBS', '-1'))
    except Exception:
        return -1

def _ensure_blas_single_thread():
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        if os.environ.get(var) is None:
            os.environ[var] = "1"

_ensure_blas_single_thread()

parser = argparse.ArgumentParser(description="SARIMAX 两阶段搜索（粗筛+精调）参数调节")
parser.add_argument("--coarse-maxiter", type=int, default=_env_int("COARSE_MAXITER", 50), help="粗筛最大迭代次数，默认 50")
parser.add_argument("--fine-maxiter",   type=int, default=_env_int("FINE_MAXITER", 300), help="精调最大迭代次数，默认 300")
parser.add_argument("--top-k",          type=int, default=_env_int("TOP_K", 10),        help="粗筛保留的前 K 个模型，默认 10")
parser.add_argument("--small-gap",      type=float, default=_env_float("SMALL_GAP", 0.5), help="AIC 差距阈值，小于则跳过精调，默认 0.5")
parser.add_argument("--no-fine-expand", action="store_true", help="禁用精调阶段的局部扩展网格")
args, _ = parser.parse_known_args()

COARSE_MAXITER = args.coarse_maxiter
FINE_MAXITER   = args.fine_maxiter
TOP_K          = args.top_k
SMALL_GAP      = args.small_gap
FINE_EXPAND    = not args.no_fine_expand

print(f"[参数] COARSE_MAXITER={COARSE_MAXITER}, FINE_MAXITER={FINE_MAXITER}, TOP_K={TOP_K}, SMALL_GAP={SMALL_GAP}, FINE_EXPAND={FINE_EXPAND}")

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']

# -----------------------------------------------
# 数据读取与预处理
# -----------------------------------------------
raw = pd.read_csv('data.csv')
if 'timestamp' in raw.columns:
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    raw = raw.set_index('timestamp')
else:
    raw.iloc[:, 0] = pd.to_datetime(raw.iloc[:, 0])
    raw = raw.set_index(raw.columns[0])

raw = raw.sort_index()
raw = raw.asfreq('T')

missing_count = raw['VALUE'].isna().sum()
if missing_count > 0:
    print(f'发现缺失值 {missing_count} 个，使用线性插值修复。')
    raw['VALUE'] = raw['VALUE'].interpolate()

if raw.index.duplicated().any():
    print('发现重复时间戳，按均值聚合。')
    raw = raw.groupby(raw.index).mean()
    raw = raw.asfreq('T')

print('数据起止时间:', raw.index.min(), '->', raw.index.max())
print('总样本数:', len(raw))
print('是否存在缺失:', raw['VALUE'].isna().any())
if len(raw) < 500:
    print('样本少于 500，建议谨慎选择较大的季节长度。')

data = raw.copy()

# -----------------------------------------------
# 平稳性检验
# -----------------------------------------------
adf_stat, adf_p, *_ = adfuller(data['VALUE'])
print(f'原始序列 ADF Statistic: {adf_stat:.4f}, p-value: {adf_p:.6g}')
if adf_p < 0.05:
    d = 0
    print('原始序列已平稳，d=0')
else:
    d = 1
    print('原始序列不平稳，采用一次差分 d=1')
    data['VALUE_DIFF'] = data['VALUE'].diff()

if d == 1:
    diff_adf_stat, diff_adf_p, *_ = adfuller(data['VALUE'].dropna().diff().dropna())
    print(f'差分后 ADF Statistic: {diff_adf_stat:.4f}, p-value: {diff_adf_p:.6g}')

# -----------------------------------------------
# 搜索空间
# -----------------------------------------------
p_range = range(0, 3)
q_range = range(0, 3)
s_candidates = [s for s in ([60, 120] if len(data) >= 120 else [60]) if len(data) >= 3*s]
if not s_candidates:
    s_candidates = [60]
print(f'季节候选周期: {s_candidates}')

P_range = range(0, 2)
Q_range = range(0, 2)
D_values = [0, 1]

param_grid = [(p, d, q, P, D_, Q, s_) for p in p_range for q in q_range
              for P in P_range for Q in Q_range for D_ in D_values for s_ in s_candidates]
print(f'总参数组合数: {len(param_grid)}')

# -----------------------------------------------
# 拟合相关函数
# -----------------------------------------------
def _fit_robust(model, maxiter=200, stage='coarse'):
    methods_coarse = ['lbfgs', 'powell']
    methods_fine = ['lbfgs', 'powell', 'nm', 'bfgs', 'cg']
    methods = methods_coarse if stage == 'coarse' else methods_fine
    for m in methods:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                warnings.filterwarnings('ignore', message='Non-stationary starting seasonal autoregressive', category=UserWarning)
                warnings.filterwarnings('ignore', message='Non-invertible starting seasonal moving average', category=UserWarning)
                warnings.filterwarnings('ignore', message='Non-stationary starting autoregressive parameters', category=UserWarning)
                # ConvergenceWarning 已在全局屏蔽
                res = model.fit(method=m, disp=False, maxiter=maxiter)
            if bool(res.mle_retvals.get('converged', True)) and np.isfinite(res.aic):
                return res
        except Exception:
            continue
    return None

def _build_model(endog, order, seasonal_order, simple_diff=False):
    return SARIMAX(
        endog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=True,
        simple_differencing=simple_diff
    )

def _fit_one_stage(endog, order, seasonal_order, maxiter=200, stage='coarse'):
    try:
        model = _build_model(endog, order, seasonal_order, simple_diff=False)
        res = _fit_robust(model, maxiter=maxiter, stage=stage)
        if res is None and (order[1] > 0 or seasonal_order[1] > 0):
            model_sd = _build_model(endog, order, seasonal_order, simple_diff=True)
            res = _fit_robust(model_sd, maxiter=maxiter, stage=stage)
        if res is None or not np.isfinite(res.aic):
            return None
        return {
            'p': order[0], 'd': order[1], 'q': order[2],
            'P': seasonal_order[0], 'D': seasonal_order[1], 'Q': seasonal_order[2], 's': seasonal_order[3],
            'AIC': float(res.aic)
        }
    except Exception:
        return None

def _clip(v, low, high):
    return max(low, min(high, v))

def _make_local_grid(base_rows, p_max, q_max, P_max, Q_max):
    local = set()
    for _, row in base_rows.iterrows():
        p0, d0, q0 = int(row.p), int(row.d), int(row.q)
        P0, D0, Q0, s0 = int(row.P), int(row.D), int(row.Q), int(row.s)
        for dp in [-1, 0, 1]:
            for dq in [-1, 0, 1]:
                for dP in [-1, 0, 1]:
                    for dQ in [-1, 0, 1]:
                        p_new = _clip(p0 + dp, 0, p_max)
                        q_new = _clip(q0 + dq, 0, q_max)
                        P_new = _clip(P0 + dP, 0, P_max)
                        Q_new = _clip(Q0 + dQ, 0, Q_max)
                        local.add((p_new, d0, q_new, P_new, D0, Q_new, s0))
    return list(local)

def parallel_search(endog, grid, n_jobs, maxiter, stage_desc, stage_key):
    tasks = [((p, d_, q), (P, D_, Q, s_)) for (p, d_, q, P, D_, Q, s_) in grid]
    print(f'{stage_desc}：任务数={len(tasks)}, n_jobs={n_jobs}, maxiter={maxiter}')
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_fit_one_stage)(endog, order, seasonal, maxiter, stage=stage_key)
        for order, seasonal in tqdm(tasks, desc=stage_desc, ncols=80)
    )
    results = [r for r in results if r is not None]
    if not results:
        return pd.DataFrame(columns=['p','d','q','P','D','Q','s','AIC'])
    return pd.DataFrame(results).sort_values('AIC').reset_index(drop=True)

# -----------------------------------------------
# 两阶段搜索
# -----------------------------------------------
TEST_STEPS = 100
train = data.iloc[:-TEST_STEPS]
N_JOBS = _get_n_jobs()

print('开始粗筛参数搜索...')
coarse_df = parallel_search(train['VALUE'], param_grid, N_JOBS, COARSE_MAXITER, '并行拟合(粗筛)', 'coarse')
print('粗筛完成，前 10 个结果:')
print(coarse_df.head(min(10, len(coarse_df))))
if coarse_df.empty:
    raise RuntimeError('粗筛阶段没有可用模型。')

need_fine = True
if len(coarse_df) >= TOP_K:
    gap = coarse_df.iloc[TOP_K-1].AIC - coarse_df.iloc[0].AIC
    print(f'粗筛第1与第{TOP_K}模型 AIC 差: {gap:.4f}')
    if gap < SMALL_GAP:
        print('AIC 差距很小，跳过精调阶段。')
        need_fine = False
else:
    print(f'可用模型不足 TOP_K={TOP_K}，跳过精调。')
    need_fine = False

fine_df = None
if need_fine and FINE_EXPAND:
    base_top = coarse_df.head(TOP_K)
    local_grid = _make_local_grid(base_top, max(p_range), max(q_range), max(P_range), max(Q_range))
    coarse_set = set(param_grid)
    local_grid = [g for g in local_grid if g in coarse_set]
    print(f'精调局部网格大小: {len(local_grid)}')
    if local_grid:
        print('开始精调参数搜索...')
        fine_df = parallel_search(train['VALUE'], local_grid, N_JOBS, FINE_MAXITER, '并行拟合(精调)', 'fine')
        print('精调完成，前 5 个结果:')
        print(fine_df.head(min(5, len(fine_df))))
    else:
        print('局部扩展网格为空，跳过精调。')
        need_fine = False

if need_fine and fine_df is not None and not fine_df.empty:
    best_all = pd.concat([coarse_df.head(TOP_K), fine_df], ignore_index=True).sort_values('AIC').reset_index(drop=True)
    final_row = best_all.iloc[0]
    print('最终最优参数来自精调合并结果:')
else:
    final_row = coarse_df.iloc[0]
    print('最终最优参数来自粗筛结果（或跳过精调）。')

print(final_row)
final_order = (int(final_row.p), int(final_row.d), int(final_row.q))
final_seasonal = (int(final_row.P), int(final_row.D), int(final_row.Q), int(final_row.s))
print(f'最终使用 order={final_order}, seasonal_order={final_seasonal}')

# -----------------------------------------------
# 最终模型拟合
# -----------------------------------------------
final_model = _build_model(data['VALUE'], final_order, final_seasonal, simple_diff=False)
final_res = _fit_robust(final_model, maxiter=max(COARSE_MAXITER, FINE_MAXITER, 500), stage='fine')
if final_res is None and (final_order[1] > 0 or final_seasonal[1] > 0):
    final_model_sd = _build_model(data['VALUE'], final_order, final_seasonal, simple_diff=True)
    final_res = _fit_robust(final_model_sd, maxiter=max(COARSE_MAXITER, FINE_MAXITER, 600), stage='fine')
if final_res is None:
    raise RuntimeError('最终模型多优化器尝试仍未收敛，请缩小搜索空间或调整差分/季节参数。')

print(final_res.summary())

# -----------------------------------------------
# 预测评估
# -----------------------------------------------
forecast = final_res.get_forecast(steps=TEST_STEPS)
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

fig, ax = plt.subplots(1,2, figsize=(14,4))
plot_acf(data['VALUE'].diff().dropna() if d==1 else data['VALUE'], ax=ax[0], lags=60)
ax[0].set_title('ACF')
plot_pacf(data['VALUE'].diff().dropna() if d==1 else data['VALUE'], ax=ax[1], lags=60, method='ywm')
ax[1].set_title('PACF')
plt.tight_layout()

try:
    decomp = seasonal_decompose(data['VALUE'], period=final_seasonal[3], model='additive', extrapolate_trend='freq')
    decomp.plot()
    plt.suptitle(f'季节分解 (period={final_seasonal[3]})')
    plt.tight_layout()
except Exception as e:
    print('季节分解失败:', e)

plt.figure(figsize=(12,4))
plt.plot(data['VALUE'], label='实际')
plt.plot(final_res.fittedvalues, label='拟合', alpha=0.7)
plt.title('拟合值 vs 实际值')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(12,4))
plt.plot(data.index, data['VALUE'], label='历史')
future_index = pd.date_range(start=data.index[-1] + pd.Timedelta(minutes=1), periods=TEST_STEPS, freq='T')
plt.plot(future_index, forecast_mean, label='预测')
plt.fill_between(future_index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1], color='lightgray', alpha=0.5, label='95% CI')
plt.title(f'{TEST_STEPS} 步预测')
plt.legend()
plt.tight_layout()

try:
    final_res.plot_diagnostics(figsize=(12,8))
    plt.suptitle('残差诊断图')
    plt.tight_layout()
except Exception as e:
    print('残差诊断绘图失败:', e)

plt.show()

print('===== 模型选择与结果小结 =====')
print(f'最优非季节参数 (p,d,q): {final_order}')
print(f'最优季节参数 (P,D,Q,s): {final_seasonal}')
print(f'最优模型 AIC: {final_row.AIC:.2f}')
print(f'测试集 MSE: {mse:.4f}')
print('两阶段配置: COARSE_MAXITER={} FINE_MAXITER={} TOP_K={} SMALL_GAP={} FINE_EXPAND={}'.format(
    COARSE_MAXITER, FINE_MAXITER, TOP_K, SMALL_GAP, FINE_EXPAND))
print('并发设置 n_jobs={}，BLAS 线程固定为 1；未收敛模型已自动剔除。'.format(_get_n_jobs()))
print('已屏蔽 ConvergenceWarning；若需诊断具体未收敛原因，可临时注释该过滤行。')
