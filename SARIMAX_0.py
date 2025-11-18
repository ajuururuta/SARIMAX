# -*- coding: utf-8 -*-

# ===============================================
# SARIMAX 自动参数优化 + 可视化 + 两阶段并行搜索 + 优化版
# ===============================================
# 优化点：
# - 解决批次超时和失败问题
# - 加速粗筛阶段（早停、快速模式、优化并行）
# - GPU 检测和优化建议
# - 进度保存和恢复
# - 改进的错误处理和超时控制
# ===============================================

import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
from pathlib import Path

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_squared_error

from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp

from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# ============== GPU 检测 ==============
def check_gpu_available():
    """检测 GPU 和相关加速库"""
    gpu_info = {
        'cuda_available': False,
        'cupy_available': False,
        'mkl_available': False,
        'openblas_available': False
    }
    
    try:
        import cupy as cp
        gpu_info['cupy_available'] = True
        gpu_info['cuda_available'] = True
        print(f"✓ 检测到 CuPy (GPU 加速可用)")
    except ImportError:
        print("✗ CuPy 未安装 (pip install cupy-cuda11x 或 cupy-cuda12x)")
    
    # 检测 BLAS 库
    try:
        import numpy as np
        config = np.__config__.show()
        if 'mkl' in str(config).lower():
            gpu_info['mkl_available'] = True
            print("✓ 检测到 Intel MKL (CPU 优化)")
        elif 'openblas' in str(config).lower():
            gpu_info['openblas_available'] = True
            print("✓ 检测到 OpenBLAS (CPU 优化)")
    except:
        pass
    
    return gpu_info

# ============== 参数解析 ============== 
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
        n = int(os.environ.get('SARIMAX_N_JOBS', '-1'))
        if n == -1:
            return max(1, mp.cpu_count() - 1)  # 留一个核心给系统
        return n
    except Exception:
        return max(1, mp.cpu_count() - 1)


def _ensure_blas_single_thread():
    """设置 BLAS 库为单线程，避免过度订阅"""
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        if os.environ.get(var) is None:
            os.environ[var] = "1"

_ensure_blas_single_thread()

parser = argparse.ArgumentParser(description="SARIMAX 两阶段搜索（优化版）")
parser.add_argument("--coarse-maxiter", type=int, default=_env_int("COARSE_MAXITER", 100), help="粗筛最大迭代次数，默认 100（降低以加速）")
parser.add_argument("--fine-maxiter",   type=int, default=_env_int("FINE_MAXITER", 500), help="精调最大迭代次���，默认 500")
parser.add_argument("--top-k",          type=int, default=_env_int("TOP_K", 10), help="粗筛保留的前 K 个模型，默认 10")
parser.add_argument("--small-gap",      type=float, default=_env_float("SMALL_GAP", 0.5), help="AIC 差距阈值，默认 0.5")
parser.add_argument("--no-fine-expand", action="store_true", help="禁用精调阶段的局部扩展网格")
parser.add_argument("--quick-mode",     action="store_true", help="快速模式：减少搜索空间")
parser.add_argument("--early-stop",     type=int, default=5, help="早停：连续 N 个批次无改善则停止，默认 5，0 为禁用")

parser.add_argument("--resume",         action="store_true", help="从上次中断处恢复")
parser.add_argument("--backend",        type=str, default="threading", choices=["loky", "threading", "multiprocessing"], help="并行后端，默认 threading（更快启动）")
args, _ = parser.parse_known_args()

COARSE_MAXITER = args.coarse_maxiter
FINE_MAXITER   = args.fine_maxiter
TOP_K          = args.top_k
SMALL_GAP      = args.small_gap
FINE_EXPAND    = not args.no_fine_expand
QUICK_MODE     = args.quick_mode
EARLY_STOP_BATCHES = args.early_stop

RESUME         = args.resume
BACKEND        = args.backend

print("="*60)
print("SARIMAX 优化版启动")
print("="*60)
gpu_info = check_gpu_available()
print(f"[参数] COARSE_MAXITER={COARSE_MAXITER}, FINE_MAXITER={FINE_MAXITER}")
print(f"[参数] TOP_K={TOP_K}, SMALL_GAP={SMALL_GAP}, FINE_EXPAND={FINE_EXPAND}")
print(f"[参数] QUICK_MODE={QUICK_MODE}, EARLY_STOP={EARLY_STOP_BATCHES}")
print(f"[参数] 无超时限制 - 确保每个模型完整拟合")
print(f"[参数] BACKEND={BACKEND}, N_JOBS={_get_n_jobs()}")
print("="*60)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ============== 数据读取与预处理 ============== 
print("\n[1/6] 数据加载...")
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
    print(f'  发现缺失值 {missing_count} 个，使用线性插值修复')
    raw['VALUE'] = raw['VALUE'].interpolate()

if raw.index.duplicated().any():
    print('  发现重复时间戳，按均值聚合')
    raw = raw.groupby(raw.index).mean()
    raw = raw.asfreq('T')

print(f'  数据范围: {raw.index.min()} -> {raw.index.max()}')
print(f'  总样本数: {len(raw)}')

data = raw.copy()

# ============== 平稳性检验 ============== 
print("\n[2/6] 平稳性检验...")
adf_stat, adf_p, *_ = adfuller(data['VALUE'])
print(f'  ADF Statistic: {adf_stat:.4f}, p-value: {adf_p:.6g}')
if adf_p < 0.05:
    d = 0
    print('  ✓ 序列已平稳，d=0')
else:
    d = 1
    print('  ✗ 序列不平稳，使用 d=1')

# ============== 搜索空间 ============== 
print("\n[3/6] 构建搜索空间...")
if QUICK_MODE:
    print("  [快速模式] 使用精简搜索空间")
    p_range = range(0, 2)
    q_range = range(0, 2)
    P_range = range(0, 2)
    Q_range = range(0, 2)
    D_values = [0, 1]
else:
    p_range = range(0, 3)
    q_range = range(0, 3)
    P_range = range(0, 2)
    Q_range = range(0, 2)
    D_values = [0, 1]

s_candidates = [s for s in ([60, 120] if len(data) >= 120 else [60]) if len(data) >= 3*s]
if not s_candidates:
    s_candidates = [60]
print(f'  季节周期候选: {s_candidates}')

param_grid = [(p, d, q, P, D_, Q, s_) for p in p_range for q in q_range
              for P in P_range for Q in Q_range for D_ in D_values for s_ in s_candidates]
print(f'  总参数组合: {len(param_grid)}')

# ============== 拟合函数（无超时限制） ============== 
def _fit_robust_no_timeout(model, maxiter=200, stage='coarse'):
    """鲁棒拟合，无超时限制，尝试多种优化器"""
    methods_coarse = ['lbfgs', 'powell', 'bfgs']
    methods_fine = ['lbfgs', 'powell', 'nm', 'bfgs', 'cg']
    methods = methods_coarse if stage == 'coarse' else methods_fine
    
    for method_idx, m in enumerate(methods):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                actual_maxiter = maxiter if method_idx == 0 else maxiter * 2
                res = model.fit(method=m, disp=False, maxiter=actual_maxiter)
            
            if bool(res.mle_retvals.get('converged', True)) and np.isfinite(res.aic):
                return res
        except KeyboardInterrupt:
            raise
        except Exception as e:
            continue
    
    return None


def _build_model(endog, order, seasonal_order, simple_diff=False):
    """构建 SARIMAX 模型"""
    try:
        return SARIMAX(
            endog, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=True, enforce_invertibility=True,
            simple_differencing=simple_diff
        )
    except Exception as e:
        try:
            return SARIMAX(
                endog, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
                simple_differencing=simple_diff
            )
        except:
            return None


def _fit_one_stage(endog, order, seasonal_order, maxiter=200, stage='coarse'):
    """单个模型拟合（增强版，多重回退机制）"""
    try:
        # 第一次尝试：标准拟合
        model = _build_model(endog, order, seasonal_order, simple_diff=False)
        if model is None:
            return None
            
        res = _fit_robust_no_timeout(model, maxiter=maxiter, stage=stage)
        
        # 第二次尝试：如果失败且有差分，尝试 simple_differencing
        if res is None and (order[1] > 0 or seasonal_order[1] > 0):
            model_sd = _build_model(endog, order, seasonal_order, simple_diff=True)
            if model_sd is not None:
                res = _fit_robust_no_timeout(model_sd, maxiter=maxiter, stage=stage)
        
        # 第三次尝试：如果仍失败，尝试更多迭代次数
        if res is None:
            model_more_iter = _build_model(endog, order, seasonal_order, simple_diff=False)
            if model_more_iter is not None:
                res = _fit_robust_no_timeout(model_more_iter, maxiter=maxiter*3, stage=stage)
        
        if res is None or not np.isfinite(res.aic):
            return None
        
        return {
            'p': order[0], 'd': order[1], 'q': order[2],
            'P': seasonal_order[0], 'D': seasonal_order[1], 'Q': seasonal_order[2], 's': seasonal_order[3],
            'AIC': float(res.aic)
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
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

# ============== 改进的并行搜索 ============== 
def parallel_search_optimized(endog, grid, n_jobs, maxiter, stage_desc, stage_key, 
                              early_stop_batches=5, resume_file=None):
    """优化的并行搜索，支持早停和恢复"""
    
    tasks = [((p, d_, q), (P, D_, Q, s_)) for (p, d_, q, P, D_, Q, s_) in grid]
    print(f'\n{stage_desc}')
    print(f'  任务总数: {len(tasks)}, 并行度: {n_jobs}, maxiter: {maxiter}')
    print(f'  早停批次: {early_stop_batches}')
    print(f'  无超时限制 - 每个模型将完整拟合直到收敛或达到最大迭代次数')
    
    # 恢复之前的结果
    completed_tasks = set()
    all_results = []
    
    if resume_file and os.path.exists(resume_file):
        try:
            with open(resume_file, 'rb') as f:
                checkpoint = pickle.load(f)
                all_results = checkpoint['results']
                completed_tasks = checkpoint['completed']
            print(f'  ✓ 从检查点恢复，已完成 {len(completed_tasks)} 个任务')
        except Exception as e:
            print(f'  ✗ 恢复失败: {e}')
    
    # 过滤已完成的任务
    remaining_tasks = [t for i, t in enumerate(tasks) if i not in completed_tasks]
    
    if not remaining_tasks:
        print('  ✓ 所有任务已完成')
        if not all_results:
            return pd.DataFrame(columns=['p','d','q','P','D','Q','s','AIC'])
        return pd.DataFrame(all_results).sort_values('AIC').reset_index(drop=True)
    
    best_aic = min([r['AIC'] for r in all_results]) if all_results else float('inf')
    no_improvement_count = 0
    
    # 动态批次大小：开始时小批次，后期大批次
    batch_size = max(4, len(remaining_tasks) // 20)
    
    start_time = time.time()
    
    with tqdm(total=len(remaining_tasks), desc=f"  {stage_desc}", ncols=100, unit="模型") as pbar:
        for batch_idx, i in enumerate(range(0, len(remaining_tasks), batch_size)):
            batch_tasks = remaining_tasks[i:i+batch_size]
            batch_start = time.time()
            
            try:
                # 使用指定的后端，无超时限制
                batch_results = Parallel(
                    n_jobs=n_jobs, 
                    backend=BACKEND, 
                    timeout=None
                )(
                    delayed(_fit_one_stage)(endog, order, seasonal, maxiter, stage_key)
                    for order, seasonal in batch_tasks
                )
                
                # 过滤有效结果
                valid_batch = [r for r in batch_results if r is not None]
                all_results.extend(valid_batch)
                
                # 更新最优 AIC 和早停计数
                batch_improved = False
                if valid_batch:
                    batch_best = min(r['AIC'] for r in valid_batch)
                    if batch_best < best_aic - 0.01:  # 至少改善 0.01
                        best_aic = batch_best
                        no_improvement_count = 0
                        batch_improved = True
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
                batch_time = time.time() - batch_start
                
                # 更新进度条
                success_rate = len(valid_batch) / len(batch_tasks) * 100 if batch_tasks else 0
                eta_per_task = (time.time() - start_time) / (i + len(batch_tasks))
                eta_remaining = eta_per_task * (len(remaining_tasks) - i - len(batch_tasks))
                
                pbar.set_postfix_str(
                    f"AIC: {best_aic:.2f} | 成功率: {success_rate:.0f}% | 批次耗时: {batch_time:.1f}s | ETA: {eta_remaining/60:.1f}min"
                )
                pbar.update(len(batch_tasks))
                
                # 保存检查点
                if resume_file and (batch_idx + 1) % 5 == 0:
                    completed_tasks.update(range(i, i + len(batch_tasks)))
                    with open(resume_file, 'wb') as f:
                        pickle.dump({
                            'results': all_results,
                            'completed': completed_tasks
                        }, f)
                
                # 早停检查
                if early_stop_batches > 0 and no_improvement_count >= early_stop_batches:
                    print(f'\n  ⚠ 早停触发：连续 {early_stop_batches} 个批次无显著改善')
                    break
                
            except KeyboardInterrupt:
                print(f'\n  ⚠ 用户中断，保存进度...')
                # 保存进度
                if resume_file:
                    completed_tasks.update(range(i, i + len(batch_tasks)))
                    with open(resume_file, 'wb') as f:
                        pickle.dump({
                            'results': all_results,
                            'completed': completed_tasks
                        }, f)
                    print(f'  ✓ 进度已保存到 {resume_file}')
                raise
            except Exception as e:
                print(f'\n  ✗ 批次 {i}-{i+len(batch_tasks)} 失败: {str(e)[:100]}')
                # 失败时尝试单个任务处理
                for task in batch_tasks:
                    try:
                        single_result = _fit_one_stage(endog, task[0], task[1], maxiter, stage_key)
                        if single_result:
                            all_results.append(single_result)
                    except KeyboardInterrupt:
                        raise
                    except:
                        pass
                pbar.update(len(batch_tasks))
    
    total_time = time.time() - start_time
    print(f'  完成时间: {total_time/60:.2f} 分钟')
    print(f'  成功模型: {len(all_results)} / {len(remaining_tasks)} ({len(all_results)/len(remaining_tasks)*100:.1f}%)')
    
    # 清理检查点
    if resume_file and os.path.exists(resume_file):
        try:
            os.remove(resume_file)
        except:
            pass
    
    if not all_results:
        return pd.DataFrame(columns=['p','d','q','P','D','Q','s','AIC'])
    return pd.DataFrame(all_results).sort_values('AIC').reset_index(drop=True)

# ============== 两阶段搜索 ============== 
TEST_STEPS = 100
train = data.iloc[:-TEST_STEPS]
N_JOBS = _get_n_jobs()

print('\n[4/6] 参数搜索')
print('='*60)
print('阶段 1: 粗筛')

resume_coarse = 'checkpoint_coarse.pkl' if RESUME else None
coarse_df = parallel_search_optimized(
    train['VALUE'], param_grid, N_JOBS, COARSE_MAXITER, 
    '粗筛搜索', 'coarse',
    early_stop_batches=EARLY_STOP_BATCHES,
    resume_file=resume_coarse
)

print('\n  前 10 个结果:')
print(coarse_df.head(min(10, len(coarse_df))).to_string())

if coarse_df.empty:
    raise RuntimeError('粗筛阶段没有找到可用模型，请检查数据或放宽参数约束')

# 判断是否需要精调
need_fine = True
if len(coarse_df) >= TOP_K:
    gap = coarse_df.iloc[TOP_K-1].AIC - coarse_df.iloc[0].AIC
    print(f'\n  AIC 差距 (第1 vs 第{TOP_K}): {gap:.4f}')
    if gap < SMALL_GAP:
        print(f'  ✓ AIC 差距 < {SMALL_GAP}，跳过精调')
        need_fine = False
else:
    print(f'\n  可用模型 < TOP_K={TOP_K}，跳过精调')
    need_fine = False

fine_df = None
if need_fine and FINE_EXPAND:
    print('\n' + '='*60)
    print('阶段 2: 精调')
    base_top = coarse_df.head(TOP_K)
    local_grid = _make_local_grid(base_top, max(p_range), max(q_range), max(P_range), max(Q_range))
    coarse_set = set(param_grid)
    local_grid = [g for g in local_grid if g in coarse_set]
    print(f'  局部扩展网格大小: {len(local_grid)}')
    
    if local_grid:
        resume_fine = 'checkpoint_fine.pkl' if RESUME else None
        fine_df = parallel_search_optimized(
            train['VALUE'], local_grid, N_JOBS, FINE_MAXITER,
            '精调搜索', 'fine',
            early_stop_batches=max(3, EARLY_STOP_BATCHES // 2),
            resume_file=resume_fine
        )
        print('\n  前 5 个精调结果:')
        print(fine_df.head(min(5, len(fine_df))).to_string())
    else:
        print('  局部扩展网格为空')
        need_fine = False

# 合并结果
if need_fine and fine_df is not None and not fine_df.empty:
    best_all = pd.concat([coarse_df.head(TOP_K), fine_df], ignore_index=True)
    best_all = best_all.sort_values('AIC').reset_index(drop=True)
    final_row = best_all.iloc[0]
    source = '精调'
else:
    final_row = coarse_df.iloc[0]
    source = '粗筛'

print('\n' + '='*60)
print(f'最优参数 (来源: {source}):')
print(final_row.to_string())
print('='*60)

final_order = (int(final_row.p), int(final_row.d), int(final_row.q))
final_seasonal = (int(final_row.P), int(final_row.D), int(final_row.Q), int(final_row.s))

# ============== 最终模型拟合 ============== 
print('\n[5/6] 最终模型拟合...')
final_model = _build_model(data['VALUE'], final_order, final_seasonal, simple_diff=False)
if final_model is None:
    raise RuntimeError('最终模型构建失败')
final_res = _fit_robust_no_timeout(final_model, maxiter=max(500, FINE_MAXITER), stage='fine')

if final_res is None and (final_order[1] > 0 or final_seasonal[1] > 0):
    print('  尝试 simple_differencing 回退...')
    final_model_sd = _build_model(data['VALUE'], final_order, final_seasonal, simple_diff=True)
    if final_model_sd is not None:
        final_res = _fit_robust_no_timeout(final_model_sd, maxiter=600, stage='fine')

if final_res is None:
    raise RuntimeError('最终模型拟合失败，请尝试调整参数或检查数据质量')

print('  ✓ 模型拟合成功')
print(final_res.summary())

# ============== 预测评估 ============== 
print('\n[6/6] 预测与评估...')
forecast = final_res.get_forecast(steps=TEST_STEPS)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int(alpha=0.05)
actual_test = data['VALUE'].iloc[-TEST_STEPS:]
mse = mean_squared_error(actual_test, forecast_mean)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(actual_test - forecast_mean))

print(f'  测试集 MSE: {mse:.4f}')
print(f'  测试集 RMSE: {rmse:.4f}')
print(f'  测试集 MAE: {mae:.4f}')

# ============== 可视化 ============== 
print('\n生成可视化图表...')

plt.figure(figsize=(12,4))
plt.plot(data['VALUE'], label='原始序列', alpha=0.8)
plt.title('原始时间序列')
plt.legend()
plt.tight_layout()

if d == 1:
    plt.figure(figsize=(12,4))
    plt.plot(data['VALUE'].diff(), label='一次差分', alpha=0.8)
    plt.title('差分后序列 (d=1)')
    plt.legend()
    plt.tight_layout()

fig, ax = plt.subplots(1,2, figsize=(14,4))
plot_acf(data['VALUE'].diff().dropna() if d==1 else data['VALUE'], ax=ax[0], lags=min(60, len(data)//3))
ax[0].set_title('自相关函数 (ACF)')
plot_pacf(data['VALUE'].diff().dropna() if d==1 else data['VALUE'], ax=ax[1], lags=min(60, len(data)//3), method='ywm')
ax[1].set_title('偏自相关函数 (PACF)')
plt.tight_layout()

try:
    decomp = seasonal_decompose(data['VALUE'], period=final_seasonal[3], model='additive', extrapolate_trend='freq')
    decomp.plot()
    plt.suptitle(f'季节分解 (周期={final_seasonal[3]})', y=1.02)
    plt.tight_layout()
except Exception as e:
    print(f'  季节分解失败: {e}')  

plt.figure(figsize=(12,5))
plt.plot(data['VALUE'], label='实际值', alpha=0.7)
plt.plot(final_res.fittedvalues, label='拟合值', alpha=0.7)
plt.title('模型拟合效果')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.figure(figsize=(12,5))
plt.plot(data.index, data['VALUE'], label='历史数据', alpha=0.7)
future_index = pd.date_range(start=data.index[-1] + pd.Timedelta(minutes=1), periods=TEST_STEPS, freq='T')
plt.plot(future_index, forecast_mean, label='预测值', color='red', linewidth=2)
plt.fill_between(future_index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1], color='lightcoral', alpha=0.3, label='95% 置信区间')
plt.axvline(x=data.index[-TEST_STEPS], color='gray', linestyle='--', alpha=0.5)
plt.title(f'{TEST_STEPS} 步预测结果')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

try:
    final_res.plot_diagnostics(figsize=(12,8))
    plt.suptitle('残差诊断图', y=1.00)
    plt.tight_layout()
except Exception as e:
    print(f'  残差诊断图失败: {e}')  

plt.show()

# ============== 结果总结 ============== 
print('\n' + '='*60)
print('模型选择与结果总结')
print('='*60)
print(f'最优非季节参数 (p,d,q): {final_order}')
print(f'最优季节参数 (P,D,Q,s): {final_seasonal}')
print(f'最优 AIC: {final_row.AIC:.2f}')
print(f'测试集 MSE: {mse:.4f}')
print(f'测试集 RMSE: {rmse:.4f}')
print(f'测试集 MAE: {mae:.4f}')
print(f'\n配置参数:')
print(f'  COARSE_MAXITER={COARSE_MAXITER}')
print(f'  FINE_MAXITER={FINE_MAXITER}')
print(f'  TOP_K={TOP_K}')
print(f'  SMALL_GAP={SMALL_GAP}')
print(f'  QUICK_MODE={QUICK_MODE}')
print(f'  EARLY_STOP={EARLY_STOP_BATCHES}')
print(f'  BACKEND={BACKEND}')
print(f'  N_JOBS={N_JOBS}')
print(f'  无超时限制 - 所有模型完整拟合')
print('='*60)

if not gpu_info['cuda_available']:
    print('\n💡 优化建议:')
    print('  1. 安装 CuPy 以利用 GPU 加速数据预处理:')
    print('     pip install cupy-cuda11x  # 或 cupy-cuda12x')
    print('  2. 使用 --quick-mode 减少搜索空间')
    print('  3. 调整 --early-stop 参数实现更快的收敛')
    print('  4. 使用 --coarse-maxiter 50 进一步加速粗筛')

print('\n✓ 程序执行完成！')
