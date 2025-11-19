# -*- coding: utf-8 -*-

# ===============================================
# SARIMAX 性能优化版本 - 智能搜索 + 缓存 + 增强并行
# ===============================================
# 优化特性：
# 1. 贝叶斯优化替代网格搜索
# 2. 模型结果缓存系统
# 3. 早停机制
# 4. 智能参数剪枝
# 5. 多阶段搜索优化
# 6. 性能监控和日志
# 7. 内存管理优化
# 8. 配置管理系统
# ===============================================

import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import time
import hashlib
import pickle
import json
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any

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
import gc

# 贝叶斯优化支持
try:
    from skopt import gp_minimize
    from skopt.space import Integer, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: scikit-optimize not available, falling back to grid search")

from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# ===============================================
# 配置管理系统
# ===============================================
class OptimizationConfig:
    """优化配置管理类"""
    
    def __init__(self):
        # 基础参数
        self.coarse_maxiter = 50
        self.fine_maxiter = 300
        self.top_k = 10
        self.small_gap = 0.5
        self.fine_expand = True
        
        # 优化级别 (0=原始, 1=基础优化, 2=激进优化)
        self.optimization_level = 1
        
        # 贝叶斯优化参数
        self.use_bayesian = BAYESIAN_AVAILABLE
        self.bayesian_n_calls = 50
        self.bayesian_n_initial_points = 10
        
        # 缓存配置
        self.enable_cache = True
        self.cache_dir = Path('.sarimax_cache')
        
        # 早停配置
        self.early_stop_patience = 5
        self.early_stop_threshold = 0.01
        
        # 内存管理
        self.gc_frequency = 10
        self.max_memory_models = 50
        
        # 性能监控
        self.enable_profiling = True
        self.log_file = 'sarimax_optimization.log'
        
    def from_args(self, args):
        """从命令行参数更新配置"""
        if hasattr(args, 'optimization_level'):
            self.optimization_level = args.optimization_level
        if hasattr(args, 'coarse_maxiter'):
            self.coarse_maxiter = args.coarse_maxiter
        if hasattr(args, 'fine_maxiter'):
            self.fine_maxiter = args.fine_maxiter
        if hasattr(args, 'top_k'):
            self.top_k = args.top_k
        if hasattr(args, 'small_gap'):
            self.small_gap = args.small_gap
        if hasattr(args, 'no_fine_expand'):
            self.fine_expand = not args.no_fine_expand
        if hasattr(args, 'no_cache'):
            self.enable_cache = not args.no_cache
        if hasattr(args, 'no_bayesian'):
            self.use_bayesian = not args.no_bayesian and BAYESIAN_AVAILABLE
        return self
    
    def apply_optimization_level(self):
        """根据优化级别调整参数"""
        if self.optimization_level == 0:
            # 原始模式，最保守
            self.use_bayesian = False
            self.enable_cache = False
            self.early_stop_patience = 999
        elif self.optimization_level == 1:
            # 基础优化，平衡速度和准确性
            self.bayesian_n_calls = 50
            self.early_stop_patience = 5
        elif self.optimization_level >= 2:
            # 激进优化，最快速度
            self.bayesian_n_calls = 30
            self.early_stop_patience = 3
            self.coarse_maxiter = min(self.coarse_maxiter, 30)
            self.top_k = min(self.top_k, 5)

# ===============================================
# 性能监控和日志
# ===============================================
class PerformanceMonitor:
    """性能监控类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'models_evaluated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'early_stops': 0,
            'memory_cleanups': 0,
            'stage_times': {},
        }
        self.stage_stack = []
        
    def start_stage(self, stage_name: str):
        """开始一个阶段"""
        self.stage_stack.append({
            'name': stage_name,
            'start': time.time()
        })
        
    def end_stage(self):
        """结束当前阶段"""
        if self.stage_stack:
            stage = self.stage_stack.pop()
            duration = time.time() - stage['start']
            self.metrics['stage_times'][stage['name']] = duration
            return duration
        return 0
    
    def log(self, message: str):
        """记录日志"""
        if self.config.enable_profiling:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            log_msg = f"[{timestamp}] {message}"
            print(log_msg)
            if self.config.log_file:
                with open(self.config.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_msg + '\n')
    
    def summary(self) -> str:
        """生成性能摘要"""
        total_time = self.metrics.get('end_time', time.time()) - self.metrics.get('start_time', time.time())
        cache_rate = 0
        if self.metrics['cache_hits'] + self.metrics['cache_misses'] > 0:
            cache_rate = self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) * 100
        
        summary = [
            "\n" + "="*60,
            "性能监控摘要",
            "="*60,
            f"总用时: {total_time:.2f}秒",
            f"评估模型数: {self.metrics['models_evaluated']}",
            f"缓存命中率: {cache_rate:.1f}% ({self.metrics['cache_hits']}/{self.metrics['cache_hits']+self.metrics['cache_misses']})",
            f"早停次数: {self.metrics['early_stops']}",
            f"内存清理次数: {self.metrics['memory_cleanups']}",
            "\n阶段耗时:",
        ]
        
        for stage, duration in self.metrics['stage_times'].items():
            summary.append(f"  {stage}: {duration:.2f}秒")
        
        summary.append("="*60)
        return '\n'.join(summary)

# ===============================================
# 模型缓存系统
# ===============================================
class ModelCache:
    """模型结果缓存系统"""
    
    def __init__(self, config: OptimizationConfig, monitor: PerformanceMonitor):
        self.config = config
        self.monitor = monitor
        self.memory_cache = {}
        self.cache_dir = config.cache_dir
        
        if config.enable_cache:
            self.cache_dir.mkdir(exist_ok=True)
    
    def _get_key(self, endog: pd.Series, order: Tuple, seasonal_order: Tuple) -> str:
        """生成缓存键"""
        # 使用数据哈希 + 参数组合
        data_hash = hashlib.md5(endog.values.tobytes()).hexdigest()[:8]
        param_str = f"{order}_{seasonal_order}"
        return f"{data_hash}_{param_str}"
    
    def get(self, endog: pd.Series, order: Tuple, seasonal_order: Tuple) -> Optional[Dict]:
        """从缓存获取结果"""
        if not self.config.enable_cache:
            return None
        
        key = self._get_key(endog, order, seasonal_order)
        
        # 先检查内存缓存
        if key in self.memory_cache:
            self.monitor.metrics['cache_hits'] += 1
            return self.memory_cache[key]
        
        # 检查磁盘缓存
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                self.memory_cache[key] = result
                self.monitor.metrics['cache_hits'] += 1
                return result
            except Exception:
                pass
        
        self.monitor.metrics['cache_misses'] += 1
        return None
    
    def put(self, endog: pd.Series, order: Tuple, seasonal_order: Tuple, result: Dict):
        """保存结果到缓存"""
        if not self.config.enable_cache:
            return
        
        key = self._get_key(endog, order, seasonal_order)
        
        # 保存到内存
        self.memory_cache[key] = result
        
        # 控制内存缓存大小
        if len(self.memory_cache) > self.config.max_memory_models:
            # 移除最老的一半
            keys = list(self.memory_cache.keys())
            for k in keys[:len(keys)//2]:
                del self.memory_cache[k]
            self.monitor.metrics['memory_cleanups'] += 1
            gc.collect()
        
        # 保存到磁盘
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception:
            pass
    
    def clear(self):
        """清空缓存"""
        self.memory_cache.clear()
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass

# ===============================================
# 早停机制
# ===============================================
class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int, threshold: float):
        self.patience = patience
        self.threshold = threshold
        self.best_score = float('inf')
        self.counter = 0
        self.history = []
    
    def should_stop(self, score: float) -> bool:
        """判断是否应该早停"""
        self.history.append(score)
        
        if score < self.best_score - self.threshold:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def reset(self):
        """重置早停状态"""
        self.best_score = float('inf')
        self.counter = 0
        self.history = []

# ===============================================
# 参数辅助函数
# ===============================================
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

# ===============================================
# 命令行参数解析
# ===============================================
parser = argparse.ArgumentParser(description="SARIMAX 性能优化版本")
parser.add_argument("--optimization-level", type=int, default=1, choices=[0, 1, 2],
                    help="优化级别: 0=原始, 1=基础优化, 2=激进优化 (默认: 1)")
parser.add_argument("--coarse-maxiter", type=int, default=_env_int("COARSE_MAXITER", 50),
                    help="粗筛最大迭代次数 (默认: 50)")
parser.add_argument("--fine-maxiter", type=int, default=_env_int("FINE_MAXITER", 300),
                    help="精调最大迭代次数 (默认: 300)")
parser.add_argument("--top-k", type=int, default=_env_int("TOP_K", 10),
                    help="粗筛保留的前 K 个模型 (默认: 10)")
parser.add_argument("--small-gap", type=float, default=_env_float("SMALL_GAP", 0.5),
                    help="AIC 差距阈值 (默认: 0.5)")
parser.add_argument("--no-fine-expand", action="store_true",
                    help="禁用精调阶段的局部扩展网格")
parser.add_argument("--no-cache", action="store_true",
                    help="禁用模型缓存")
parser.add_argument("--no-bayesian", action="store_true",
                    help="禁用贝叶斯优化")
parser.add_argument("--clear-cache", action="store_true",
                    help="清空缓存后退出")

args, _ = parser.parse_known_args()

# 创建配置
config = OptimizationConfig()
config.from_args(args)
config.apply_optimization_level()

# 如果只是清空缓存
if args.clear_cache:
    cache_dir = config.cache_dir
    if cache_dir.exists():
        for f in cache_dir.glob("*.pkl"):
            f.unlink()
        print(f"缓存已清空: {cache_dir}")
    exit(0)

# 创建监控器和缓存
monitor = PerformanceMonitor(config)
cache = ModelCache(config, monitor)

monitor.log(f"优化配置: level={config.optimization_level}, bayesian={config.use_bayesian}, cache={config.enable_cache}")
monitor.log(f"参数: COARSE_MAXITER={config.coarse_maxiter}, FINE_MAXITER={config.fine_maxiter}, TOP_K={config.top_k}")

# 图表设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ===============================================
# 数据读取与预处理
# ===============================================
monitor.start_stage('数据预处理')

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
    monitor.log(f'发现缺失值 {missing_count} 个，使用线性插值修复')
    raw['VALUE'] = raw['VALUE'].interpolate()

if raw.index.duplicated().any():
    monitor.log('发现重复时间戳，按均值聚合')
    raw = raw.groupby(raw.index).mean()
    raw = raw.asfreq('T')

monitor.log(f'数据起止时间: {raw.index.min()} -> {raw.index.max()}')
monitor.log(f'总样本数: {len(raw)}')

data = raw.copy()

monitor.end_stage()

# ===============================================
# 平稳性检验
# ===============================================
monitor.start_stage('平稳性检验')

adf_stat, adf_p, *_ = adfuller(data['VALUE'])
monitor.log(f'原始序列 ADF Statistic: {adf_stat:.4f}, p-value: {adf_p:.6g}')

if adf_p < 0.05:
    d = 0
    monitor.log('原始序列已平稳，d=0')
else:
    d = 1
    monitor.log('原始序列不平稳，采用一次差分 d=1')
    data['VALUE_DIFF'] = data['VALUE'].diff()

if d == 1:
    diff_adf_stat, diff_adf_p, *_ = adfuller(data['VALUE'].dropna().diff().dropna())
    monitor.log(f'差分后 ADF Statistic: {diff_adf_stat:.4f}, p-value: {diff_adf_p:.6g}')

monitor.end_stage()

# ===============================================
# 智能搜索空间设定
# ===============================================
monitor.start_stage('搜索空间设定')

# 基础搜索空间
p_range = range(0, 3)
q_range = range(0, 3)
s_candidates = [s for s in ([60, 120] if len(data) >= 120 else [60]) if len(data) >= 3*s]
if not s_candidates:
    s_candidates = [60]

P_range = range(0, 2)
Q_range = range(0, 2)
D_values = [0, 1]

monitor.log(f'季节候选周期: {s_candidates}')

# 根据优化级别调整搜索空间
if config.optimization_level >= 2:
    # 激进优化：更小的搜索空间
    p_range = range(0, 2)
    q_range = range(0, 2)
    P_range = range(0, 2)
    Q_range = range(0, 1)
    monitor.log('激进优化模式：使用缩减的搜索空间')

param_grid = [(p, d, q, P, D_, Q, s_) for p in p_range for q in q_range
              for P in P_range for Q in Q_range for D_ in D_values for s_ in s_candidates]

monitor.log(f'总参数组合数: {len(param_grid)}')
monitor.end_stage()

# ===============================================
# 模型拟合函数
# ===============================================
def _fit_robust(model, maxiter=200, stage='coarse'):
    """鲁棒拟合函数"""
    methods_coarse = ['lbfgs', 'powell']
    methods_fine = ['lbfgs', 'powell', 'nm', 'bfgs', 'cg']
    methods = methods_coarse if stage == 'coarse' else methods_fine
    
    for m in methods:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                res = model.fit(method=m, disp=False, maxiter=maxiter)
            if bool(res.mle_retvals.get('converged', True)) and np.isfinite(res.aic):
                return res
        except Exception:
            continue
    return None

def _build_model(endog, order, seasonal_order, simple_diff=False):
    """构建SARIMAX模型"""
    return SARIMAX(
        endog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=True,
        simple_differencing=simple_diff
    )

def _fit_one_stage(endog, order, seasonal_order, maxiter=200, stage='coarse'):
    """拟合单个参数组合"""
    try:
        # 检查缓存
        cached_result = cache.get(endog, order, seasonal_order)
        if cached_result is not None:
            return cached_result
        
        # 拟合模型
        model = _build_model(endog, order, seasonal_order, simple_diff=False)
        res = _fit_robust(model, maxiter=maxiter, stage=stage)
        
        if res is None and (order[1] > 0 or seasonal_order[1] > 0):
            model_sd = _build_model(endog, order, seasonal_order, simple_diff=True)
            res = _fit_robust(model_sd, maxiter=maxiter, stage=stage)
        
        if res is None or not np.isfinite(res.aic):
            return None
        
        result = {
            'p': order[0], 'd': order[1], 'q': order[2],
            'P': seasonal_order[0], 'D': seasonal_order[1], 'Q': seasonal_order[2], 's': seasonal_order[3],
            'AIC': float(res.aic)
        }
        
        # 保存到缓存
        cache.put(endog, order, seasonal_order, result)
        monitor.metrics['models_evaluated'] += 1
        
        return result
    except Exception:
        return None

def _clip(v, low, high):
    """限制值在范围内"""
    return max(low, min(high, v))

def _make_local_grid(base_rows, p_max, q_max, P_max, Q_max):
    """生成局部扩展网格"""
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

# ===============================================
# 贝叶斯优化搜索
# ===============================================
def bayesian_search(endog, param_grid, n_calls, n_initial_points):
    """使用贝叶斯优化搜索最优参数"""
    monitor.log(f'使用贝叶斯优化: n_calls={n_calls}, n_initial_points={n_initial_points}')
    
    # 提取参数范围
    p_values = sorted(set([p[0] for p in param_grid]))
    q_values = sorted(set([p[2] for p in param_grid]))
    P_values = sorted(set([p[3] for p in param_grid]))
    Q_values = sorted(set([p[5] for p in param_grid]))
    D_values = sorted(set([p[4] for p in param_grid]))
    s_values = sorted(set([p[6] for p in param_grid]))
    d_values = sorted(set([p[1] for p in param_grid]))
    
    # 定义搜索空间
    space = [
        Integer(min(p_values), max(p_values), name='p'),
        Categorical(d_values, name='d'),
        Integer(min(q_values), max(q_values), name='q'),
        Integer(min(P_values), max(P_values), name='P'),
        Categorical(D_values, name='D'),
        Integer(min(Q_values), max(Q_values), name='Q'),
        Categorical(s_values, name='s'),
    ]
    
    results_list = []
    early_stop = EarlyStopping(config.early_stop_patience, config.early_stop_threshold)
    
    @use_named_args(space)
    def objective(**params):
        """优化目标函数"""
        order = (params['p'], params['d'], params['q'])
        seasonal_order = (params['P'], params['D'], params['Q'], params['s'])
        
        result = _fit_one_stage(endog, order, seasonal_order, 
                               maxiter=config.coarse_maxiter, stage='coarse')
        
        if result is None:
            return 1e10  # 返回很大的值表示失败
        
        results_list.append(result)
        
        # 检查早停
        if early_stop.should_stop(result['AIC']):
            monitor.metrics['early_stops'] += 1
            monitor.log(f'触发早停机制，当前最优AIC: {early_stop.best_score:.2f}')
            return result['AIC']  # 仍返回当前值，但后续迭代会减少
        
        # 定期内存清理
        if len(results_list) % config.gc_frequency == 0:
            gc.collect()
        
        return result['AIC']
    
    # 执行贝叶斯优化
    res = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=42,
        verbose=False
    )
    
    # 转换为DataFrame
    if results_list:
        df = pd.DataFrame(results_list).drop_duplicates()
        df = df.sort_values('AIC').reset_index(drop=True)
        return df
    else:
        return pd.DataFrame(columns=['p','d','q','P','D','Q','s','AIC'])

# ===============================================
# 网格搜索（带早停和内存管理）
# ===============================================
def parallel_search(endog, grid, n_jobs, maxiter, stage_desc, stage_key):
    """并行网格搜索"""
    tasks = [((p, d_, q), (P, D_, Q, s_)) for (p, d_, q, P, D_, Q, s_) in grid]
    monitor.log(f'{stage_desc}: 任务数={len(tasks)}, n_jobs={n_jobs}, maxiter={maxiter}')
    
    early_stop = EarlyStopping(config.early_stop_patience, config.early_stop_threshold)
    results = []
    
    # 分批处理以便早停
    batch_size = max(10, len(tasks) // 10)
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        
        batch_results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_fit_one_stage)(endog, order, seasonal, maxiter, stage=stage_key)
            for order, seasonal in batch
        )
        
        batch_results = [r for r in batch_results if r is not None]
        results.extend(batch_results)
        
        # 检查早停
        if batch_results:
            best_aic = min([r['AIC'] for r in batch_results])
            if early_stop.should_stop(best_aic):
                monitor.metrics['early_stops'] += 1
                monitor.log(f'触发早停机制，当前最优AIC: {early_stop.best_score:.2f}')
                break
        
        # 内存清理
        if (i // batch_size) % config.gc_frequency == 0:
            gc.collect()
    
    if not results:
        return pd.DataFrame(columns=['p','d','q','P','D','Q','s','AIC'])
    
    return pd.DataFrame(results).sort_values('AIC').reset_index(drop=True)

# ===============================================
# 主搜索流程
# ===============================================
monitor.metrics['start_time'] = time.time()

TEST_STEPS = 100
train = data.iloc[:-TEST_STEPS]
N_JOBS = _get_n_jobs()

# 阶段1：智能初筛
monitor.start_stage('智能初筛')
monitor.log('开始智能初筛...')

if config.use_bayesian and BAYESIAN_AVAILABLE:
    coarse_df = bayesian_search(
        train['VALUE'], 
        param_grid, 
        config.bayesian_n_calls,
        config.bayesian_n_initial_points
    )
else:
    coarse_df = parallel_search(
        train['VALUE'], 
        param_grid, 
        N_JOBS, 
        config.coarse_maxiter, 
        '并行拟合(粗筛)', 
        'coarse'
    )

monitor.log(f'初筛完成，评估了 {len(coarse_df)} 个模型')
monitor.log('前 10 个结果:')
print(coarse_df.head(min(10, len(coarse_df))))

if coarse_df.empty:
    raise RuntimeError('粗筛阶段没有可用模型')

monitor.end_stage()

# 阶段2：精调判断
need_fine = True
if len(coarse_df) >= config.top_k:
    gap = coarse_df.iloc[config.top_k-1].AIC - coarse_df.iloc[0].AIC
    monitor.log(f'粗筛第1与第{config.top_k}模型 AIC 差: {gap:.4f}')
    if gap < config.small_gap:
        monitor.log('AIC 差距很小，跳过精调阶段')
        need_fine = False
else:
    monitor.log(f'可用模型不足 TOP_K={config.top_k}，跳过精调')
    need_fine = False

# 阶段3：精调搜索
fine_df = None
if need_fine and config.fine_expand:
    monitor.start_stage('精调搜索')
    
    base_top = coarse_df.head(config.top_k)
    local_grid = _make_local_grid(base_top, max(p_range), max(q_range), max(P_range), max(Q_range))
    coarse_set = set(param_grid)
    local_grid = [g for g in local_grid if g in coarse_set]
    
    monitor.log(f'精调局部网格大小: {len(local_grid)}')
    
    if local_grid:
        monitor.log('开始精调参数搜索...')
        fine_df = parallel_search(
            train['VALUE'], 
            local_grid, 
            N_JOBS, 
            config.fine_maxiter, 
            '并行拟合(精调)', 
            'fine'
        )
        monitor.log('精调完成，前 5 个结果:')
        print(fine_df.head(min(5, len(fine_df))))
    else:
        monitor.log('局部扩展网格为空，跳过精调')
        need_fine = False
    
    monitor.end_stage()

# 选择最优参数
if need_fine and fine_df is not None and not fine_df.empty:
    best_all = pd.concat([coarse_df.head(config.top_k), fine_df], ignore_index=True)
    best_all = best_all.sort_values('AIC').reset_index(drop=True)
    final_row = best_all.iloc[0]
    monitor.log('最终最优参数来自精调合并结果')
else:
    final_row = coarse_df.iloc[0]
    monitor.log('最终最优参数来自粗筛结果')

print(final_row)
final_order = (int(final_row.p), int(final_row.d), int(final_row.q))
final_seasonal = (int(final_row.P), int(final_row.D), int(final_row.Q), int(final_row.s))
monitor.log(f'最终使用 order={final_order}, seasonal_order={final_seasonal}')

# ===============================================
# 最终模型拟合
# ===============================================
monitor.start_stage('最终模型拟合')

final_model = _build_model(data['VALUE'], final_order, final_seasonal, simple_diff=False)
final_res = _fit_robust(final_model, maxiter=max(config.coarse_maxiter, config.fine_maxiter, 500), stage='fine')

if final_res is None and (final_order[1] > 0 or final_seasonal[1] > 0):
    final_model_sd = _build_model(data['VALUE'], final_order, final_seasonal, simple_diff=True)
    final_res = _fit_robust(final_model_sd, maxiter=max(config.coarse_maxiter, config.fine_maxiter, 600), stage='fine')

if final_res is None:
    raise RuntimeError('最终模型多优化器尝试仍未收敛')

print(final_res.summary())
monitor.end_stage()

# ===============================================
# 预测评估
# ===============================================
monitor.start_stage('预测评估')

forecast = final_res.get_forecast(steps=TEST_STEPS)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int(alpha=0.05)
actual_test = data['VALUE'].iloc[-TEST_STEPS:]
mse = mean_squared_error(actual_test, forecast_mean)
monitor.log(f'测试集 {TEST_STEPS} 步预测 MSE: {mse:.4f}')

monitor.end_stage()

# ===============================================
# 可视化（与原版本一致）
# ===============================================
monitor.start_stage('结果可视化')

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
    monitor.log(f'季节分解失败: {e}')

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
    monitor.log(f'残差诊断绘图失败: {e}')

monitor.end_stage()

# ===============================================
# 性能总结
# ===============================================
monitor.metrics['end_time'] = time.time()

print('\n' + '='*60)
print('模型选择与结果小结')
print('='*60)
print(f'最优非季节参数 (p,d,q): {final_order}')
print(f'最优季节参数 (P,D,Q,s): {final_seasonal}')
print(f'最优模型 AIC: {final_row.AIC:.2f}')
print(f'测试集 MSE: {mse:.4f}')
print(f'优化级别: {config.optimization_level} (0=原始, 1=基础, 2=激进)')
print(f'贝叶斯优化: {"启用" if config.use_bayesian else "禁用"}')
print(f'模型缓存: {"启用" if config.enable_cache else "禁用"}')
print(f'并发设置: n_jobs={N_JOBS}')

# 显示性能摘要
print(monitor.summary())

# 保存性能报告
if config.enable_profiling:
    report = {
        'optimization_level': config.optimization_level,
        'total_time': monitor.metrics['end_time'] - monitor.metrics['start_time'],
        'models_evaluated': monitor.metrics['models_evaluated'],
        'cache_hits': monitor.metrics['cache_hits'],
        'cache_misses': monitor.metrics['cache_misses'],
        'early_stops': monitor.metrics['early_stops'],
        'stage_times': monitor.metrics['stage_times'],
        'final_aic': float(final_row.AIC),
        'final_mse': float(mse),
        'final_order': final_order,
        'final_seasonal': final_seasonal,
    }
    
    report_file = 'performance_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    monitor.log(f'性能报告已保存到: {report_file}')

plt.show()
