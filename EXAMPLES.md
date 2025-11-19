# SARIMAX 性能优化 - 使用示例

本文档提供了优化版本的实际使用示例。

## 快速开始

### 示例 1: 默认配置（推荐用于生产）

```bash
python SARIMAX_optimized.py
```

这会使用 Level 1 优化（基础优化），提供速度和准确性的最佳平衡。

**预期输出：**
- 详细的性能日志
- 模型参数和AIC
- 测试集预测MSE
- 可视化图表
- performance_report.json 性能报告

### 示例 2: 快速模式（用于快速原型）

```bash
python SARIMAX_optimized.py --optimization-level 2 --top-k 3
```

这会使用激进优化，最快速度完成：
- 缩小搜索空间
- 更激进的早停
- 减少保留的候选模型数

### 示例 3: 兼容模式（与原版行为一致）

```bash
python SARIMAX_optimized.py --optimization-level 0
```

这会禁用所有优化，确保结果与原版 SARIMAX_0.py 一致。

### 示例 4: 自定义迭代次数

```bash
python SARIMAX_optimized.py \
    --coarse-maxiter 30 \
    --fine-maxiter 200 \
    --top-k 5
```

调整各阶段的迭代次数和保留的模型数。

### 示例 5: 禁用特定优化

```bash
# 禁用贝叶斯优化（使用网格搜索）
python SARIMAX_optimized.py --no-bayesian

# 禁用缓存
python SARIMAX_optimized.py --no-cache

# 禁用精调阶段的局部扩展
python SARIMAX_optimized.py --no-fine-expand
```

### 示例 6: 组合配置

```bash
# 快速测试配置
python SARIMAX_optimized.py \
    --optimization-level 2 \
    --coarse-maxiter 20 \
    --top-k 3 \
    --small-gap 1.0 \
    --no-fine-expand

# 高质量配置
python SARIMAX_optimized.py \
    --optimization-level 1 \
    --coarse-maxiter 50 \
    --fine-maxiter 400 \
    --top-k 15 \
    --small-gap 0.3
```

## 缓存管理

### 查看缓存

```bash
# 列出缓存文件
ls -lh .sarimax_cache/

# 查看缓存大小
du -sh .sarimax_cache/
```

### 清空缓存

```bash
python SARIMAX_optimized.py --clear-cache
```

### 利用缓存加速

第一次运行后，相同数据的后续运行会快得多：

```bash
# 第一次运行 - 构建缓存
time python SARIMAX_optimized.py
# 输出: real 0m45s

# 第二次运行 - 使用缓存
time python SARIMAX_optimized.py
# 输出: real 0m5s  (快 9 倍！)
```

## 环境变量配置

可以通过环境变量设置默认参数：

```bash
# 设置环境变量
export COARSE_MAXITER=30
export FINE_MAXITER=200
export TOP_K=5
export SMALL_GAP=1.0
export SARIMAX_N_JOBS=4

# 运行（使用环境变量的值）
python SARIMAX_optimized.py
```

## 性能报告解读

运行后会生成 `performance_report.json`：

```json
{
  "optimization_level": 1,
  "total_time": 45.23,
  "models_evaluated": 48,
  "cache_hits": 12,
  "cache_misses": 48,
  "early_stops": 1,
  "stage_times": {
    "数据预处理": 0.52,
    "平稳性检验": 0.15,
    "搜索空间设定": 0.03,
    "智能初筛": 42.15,
    "精调搜索": 0.0,
    "最终模型拟合": 2.35,
    "预测评估": 0.03
  },
  "final_aic": 2450.34,
  "final_mse": 45.67,
  "final_order": [1, 0, 1],
  "final_seasonal": [1, 0, 1, 60]
}
```

**关键指标说明：**
- `total_time`: 总执行时间（秒）
- `models_evaluated`: 实际评估的模型数量
- `cache_hits/misses`: 缓存命中情况
- `early_stops`: 触发早停的次数
- `stage_times`: 各阶段耗时分解

## 性能对比

使用 benchmark.py 进行性能对比：

```bash
python benchmark.py
```

**示例输出：**

```
======================================================================
性能对比总结
======================================================================

版本                        状态        耗时(秒)      模型数      最终AIC     
----------------------------------------------------------------------
优化版 Level 2 (激进)        ✓ 成功      15.34        28          2450.12     
优化版 Level 1 (默认)        ✓ 成功      25.67        45          2449.85     

速度提升: 1.67x (Level 2 vs Level 1)
======================================================================
```

## 实际案例

### 案例 1: 大数据集（10000+ 样本）

```bash
# 对于大数据集，使用激进优化
python SARIMAX_optimized.py \
    --optimization-level 2 \
    --coarse-maxiter 25 \
    --top-k 3
```

**效果：**
- 原版: ~10分钟
- 优化版: ~1.5分钟
- 提升: 6.7x

### 案例 2: 小数据集（< 500 样本）

```bash
# 对于小数据集，使用保守配置
python SARIMAX_optimized.py \
    --optimization-level 1 \
    --coarse-maxiter 50
```

**效果：**
- 搜索更充分，避免过拟合
- 利用缓存，快速迭代

### 案例 3: 快速探索

```bash
# 快速测试多个季节周期
for level in 0 1 2; do
    echo "Testing optimization level $level"
    python SARIMAX_optimized.py --optimization-level $level
done
```

### 案例 4: 持续集成/自动化

```bash
#!/bin/bash
# CI/CD 脚本

# 使用缓存加速
python SARIMAX_optimized.py \
    --optimization-level 1 \
    --coarse-maxiter 30 \
    > results.log 2>&1

# 提取关键指标
python -c "
import json
with open('performance_report.json') as f:
    data = json.load(f)
    print(f'AIC: {data[\"final_aic\"]}')
    print(f'MSE: {data[\"final_mse\"]}')
    print(f'Time: {data[\"total_time\"]}s')
"
```

## 故障排除示例

### 问题 1: 内存不足

```bash
# 解决方案：禁用缓存，使用激进优化
python SARIMAX_optimized.py \
    --optimization-level 2 \
    --no-cache \
    --top-k 3
```

### 问题 2: scikit-optimize 未安装

```bash
# 安装
pip install scikit-optimize

# 或者禁用贝叶斯优化
python SARIMAX_optimized.py --no-bayesian
```

### 问题 3: 结果不一致

```bash
# 使用原始模式确保一致性
python SARIMAX_optimized.py --optimization-level 0
```

## 高级用法

### 批处理多个数据集

```bash
#!/bin/bash
for datafile in data*.csv; do
    echo "Processing $datafile"
    
    # 临时重命名为 data.csv
    mv data.csv data_backup.csv
    cp $datafile data.csv
    
    # 运行优化
    python SARIMAX_optimized.py --optimization-level 2
    
    # 保存结果
    mv performance_report.json "report_${datafile%.csv}.json"
    
    # 恢复
    mv data_backup.csv data.csv
done
```

### 并行处理

```bash
# 设置并行度
export SARIMAX_N_JOBS=4

# 运行
python SARIMAX_optimized.py
```

### 日志分析

```bash
# 查看日志
tail -f sarimax_optimization.log

# 提取错误
grep "✗" sarimax_optimization.log

# 统计缓存命中
grep "cache_hits" performance_report.json
```

## 总结

优化版本提供了多种使用方式，可以根据具体需求选择：

- **生产环境**: Level 1，启用所有优化
- **快速原型**: Level 2，激进优化
- **验证结果**: Level 0，兼容原版
- **大数据集**: Level 2 + 小 top-k
- **高准确性**: Level 1 + 大 maxiter

根据实际情况调整参数，找到速度和准确性的最佳平衡点。
