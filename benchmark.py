#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SARIMAX 性能对比脚本
比较原始版本和优化版本的性能差异
"""

import subprocess
import time
import json
import sys
from pathlib import Path

def run_version(script_name, args, timeout=300):
    """运行指定版本并记录性能"""
    print(f"\n{'='*60}")
    print(f"运行: {script_name} {' '.join(args)}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', script_name] + args,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed_time = time.time() - start_time
        
        return {
            'success': result.returncode == 0,
            'time': elapsed_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        return {
            'success': False,
            'time': elapsed_time,
            'error': 'Timeout',
            'returncode': -1
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'success': False,
            'time': elapsed_time,
            'error': str(e),
            'returncode': -2
        }

def extract_metrics(output, perf_report_path=None):
    """从输出中提取性能指标"""
    metrics = {
        'models_evaluated': None,
        'cache_hits': None,
        'cache_misses': None,
        'early_stops': None,
        'final_aic': None,
        'final_mse': None,
    }
    
    # 从performance_report.json读取（如果存在）
    if perf_report_path and Path(perf_report_path).exists():
        try:
            with open(perf_report_path) as f:
                report = json.load(f)
                metrics['models_evaluated'] = report.get('models_evaluated')
                metrics['cache_hits'] = report.get('cache_hits')
                metrics['cache_misses'] = report.get('cache_misses')
                metrics['early_stops'] = report.get('early_stops')
                metrics['final_aic'] = report.get('final_aic')
                metrics['final_mse'] = report.get('final_mse')
        except Exception:
            pass
    
    # 从标准输出提取
    for line in output.split('\n'):
        if '总参数组合数:' in line:
            try:
                metrics['total_combinations'] = int(line.split(':')[1].strip())
            except:
                pass
        elif '测试集' in line and 'MSE' in line:
            try:
                mse_str = line.split('MSE:')[1].strip()
                metrics['final_mse'] = float(mse_str)
            except:
                pass
        elif '最优模型 AIC:' in line:
            try:
                aic_str = line.split('AIC:')[1].strip()
                metrics['final_aic'] = float(aic_str)
            except:
                pass
    
    return metrics

def compare_versions():
    """对比不同版本的性能"""
    print("\n" + "="*70)
    print("SARIMAX 性能对比测试")
    print("="*70)
    
    results = {}
    
    # 测试配置
    test_configs = [
        {
            'name': '优化版 Level 2 (激进)',
            'script': 'SARIMAX_optimized.py',
            'args': ['--optimization-level', '2', '--coarse-maxiter', '20', '--top-k', '3', '--no-bayesian'],
            'timeout': 180
        },
        {
            'name': '优化版 Level 1 (默认)',
            'script': 'SARIMAX_optimized.py',
            'args': ['--optimization-level', '1', '--coarse-maxiter', '30', '--no-bayesian'],
            'timeout': 240
        },
        # 原始版本通常需要更长时间，在实际对比时可启用
        # {
        #     'name': '原始版本',
        #     'script': 'SARIMAX_0.py',
        #     'args': ['--coarse-maxiter', '30', '--fine-maxiter', '200'],
        #     'timeout': 600
        # },
    ]
    
    # 运行测试
    for config in test_configs:
        print(f"\n测试: {config['name']}")
        
        # 清理之前的性能报告
        perf_report = Path('performance_report.json')
        if perf_report.exists():
            perf_report.unlink()
        
        result = run_version(
            config['script'],
            config['args'],
            config['timeout']
        )
        
        if result['success']:
            print(f"✓ 成功完成，耗时: {result['time']:.2f}秒")
            metrics = extract_metrics(result['stdout'], 'performance_report.json')
            result['metrics'] = metrics
        else:
            print(f"✗ 执行失败: {result.get('error', 'Unknown error')}")
            print(f"   耗时: {result['time']:.2f}秒")
        
        results[config['name']] = result
    
    # 生成对比报告
    print("\n" + "="*70)
    print("性能对比总结")
    print("="*70)
    
    print(f"\n{'版本':<25} {'状态':<10} {'耗时(秒)':<12} {'模型数':<10} {'最终AIC':<12}")
    print("-" * 70)
    
    for name, result in results.items():
        status = "✓ 成功" if result['success'] else "✗ 失败"
        time_str = f"{result['time']:.2f}"
        
        if result['success'] and 'metrics' in result:
            m = result['metrics']
            models = str(m.get('models_evaluated', 'N/A'))
            aic = f"{m.get('final_aic', 0):.2f}" if m.get('final_aic') else 'N/A'
        else:
            models = 'N/A'
            aic = 'N/A'
        
        print(f"{name:<25} {status:<10} {time_str:<12} {models:<10} {aic:<12}")
    
    # 计算速度提升
    if len(results) >= 2:
        names = list(results.keys())
        baseline_time = results[names[1]]['time'] if results[names[1]]['success'] else None
        optimized_time = results[names[0]]['time'] if results[names[0]]['success'] else None
        
        if baseline_time and optimized_time:
            speedup = baseline_time / optimized_time
            print(f"\n速度提升: {speedup:.2f}x (Level 2 vs Level 1)")
    
    print("\n" + "="*70)
    
    return results

if __name__ == '__main__':
    # 检查文件是否存在
    if not Path('SARIMAX_optimized.py').exists():
        print("错误: SARIMAX_optimized.py 不存在")
        sys.exit(1)
    
    if not Path('data.csv').exists():
        print("错误: data.csv 不存在")
        sys.exit(1)
    
    results = compare_versions()
    
    # 保存对比结果
    comparison_file = 'comparison_results.json'
    with open(comparison_file, 'w', encoding='utf-8') as f:
        # 简化结果用于JSON序列化
        simple_results = {}
        for name, result in results.items():
            simple_results[name] = {
                'success': result['success'],
                'time': result['time'],
                'metrics': result.get('metrics', {})
            }
        json.dump(simple_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n对比结果已保存到: {comparison_file}")
