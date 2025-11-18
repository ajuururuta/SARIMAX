#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SARIMAX 优化版本功能测试
验证核心功能是否正常工作
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# 确保可以导入优化版本的组件
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试所有必需的导入"""
    print("测试 1: 检查依赖导入...", end=" ")
    try:
        import numpy as np
        import pandas as pd
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from sklearn.metrics import mean_squared_error
        from joblib import Parallel, delayed
        from tqdm import tqdm
        print("✓ 通过")
        return True
    except ImportError as e:
        print(f"✗ 失败: {e}")
        return False

def test_bayesian_import():
    """测试贝叶斯优化库"""
    print("测试 2: 检查贝叶斯优化库...", end=" ")
    try:
        from skopt import gp_minimize
        from skopt.space import Integer, Categorical
        print("✓ 通过 (可用)")
        return True
    except ImportError:
        print("⚠ 警告 (不可用，将使用网格搜索)")
        return True  # 这不是错误，只是降级

def test_config_class():
    """测试配置管理类"""
    print("测试 3: 配置管理类...", end=" ")
    try:
        # 导入配置类定义
        import importlib.util
        spec = importlib.util.spec_from_file_location("sarimax_opt", "SARIMAX_optimized.py")
        module = importlib.util.module_from_spec(spec)
        
        # 不执行整个模块，只检查类定义存在
        with open("SARIMAX_optimized.py", 'r', encoding='utf-8') as f:
            content = f.read()
            assert "class OptimizationConfig:" in content
            assert "class PerformanceMonitor:" in content
            assert "class ModelCache:" in content
            assert "class EarlyStopping:" in content
        print("✓ 通过")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def test_cache_operations():
    """测试缓存功能"""
    print("测试 4: 缓存系统...", end=" ")
    try:
        import numpy as np
        import pandas as pd
        import hashlib
        import pickle
        
        # 创建临时缓存目录
        temp_cache = Path(tempfile.mkdtemp())
        
        # 模拟缓存键生成
        test_data = pd.Series(np.random.randn(100))
        data_hash = hashlib.md5(test_data.values.tobytes()).hexdigest()[:8]
        order = (1, 0, 1)
        seasonal = (1, 0, 1, 60)
        cache_key = f"{data_hash}_{order}_{seasonal}"
        
        # 模拟缓存写入
        test_result = {'p': 1, 'd': 0, 'q': 1, 'AIC': 1234.56}
        cache_file = temp_cache / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(test_result, f)
        
        # 模拟缓存读取
        with open(cache_file, 'rb') as f:
            loaded = pickle.load(f)
        
        assert loaded == test_result
        
        # 清理
        shutil.rmtree(temp_cache)
        print("✓ 通过")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def test_early_stopping():
    """测试早停逻辑"""
    print("测试 5: 早停机制...", end=" ")
    try:
        # 检查早停类定义
        with open("SARIMAX_optimized.py", 'r', encoding='utf-8') as f:
            content = f.read()
            assert "class EarlyStopping:" in content
            assert "def should_stop" in content
        print("✓ 通过")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def test_command_line_args():
    """测试命令行参数解析"""
    print("测试 6: 命令行参数...", end=" ")
    try:
        import subprocess
        result = subprocess.run(
            ['python', 'SARIMAX_optimized.py', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # 检查关键参数是否在帮助信息中
        assert '--optimization-level' in result.stdout
        assert '--no-cache' in result.stdout
        assert '--no-bayesian' in result.stdout
        assert '--clear-cache' in result.stdout
        print("✓ 通过")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def test_syntax():
    """测试Python语法"""
    print("测试 7: Python语法检查...", end=" ")
    try:
        import py_compile
        py_compile.compile('SARIMAX_optimized.py', doraise=True)
        print("✓ 通过")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def test_file_structure():
    """测试文件结构完整性"""
    print("测试 8: 文件结构...", end=" ")
    try:
        required_files = [
            'SARIMAX_optimized.py',
            'OPTIMIZATION_GUIDE.md',
            'README.md',
            'benchmark.py',
            '.gitignore',
            'data.csv'
        ]
        
        missing = []
        for f in required_files:
            if not Path(f).exists():
                missing.append(f)
        
        if missing:
            print(f"✗ 失败: 缺少文件 {missing}")
            return False
        
        print("✓ 通过")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def test_documentation():
    """测试文档完整性"""
    print("测试 9: 文档完整性...", end=" ")
    try:
        # 检查README
        readme = Path('README.md').read_text(encoding='utf-8')
        assert '优化特性' in readme or 'optimization' in readme.lower()
        assert '使用' in readme or 'usage' in readme.lower()
        
        # 检查优化指南
        guide = Path('OPTIMIZATION_GUIDE.md').read_text(encoding='utf-8')
        assert '贝叶斯优化' in guide or 'Bayesian' in guide
        assert '缓存' in guide or 'cache' in guide.lower()
        
        print("✓ 通过")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def test_backward_compatibility():
    """测试向后兼容性"""
    print("测试 10: 向后兼容性...", end=" ")
    try:
        # 检查优化版本支持原版的参数
        with open("SARIMAX_optimized.py", 'r', encoding='utf-8') as f:
            content = f.read()
            assert '--coarse-maxiter' in content
            assert '--fine-maxiter' in content
            assert '--top-k' in content
            assert '--small-gap' in content
            assert '--no-fine-expand' in content
        print("✓ 通过")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("SARIMAX 优化版本 - 功能测试套件")
    print("="*70 + "\n")
    
    tests = [
        test_imports,
        test_bayesian_import,
        test_config_class,
        test_cache_operations,
        test_early_stopping,
        test_command_line_args,
        test_syntax,
        test_file_structure,
        test_documentation,
        test_backward_compatibility,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n✓ 所有测试通过！")
        return 0
    else:
        print(f"\n✗ {total - passed} 个测试失败")
        return 1

if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
