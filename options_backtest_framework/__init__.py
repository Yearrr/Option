#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权回测框架 - 完整的期权策略回测系统

这是一个专业的期权量化交易回测框架，包含以下核心模块：

- data_manager: 数据管理，支持历史数据获取和缓存
- option_pricing: 期权定价，实现Black-Scholes和Greeks计算
- signal_generator: 信号生成，提供多种技术分析信号
- risk_manager: 风险管理，包含止盈止损和对冲功能
- trading_engine: 交易引擎，负责订单执行和持仓管理
- backtest_framework: 主框架，整合所有模块进行完整回测

主要功能：
1. 历史数据导入和管理
2. 期权定价和Greeks计算
3. 多种期权策略信号生成
4. 风险管理和资金管理
5. 订单执行和持仓跟踪
6. 完整的回测流程和结果分析

支持的期权策略：
- Long/Short Straddle (跨式)
- Long/Short Strangle (宽跨式)
- Bull/Bear Spreads (价差)
- Iron Condor (铁鹰)
- Ratio Spreads (比例价差)
- 以及更多自定义策略

使用示例：
    from options_backtest_framework import OptionsBacktestFramework
    from options_backtest_framework.signal_generator import VolatilitySignalGenerator
    
    # 配置参数
    config = {
        'underlying_symbol': 'AAPL',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_cash': 100000,
        'risk_management': {
            'max_loss_pct': 0.5,
            'take_profit_pct': 1.0
        }
    }
    
    # 创建框架
    framework = OptionsBacktestFramework(config)
    
    # 设置信号生成器
    signal_generators = [VolatilitySignalGenerator()]
    
    # 运行回测
    results = framework.run_backtest(signal_generators)
    
    # 保存和查看结果
    framework.save_results()
    framework.plot_results()

作者: 资深期权量化交易员
版本: 1.0.0
"""

__version__ = '1.0.0'
__author__ = '资深期权量化交易员'

# 导入主要类和函数
from .backtest_framework import OptionsBacktestFramework
from .data_manager import DataManager
from .option_pricing import OptionPricer, StrategyPricer
from .signal_generator import (
    BaseSignalGenerator,
    VolatilitySignalGenerator,
    MomentumSignalGenerator,
    SkewSignalGenerator,
    VIXSignalGenerator,
    EarningsSignalGenerator,
    CompositeSignalGenerator
)
from .risk_manager import RiskManager, PositionSizer
from .trading_engine import TradingEngine, Order, Position

# 定义对外接口
__all__ = [
    # 主框架
    'OptionsBacktestFramework',
    
    # 数据管理
    'DataManager',
    
    # 期权定价
    'OptionPricer',
    'StrategyPricer',
    
    # 信号生成
    'BaseSignalGenerator',
    'VolatilitySignalGenerator',
    'MomentumSignalGenerator',
    'SkewSignalGenerator',
    'VIXSignalGenerator',
    'EarningsSignalGenerator',
    'CompositeSignalGenerator',
    
    # 风险管理
    'RiskManager',
    'PositionSizer',
    
    # 交易引擎
    'TradingEngine',
    'Order',
    'Position',
]

# 版本信息
def get_version():
    """返回框架版本"""
    return __version__

def get_author():
    """返回作者信息"""
    return __author__

# 框架信息
FRAMEWORK_INFO = {
    'name': 'Options Backtest Framework',
    'version': __version__,
    'author': __author__,
    'description': '专业的期权量化交易回测框架',
    'features': [
        '历史数据管理',
        '期权定价和Greeks计算', 
        '多策略信号生成',
        '风险管理和资金管理',
        '完整回测流程',
        '结果分析和可视化'
    ],
    'supported_strategies': [
        'Long/Short Straddle',
        'Long/Short Strangle', 
        'Bull/Bear Spreads',
        'Iron Condor',
        'Ratio Spreads',
        'Custom Strategies'
    ]
}

def print_framework_info():
    """打印框架信息"""
    info = FRAMEWORK_INFO
    print("=" * 60)
    print(f"{info['name']} v{info['version']}")
    print(f"作者: {info['author']}")
    print("=" * 60)
    print(f"描述: {info['description']}")
    print("\n主要功能:")
    for feature in info['features']:
        print(f"  ✓ {feature}")
    print("\n支持的策略:")
    for strategy in info['supported_strategies']:
        print(f"  ✓ {strategy}")
    print("=" * 60)

# 欢迎信息
def welcome():
    """显示欢迎信息"""
    print_framework_info()
    print("\n快速开始:")
    print("1. 配置回测参数")
    print("2. 创建 OptionsBacktestFramework 实例")
    print("3. 设置信号生成器")
    print("4. 运行 run_backtest() 开始回测")
    print("5. 使用 save_results() 和 plot_results() 查看结果")
    print("\n详细文档请参考各模块的 docstring")
    print("=" * 60)

# 模块级别的配置
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 设置默认的 matplotlib 后端（如果需要）
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
except ImportError:
    pass