#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权回测框架配置文件
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1000000  # 初始资金
    commission_rate: float = 0.001    # 手续费率
    risk_free_rate: float = 0.03      # 无风险利率
    default_volatility: float = 0.2   # 默认波动率
    
    # 风险管理参数
    stop_loss_pct: float = 0.2        # 止损比例
    take_profit_pct: float = 0.5      # 止盈比例
    max_position_size: float = 1000000 # 最大持仓规模
    delta_hedge_threshold: float = 0.1 # Delta对冲阈值
    
    # 对冲参数
    hedge_ratio: float = 1.0          # 对冲比例
    hedge_frequency: str = 'daily'    # 对冲频率
    
    # 输出设置
    export_records: bool = True       # 是否导出记录
    plot_results: bool = True         # 是否绘制结果
    file_prefix: str = 'option_backtest'  # 文件前缀

@dataclass
class SnowballConfig:
    """雪球期权配置"""
    knock_out_barrier: float = 1.03   # 敲出障碍
    knock_in_barrier: float = 0.75    # 敲入障碍
    coupon_rate: float = 0.2          # 票息率（年化）
    observation_frequency: int = 22   # 观察频率（交易日）
    notional: float = 1000000         # 名义本金
    
    # 敲出观察日（相对于开始日期的交易日数）
    observation_dates: list = None
    
    def __post_init__(self):
        if self.observation_dates is None:
            # 默认每月观察一次，持续12个月
            self.observation_dates = [i * self.observation_frequency for i in range(1, 13)]

# 全局配置实例
BACKTEST_CONFIG = BacktestConfig()
SNOWBALL_CONFIG = SnowballConfig()

# 期权类型映射
OPTION_TYPES = {
    'call': 'call',
    'put': 'put', 
    'barrier': 'barrier',
    'snowball': 'snowball',
    'knockout': 'barrier',
    'knockin': 'barrier'
}

# 默认期权参数
DEFAULT_OPTION_PARAMS = {
    'call': {
        'moneyness': 1.0,  # ATM
        'time_to_expiry': 30,  # 30天
        'volatility': 0.2
    },
    'put': {
        'moneyness': 1.0,
        'time_to_expiry': 30,
        'volatility': 0.2
    },
    'barrier': {
        'moneyness': 1.0,
        'time_to_expiry': 30,
        'volatility': 0.2,
        'barrier_up': 1.2,
        'barrier_down': 0.8,
        'rebate': 0.05
    },
    'snowball': {
        'knock_out_barrier': 1.03,
        'knock_in_barrier': 0.75,
        'coupon_rate': 0.2,
        'time_to_expiry': 252  # 1年
    }
}