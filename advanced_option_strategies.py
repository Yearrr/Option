#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级期权策略模块
包含ratio spreads, iron condor, butterfly, strangle等多种策略
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from option_spread_strategy import (
    OptionLeg, SpreadStrategy, black_scholes_price, calculate_greeks,
    SpreadBacktestEngine, create_realistic_market_data
)

class RatioSpreadStrategy:
    """比率价差策略（Ratio Spread）"""
    
    def __init__(self, ratio: int = 2, short_strike_delta: float = 0.3, 
                 long_strike_delta: float = 0.5, days_to_expiry: int = 30):
        """
        初始化比率价差策略
        
        Parameters:
        -----------
        ratio : int
            卖出与买入的比例（例如2表示卖2买1）
        short_strike_delta : float
            卖出期权的delta目标
        long_strike_delta : float
            买入期权的delta目标
        days_to_expiry : int
            期权到期天数
        """
        self.ratio = ratio
        self.short_strike_delta = short_strike_delta
        self.long_strike_delta = long_strike_delta
        self.days_to_expiry = days_to_expiry
    
    def create_call_ratio_spread(self, spot_price: float, current_date: str, iv: float) -> SpreadStrategy:
        """创建看涨比率价差"""
        
        time_to_expiry = self.days_to_expiry / 365.0
        expiry_date = (pd.to_datetime(current_date) + timedelta(days=self.days_to_expiry)).strftime('%Y-%m-%d')
        
        # 找到行权价
        strikes = np.linspace(spot_price * 0.8, spot_price * 1.5, 100)
        
        long_strike = None
        short_strike = None
        min_error_long = float('inf')
        min_error_short = float('inf')
        
        for strike in strikes:
            greeks = calculate_greeks(spot_price, strike, time_to_expiry, 0.03, iv, 'call')
            
            # 寻找买入行权价（通常为ATM或ITM）
            error_long = abs(greeks['delta'] - self.long_strike_delta)
            if error_long < min_error_long:
                min_error_long = error_long
                long_strike = strike
                
            # 寻找卖出行权价（通常为OTM）
            error_short = abs(greeks['delta'] - self.short_strike_delta)
            if error_short < min_error_short:
                min_error_short = error_short
                short_strike = strike
        
        # 计算期权价格
        long_call_price = black_scholes_price(spot_price, long_strike, time_to_expiry, 0.03, iv, 'call')
        short_call_price = black_scholes_price(spot_price, short_strike, time_to_expiry, 0.03, iv, 'call')
        
        # 创建期权腿：买1手ATM，卖2手OTM
        legs = [
            OptionLeg(
                option_type='call',
                strike=long_strike,
                expiry_date=expiry_date,
                position=1,  # 买入1手
                premium=long_call_price,
                iv=iv
            ),
            OptionLeg(
                option_type='call',
                strike=short_strike,
                expiry_date=expiry_date,
                position=-self.ratio,  # 卖出ratio手
                premium=short_call_price,
                iv=iv
            )
        ]
        
        # 计算策略参数
        net_credit = self.ratio * short_call_price - long_call_price
        
        strategy = SpreadStrategy(
            name=f"CallRatioSpread_{current_date}_{self.ratio}x1",
            legs=legs,
            max_profit=net_credit,  # 简化计算
            max_loss=float('inf'),  # 理论上无限损失
            breakeven_points=[short_strike + net_credit]
        )
        
        return strategy

class IronCondorStrategy:
    """铁鹰策略（Iron Condor）"""
    
    def __init__(self, put_spread_width: float = 10, call_spread_width: float = 10,
                 days_to_expiry: int = 30):
        """
        初始化铁鹰策略
        
        Parameters:
        -----------
        put_spread_width : float
            看跌价差的宽度
        call_spread_width : float
            看涨价差的宽度
        days_to_expiry : int
            期权到期天数
        """
        self.put_spread_width = put_spread_width
        self.call_spread_width = call_spread_width
        self.days_to_expiry = days_to_expiry
    
    def create_iron_condor(self, spot_price: float, current_date: str, iv: float) -> SpreadStrategy:
        """创建铁鹰组合"""
        
        time_to_expiry = self.days_to_expiry / 365.0
        expiry_date = (pd.to_datetime(current_date) + timedelta(days=self.days_to_expiry)).strftime('%Y-%m-%d')
        
        # 设定行权价（简化：基于现价的固定百分比）
        put_short_strike = spot_price * 0.95  # 卖出看跌
        put_long_strike = put_short_strike - self.put_spread_width  # 买入看跌
        call_short_strike = spot_price * 1.05  # 卖出看涨
        call_long_strike = call_short_strike + self.call_spread_width  # 买入看涨
        
        # 计算期权价格
        put_short_price = black_scholes_price(spot_price, put_short_strike, time_to_expiry, 0.03, iv, 'put')
        put_long_price = black_scholes_price(spot_price, put_long_strike, time_to_expiry, 0.03, iv, 'put')
        call_short_price = black_scholes_price(spot_price, call_short_strike, time_to_expiry, 0.03, iv, 'call')
        call_long_price = black_scholes_price(spot_price, call_long_strike, time_to_expiry, 0.03, iv, 'call')
        
        # 创建四腿期权组合
        legs = [
            # 看跌价差（卖低买高）
            OptionLeg('put', put_short_strike, expiry_date, -1, put_short_price, iv),
            OptionLeg('put', put_long_strike, expiry_date, 1, put_long_price, iv),
            # 看涨价差（卖低买高）
            OptionLeg('call', call_short_strike, expiry_date, -1, call_short_price, iv),
            OptionLeg('call', call_long_strike, expiry_date, 1, call_long_price, iv)
        ]
        
        # 计算策略参数
        net_credit = (put_short_price - put_long_price) + (call_short_price - call_long_price)
        max_profit = net_credit
        max_loss = max(self.put_spread_width, self.call_spread_width) - net_credit
        
        strategy = SpreadStrategy(
            name=f"IronCondor_{current_date}",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[put_short_strike - net_credit, call_short_strike + net_credit]
        )
        
        return strategy

class ButterflyStrategy:
    """蝶式策略（Butterfly Spread）"""
    
    def __init__(self, strike_spacing: float = 10, days_to_expiry: int = 30,
                 option_type: str = 'call'):
        """
        初始化蝶式策略
        
        Parameters:
        -----------
        strike_spacing : float
            行权价间距
        days_to_expiry : int
            期权到期天数
        option_type : str
            期权类型（'call' 或 'put'）
        """
        self.strike_spacing = strike_spacing
        self.days_to_expiry = days_to_expiry
        self.option_type = option_type
    
    def create_butterfly(self, spot_price: float, current_date: str, iv: float) -> SpreadStrategy:
        """创建蝶式组合"""
        
        time_to_expiry = self.days_to_expiry / 365.0
        expiry_date = (pd.to_datetime(current_date) + timedelta(days=self.days_to_expiry)).strftime('%Y-%m-%d')
        
        # 设定三个行权价
        middle_strike = spot_price  # 中间行权价为ATM
        lower_strike = middle_strike - self.strike_spacing
        upper_strike = middle_strike + self.strike_spacing
        
        # 计算期权价格
        lower_price = black_scholes_price(spot_price, lower_strike, time_to_expiry, 0.03, iv, self.option_type)
        middle_price = black_scholes_price(spot_price, middle_strike, time_to_expiry, 0.03, iv, self.option_type)
        upper_price = black_scholes_price(spot_price, upper_strike, time_to_expiry, 0.03, iv, self.option_type)
        
        # 创建蝶式组合：买1手低行权价，卖2手中行权价，买1手高行权价
        legs = [
            OptionLeg(self.option_type, lower_strike, expiry_date, 1, lower_price, iv),
            OptionLeg(self.option_type, middle_strike, expiry_date, -2, middle_price, iv),
            OptionLeg(self.option_type, upper_strike, expiry_date, 1, upper_price, iv)
        ]
        
        # 计算策略参数
        net_debit = lower_price - 2 * middle_price + upper_price
        max_profit = self.strike_spacing - abs(net_debit)
        max_loss = abs(net_debit)
        
        strategy = SpreadStrategy(
            name=f"{self.option_type.title()}Butterfly_{current_date}",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[lower_strike + abs(net_debit), upper_strike - abs(net_debit)]
        )
        
        return strategy

class StrangleStrategy:
    """宽跨式策略（Strangle）"""
    
    def __init__(self, put_delta: float = 0.25, call_delta: float = 0.25,
                 days_to_expiry: int = 30, is_short: bool = True):
        """
        初始化宽跨式策略
        
        Parameters:
        -----------
        put_delta : float
            看跌期权的delta目标
        call_delta : float
            看涨期权的delta目标
        days_to_expiry : int
            期权到期天数
        is_short : bool
            是否为卖出宽跨式
        """
        self.put_delta = put_delta
        self.call_delta = call_delta
        self.days_to_expiry = days_to_expiry
        self.is_short = is_short
    
    def create_strangle(self, spot_price: float, current_date: str, iv: float) -> SpreadStrategy:
        """创建宽跨式组合"""
        
        time_to_expiry = self.days_to_expiry / 365.0
        expiry_date = (pd.to_datetime(current_date) + timedelta(days=self.days_to_expiry)).strftime('%Y-%m-%d')
        
        # 找到目标delta对应的行权价
        strikes = np.linspace(spot_price * 0.7, spot_price * 1.3, 100)
        
        put_strike = None
        call_strike = None
        min_error_put = float('inf')
        min_error_call = float('inf')
        
        for strike in strikes:
            put_greeks = calculate_greeks(spot_price, strike, time_to_expiry, 0.03, iv, 'put')
            call_greeks = calculate_greeks(spot_price, strike, time_to_expiry, 0.03, iv, 'call')
            
            # 寻找看跌期权行权价
            error_put = abs(abs(put_greeks['delta']) - self.put_delta)
            if error_put < min_error_put:
                min_error_put = error_put
                put_strike = strike
                
            # 寻找看涨期权行权价
            error_call = abs(call_greeks['delta'] - self.call_delta)
            if error_call < min_error_call:
                min_error_call = error_call
                call_strike = strike
        
        # 计算期权价格
        put_price = black_scholes_price(spot_price, put_strike, time_to_expiry, 0.03, iv, 'put')
        call_price = black_scholes_price(spot_price, call_strike, time_to_expiry, 0.03, iv, 'call')
        
        # 创建宽跨式组合
        position = -1 if self.is_short else 1
        legs = [
            OptionLeg('put', put_strike, expiry_date, position, put_price, iv),
            OptionLeg('call', call_strike, expiry_date, position, call_price, iv)
        ]
        
        # 计算策略参数
        net_premium = put_price + call_price
        if self.is_short:
            max_profit = net_premium
            max_loss = float('inf')  # 理论无限
        else:
            max_profit = float('inf')
            max_loss = net_premium
        
        strategy_name = f"{'Short' if self.is_short else 'Long'}Strangle_{current_date}"
        
        strategy = SpreadStrategy(
            name=strategy_name,
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[put_strike - net_premium, call_strike + net_premium]
        )
        
        return strategy

class MultiStrategyBacktest(SpreadBacktestEngine):
    """多策略期权回测引擎"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_factories = {
            'sell_call_spread': self._create_sell_call_spread,
            'call_ratio_spread': self._create_call_ratio_spread,
            'iron_condor': self._create_iron_condor,
            'butterfly': self._create_butterfly,
            'short_strangle': self._create_short_strangle
        }
        
    def _create_sell_call_spread(self, spot_price: float, date: str, iv: float) -> SpreadStrategy:
        """创建卖出看涨价差"""
        from option_spread_strategy import SellCallSpreadStrategy
        strategy = SellCallSpreadStrategy()
        return strategy.create_spread(spot_price, date, iv)
    
    def _create_call_ratio_spread(self, spot_price: float, date: str, iv: float) -> SpreadStrategy:
        """创建看涨比率价差"""
        strategy = RatioSpreadStrategy(ratio=2)
        return strategy.create_call_ratio_spread(spot_price, date, iv)
    
    def _create_iron_condor(self, spot_price: float, date: str, iv: float) -> SpreadStrategy:
        """创建铁鹰策略"""
        strategy = IronCondorStrategy()
        return strategy.create_iron_condor(spot_price, date, iv)
    
    def _create_butterfly(self, spot_price: float, date: str, iv: float) -> SpreadStrategy:
        """创建蝶式策略"""
        strategy = ButterflyStrategy()
        return strategy.create_butterfly(spot_price, date, iv)
    
    def _create_short_strangle(self, spot_price: float, date: str, iv: float) -> SpreadStrategy:
        """创建卖出宽跨式"""
        strategy = StrangleStrategy(is_short=True)
        return strategy.create_strangle(spot_price, date, iv)
    
    def run_multi_strategy_backtest(self, strategy_configs: Dict[str, Dict]):
        """
        运行多策略回测
        
        Parameters:
        -----------
        strategy_configs : Dict[str, Dict]
            策略配置字典，格式如：
            {
                'sell_call_spread': {'frequency': 10, 'max_positions': 5, 'iv_threshold': 0.25},
                'iron_condor': {'frequency': 15, 'max_positions': 3, 'iv_threshold': 0.3}
            }
        """
        
        print("开始多策略期权回测...")
        print("="*60)
        
        strategy_counters = {name: 0 for name in strategy_configs.keys()}
        
        for i, (date, row) in enumerate(self.market_data.iterrows()):
            date_str = date.strftime('%Y-%m-%d')
            
            if i < 30:  # 确保有足够历史数据
                continue
                
            spot_price = row['close']
            iv = row['iv']
            
            # 为每个策略检查开仓信号
            for strategy_name, config in strategy_configs.items():
                if strategy_name not in self.strategy_factories:
                    continue
                    
                frequency = config.get('frequency', 15)
                max_positions = config.get('max_positions', 5)
                iv_threshold = config.get('iv_threshold', 0.2)
                contracts = config.get('contracts_per_trade', 1)
                
                # 检查开仓条件
                if (i % frequency == 0 and 
                    len([p for p in self.positions if strategy_name in p['strategy'].name]) < max_positions and
                    iv >= iv_threshold):
                    
                    try:
                        strategy = self.strategy_factories[strategy_name](spot_price, date_str, iv)
                        self.open_spread_position(strategy, date_str, contracts)
                        strategy_counters[strategy_name] += 1
                    except Exception as e:
                        print(f"创建{strategy_name}失败 {date_str}: {e}")
            
            # 更新所有持仓
            self.update_positions(date_str)
        
        # 计算每日收益率
        self._calculate_daily_returns()
        
        print("多策略回测完成！")
        for name, count in strategy_counters.items():
            print(f"{name}: {count} 个头寸")
    
    def analyze_strategy_performance(self) -> Dict[str, Dict]:
        """分析各策略的绩效"""
        
        df_trades = pd.DataFrame([t for t in self.trade_history if t['action'] == 'close'])
        
        if df_trades.empty:
            return {}
        
        strategy_analysis = {}
        
        # 按策略名称分组分析
        strategy_names = df_trades['strategy_name'].str.extract(r'([A-Za-z]+)')[0].unique()
        
        for strategy_type in strategy_names:
            strategy_trades = df_trades[df_trades['strategy_name'].str.contains(strategy_type)]
            
            if len(strategy_trades) == 0:
                continue
                
            win_trades = strategy_trades[strategy_trades['net_pnl'] > 0]
            
            analysis = {
                'total_trades': len(strategy_trades),
                'win_rate': len(win_trades) / len(strategy_trades),
                'avg_pnl': strategy_trades['net_pnl'].mean(),
                'total_pnl': strategy_trades['net_pnl'].sum(),
                'avg_days_held': strategy_trades['days_held'].mean(),
                'best_trade': strategy_trades['net_pnl'].max(),
                'worst_trade': strategy_trades['net_pnl'].min(),
                'std_pnl': strategy_trades['net_pnl'].std()
            }
            
            strategy_analysis[strategy_type] = analysis
        
        return strategy_analysis
    
    def plot_strategy_comparison(self):
        """绘制策略对比图"""
        
        analysis = self.analyze_strategy_performance()
        
        if not analysis:
            print("没有可分析的策略数据")
            return
        
        strategies = list(analysis.keys())
        metrics = ['win_rate', 'avg_pnl', 'total_pnl', 'avg_days_held']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [analysis[s][metric] for s in strategies]
            
            bars = axes[i].bar(strategies, values, alpha=0.7)
            axes[i].set_title(f'{metric} 对比')
            axes[i].set_ylabel(metric)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', va='bottom')
            
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def run_comprehensive_strategy_test():
    """运行综合策略测试"""
    
    print("期权策略组合回测演示")
    print("="*70)
    
    # 创建市场数据
    market_data = create_realistic_market_data('2022-01-01', '2024-12-31')
    
    # 创建多策略回测引擎
    engine = MultiStrategyBacktest(initial_capital=200000, commission_per_contract=2.0)
    engine.add_market_data(market_data)
    
    # 配置多个策略
    strategy_configs = {
        'sell_call_spread': {
            'frequency': 12,  # 每12天评估一次
            'max_positions': 4,
            'iv_threshold': 0.22,
            'contracts_per_trade': 2
        },
        'iron_condor': {
            'frequency': 20,  # 每20天评估一次
            'max_positions': 3,
            'iv_threshold': 0.25,
            'contracts_per_trade': 1
        },
        'short_strangle': {
            'frequency': 25,  # 每25天评估一次
            'max_positions': 2,
            'iv_threshold': 0.3,
            'contracts_per_trade': 1
        },
        'butterfly': {
            'frequency': 15,  # 每15天评估一次
            'max_positions': 3,
            'iv_threshold': 0.2,
            'contracts_per_trade': 1
        }
    }
    
    # 运行多策略回测
    engine.run_multi_strategy_backtest(strategy_configs)
    
    # 获取整体绩效
    overall_summary = engine.get_performance_summary()
    print("\n整体绩效摘要:")
    print("-" * 50)
    for key, value in overall_summary.items():
        if isinstance(value, float):
            if 'rate' in key or 'return' in key or 'ratio' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # 分析各策略绩效
    strategy_analysis = engine.analyze_strategy_performance()
    print("\n各策略绩效分析:")
    print("-" * 50)
    for strategy, metrics in strategy_analysis.items():
        print(f"\n{strategy}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                if 'rate' in metric:
                    print(f"  {metric}: {value:.2%}")
                else:
                    print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")
    
    # 绘制结果
    engine.plot_results()
    engine.plot_strategy_comparison()
    
    # 导出结果
    engine.export_results('multi_strategy_backtest')
    
    return engine, overall_summary, strategy_analysis

if __name__ == "__main__":
    # 运行综合测试
    engine, summary, analysis = run_comprehensive_strategy_test()
    
    print("\n策略组合分析总结:")
    print("="*70)
    print("✓ 卖出看涨价差：在横盘或小幅上涨市场中表现良好")
    print("✓ 铁鹰策略：适合低波动率环境，收益稳定但有限")
    print("✓ 卖出宽跨式：在低波动率时收益较高，但风险较大")
    print("✓ 蝶式策略：适合预期标的价格在特定区间内波动")
    print("\n优化建议:")
    print("- 根据VIX水平动态调整策略权重")
    print("- 结合技术分析选择开仓时机")
    print("- 建立动态止损机制")
    print("- 考虑期权流动性和买卖价差影响")