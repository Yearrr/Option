#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权回测框架 - 信号生成模块
提供期权交易策略信号生成的框架和基础信号
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class BaseSignalGenerator(ABC):
    """信号生成器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.signals = pd.DataFrame()
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            信号DataFrame
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required_columns)


class VolatilitySignalGenerator(BaseSignalGenerator):
    """基于波动率的信号生成器"""
    
    def __init__(self, lookback_period: int = 21, volatility_threshold: float = 0.25):
        super().__init__("VolatilitySignal")
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        基于历史波动率生成信号
        
        策略逻辑:
        - 当历史波动率低于阈值时，做多波动率策略(如straddle)
        - 当历史波动率高于阈值时，做空波动率策略
        """
        if not self.validate_data(data):
            raise ValueError("数据格式不正确")
        
        signals = pd.DataFrame(index=data.index)
        
        # 计算历史波动率
        returns = data['Close'].pct_change()
        hv = returns.rolling(self.lookback_period).std() * np.sqrt(252)
        
        # 计算波动率分位数
        hv_percentile = hv.rolling(60).rank(pct=True)
        
        # 生成信号
        signals['hv'] = hv
        signals['hv_percentile'] = hv_percentile
        signals['signal'] = 0
        
        # 低波动率做多波动率
        signals.loc[hv_percentile < 0.2, 'signal'] = 1  # 做多straddle
        
        # 高波动率做空波动率
        signals.loc[hv_percentile > 0.8, 'signal'] = -1  # 做空straddle
        
        # 添加策略建议
        signals['strategy'] = ''
        signals.loc[signals['signal'] == 1, 'strategy'] = 'long_straddle'
        signals.loc[signals['signal'] == -1, 'strategy'] = 'short_straddle'
        
        return signals


class MomentumSignalGenerator(BaseSignalGenerator):
    """基于动量的信号生成器"""
    
    def __init__(self, short_window: int = 10, long_window: int = 30):
        super().__init__("MomentumSignal")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        基于价格动量生成方向性策略信号
        
        策略逻辑:
        - 上升趋势：牛市价差策略
        - 下降趋势：熊市价差策略
        - 横盘整理：做多波动率策略
        """
        if not self.validate_data(data):
            raise ValueError("数据格式不正确")
        
        signals = pd.DataFrame(index=data.index)
        
        # 计算移动平均线
        signals['ma_short'] = data['Close'].rolling(self.short_window).mean()
        signals['ma_long'] = data['Close'].rolling(self.long_window).mean()
        
        # 计算动量指标
        signals['momentum'] = (signals['ma_short'] - signals['ma_long']) / signals['ma_long']
        
        # RSI指标
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        signals['rsi'] = 100 - (100 / (1 + rs))
        
        # 生成信号
        signals['signal'] = 0
        signals['strategy'] = ''
        
        # 强势上涨 - 牛市策略
        bull_condition = (signals['momentum'] > 0.02) & (signals['rsi'] < 70)
        signals.loc[bull_condition, 'signal'] = 1
        signals.loc[bull_condition, 'strategy'] = 'bull_call_spread'
        
        # 强势下跌 - 熊市策略
        bear_condition = (signals['momentum'] < -0.02) & (signals['rsi'] > 30)
        signals.loc[bear_condition, 'signal'] = -1
        signals.loc[bear_condition, 'strategy'] = 'bear_put_spread'
        
        # 横盘整理 - 波动率策略
        sideways_condition = (abs(signals['momentum']) <= 0.02) & (signals['rsi'] > 40) & (signals['rsi'] < 60)
        signals.loc[sideways_condition, 'signal'] = 2
        signals.loc[sideways_condition, 'strategy'] = 'iron_condor'
        
        return signals


class SkewSignalGenerator(BaseSignalGenerator):
    """基于偏度的信号生成器"""
    
    def __init__(self, lookback_period: int = 21):
        super().__init__("SkewSignal")
        self.lookback_period = lookback_period
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        基于收益率偏度生成信号
        
        策略逻辑:
        - 负偏度过大：预期向上修复，做多策略
        - 正偏度过大：预期向下修复，做空策略
        """
        if not self.validate_data(data):
            raise ValueError("数据格式不正确")
        
        signals = pd.DataFrame(index=data.index)
        
        # 计算收益率
        returns = data['Close'].pct_change()
        
        # 计算滚动偏度
        signals['skew'] = returns.rolling(self.lookback_period).skew()
        
        # 计算偏度的移动平均和标准差
        signals['skew_ma'] = signals['skew'].rolling(60).mean()
        signals['skew_std'] = signals['skew'].rolling(60).std()
        
        # 标准化偏度
        signals['skew_zscore'] = (signals['skew'] - signals['skew_ma']) / signals['skew_std']
        
        # 生成信号
        signals['signal'] = 0
        signals['strategy'] = ''
        
        # 极度负偏度 - 做多策略
        extreme_negative_skew = signals['skew_zscore'] < -2
        signals.loc[extreme_negative_skew, 'signal'] = 1
        signals.loc[extreme_negative_skew, 'strategy'] = 'long_call'
        
        # 极度正偏度 - 做空策略
        extreme_positive_skew = signals['skew_zscore'] > 2
        signals.loc[extreme_positive_skew, 'signal'] = -1
        signals.loc[extreme_positive_skew, 'strategy'] = 'short_call'
        
        return signals


class VIXSignalGenerator(BaseSignalGenerator):
    """基于VIX的信号生成器"""
    
    def __init__(self, vix_low: float = 15, vix_high: float = 30):
        super().__init__("VIXSignal")
        self.vix_low = vix_low
        self.vix_high = vix_high
    
    def generate_signals(self, data: pd.DataFrame, vix_data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        基于VIX指数生成信号
        
        策略逻辑:
        - VIX低位：做多波动率
        - VIX高位：做空波动率
        """
        signals = pd.DataFrame(index=data.index)
        
        if vix_data is not None:
            # 使用实际VIX数据
            signals['vix'] = vix_data['Close']
        else:
            # 使用模拟VIX (基于历史波动率)
            returns = data['Close'].pct_change()
            hv = returns.rolling(21).std() * np.sqrt(252) * 100
            signals['vix'] = hv
        
        # VIX分位数
        signals['vix_percentile'] = signals['vix'].rolling(252).rank(pct=True)
        
        # 生成信号
        signals['signal'] = 0
        signals['strategy'] = ''
        
        # VIX低位 - 做多波动率
        low_vix = signals['vix_percentile'] < 0.2
        signals.loc[low_vix, 'signal'] = 1
        signals.loc[low_vix, 'strategy'] = 'long_straddle'
        
        # VIX高位 - 做空波动率
        high_vix = signals['vix_percentile'] > 0.8
        signals.loc[high_vix, 'signal'] = -1
        signals.loc[high_vix, 'strategy'] = 'short_straddle'
        
        return signals


class EarningsSignalGenerator(BaseSignalGenerator):
    """基于财报的信号生成器"""
    
    def __init__(self, earnings_window: int = 5):
        super().__init__("EarningsSignal")
        self.earnings_window = earnings_window
    
    def generate_signals(self, data: pd.DataFrame, earnings_dates: List[str] = None, **kwargs) -> pd.DataFrame:
        """
        基于财报发布日期生成信号
        
        策略逻辑:
        - 财报前：做多波动率（预期波动增加）
        - 财报后：做空波动率（波动率回归）
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['strategy'] = ''
        signals['earnings_risk'] = False
        
        if earnings_dates:
            for earnings_date in earnings_dates:
                earnings_dt = pd.to_datetime(earnings_date)
                
                # 财报前几天
                pre_earnings = (signals.index >= (earnings_dt - timedelta(days=self.earnings_window))) & \
                              (signals.index < earnings_dt)
                signals.loc[pre_earnings, 'signal'] = 1
                signals.loc[pre_earnings, 'strategy'] = 'long_straddle'
                signals.loc[pre_earnings, 'earnings_risk'] = True
                
                # 财报后几天
                post_earnings = (signals.index >= earnings_dt) & \
                               (signals.index <= (earnings_dt + timedelta(days=self.earnings_window)))
                signals.loc[post_earnings, 'signal'] = -1
                signals.loc[post_earnings, 'strategy'] = 'short_straddle'
        
        return signals


class CompositeSignalGenerator:
    """组合信号生成器 - 整合多个信号源"""
    
    def __init__(self, generators: List[BaseSignalGenerator], weights: Dict[str, float] = None):
        self.generators = generators
        self.weights = weights or {gen.name: 1.0 for gen in generators}
    
    def generate_composite_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        生成组合信号
        
        Args:
            data: 市场数据
            **kwargs: 传递给各个生成器的参数
            
        Returns:
            组合信号DataFrame
        """
        all_signals = pd.DataFrame(index=data.index)
        all_signals['composite_signal'] = 0
        all_signals['confidence'] = 0
        all_signals['primary_strategy'] = ''
        
        # 收集所有信号
        signal_values = {}
        strategies = {}
        
        for generator in self.generators:
            try:
                signals = generator.generate_signals(data, **kwargs)
                weight = self.weights.get(generator.name, 1.0)
                
                signal_values[generator.name] = signals['signal'] * weight
                strategies[generator.name] = signals['strategy']
                
                # 将各个信号添加到结果中
                all_signals[f'{generator.name}_signal'] = signals['signal']
                all_signals[f'{generator.name}_strategy'] = signals['strategy']
                
            except Exception as e:
                print(f"生成器 {generator.name} 信号生成失败: {e}")
                continue
        
        # 计算组合信号
        if signal_values:
            # 加权平均信号
            weighted_signals = pd.DataFrame(signal_values)
            all_signals['composite_signal'] = weighted_signals.mean(axis=1)
            
            # 计算信号一致性（置信度）
            signal_agreement = (weighted_signals != 0).sum(axis=1)
            all_signals['confidence'] = signal_agreement / len(signal_values)
            
            # 确定主要策略
            for idx in all_signals.index:
                if all_signals.loc[idx, 'composite_signal'] != 0:
                    # 找到信号最强的策略
                    max_signal = 0
                    primary_strategy = ''
                    
                    for gen_name in signal_values:
                        if abs(signal_values[gen_name].loc[idx]) > abs(max_signal):
                            max_signal = signal_values[gen_name].loc[idx]
                            primary_strategy = strategies[gen_name].loc[idx]
                    
                    all_signals.loc[idx, 'primary_strategy'] = primary_strategy
        
        return all_signals
    
    def get_signal_summary(self, signals: pd.DataFrame) -> Dict:
        """获取信号统计摘要"""
        summary = {
            'total_signals': len(signals[signals['composite_signal'] != 0]),
            'long_signals': len(signals[signals['composite_signal'] > 0]),
            'short_signals': len(signals[signals['composite_signal'] < 0]),
            'avg_confidence': signals['confidence'].mean(),
            'strategy_distribution': signals['primary_strategy'].value_counts().to_dict()
        }
        return summary


if __name__ == "__main__":
    # 测试信号生成器
    
    # 创建模拟数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'High': prices * (1 + abs(np.random.normal(0, 0.005, len(dates)))),
        'Low': prices * (1 - abs(np.random.normal(0, 0.005, len(dates)))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # 测试各个信号生成器
    vol_gen = VolatilitySignalGenerator()
    momentum_gen = MomentumSignalGenerator()
    skew_gen = SkewSignalGenerator()
    
    # 生成信号
    vol_signals = vol_gen.generate_signals(data)
    print("波动率信号:")
    print(vol_signals[vol_signals['signal'] != 0].head())
    
    # 测试组合信号
    composite_gen = CompositeSignalGenerator([vol_gen, momentum_gen, skew_gen])
    composite_signals = composite_gen.generate_composite_signals(data)
    
    print("\n组合信号:")
    print(composite_signals[composite_signals['composite_signal'] != 0].head())
    
    # 信号摘要
    summary = composite_gen.get_signal_summary(composite_signals)
    print(f"\n信号摘要: {summary}")