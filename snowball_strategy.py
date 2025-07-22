#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雪球期权策略和高级回测示例
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from option_backtest_framework import (
    OptionBacktestEngine, OptionContract, TradeRecord, PositionRecord,
    OptionPricer, RiskManager, HedgeEngine
)
from config import BACKTEST_CONFIG, SNOWBALL_CONFIG, DEFAULT_OPTION_PARAMS

class SnowballPricer(OptionPricer):
    """雪球期权定价器"""
    
    @staticmethod
    def snowball_mc(S: float, knock_out_barrier: float, knock_in_barrier: float,
                   coupon_rate: float, T: float, r: float, sigma: float,
                   observation_dates: List[int], notional: float = 1000000,
                   n_simulations: int = 10000) -> Tuple[float, Dict[str, float]]:
        """
        蒙特卡洛方法定价雪球期权
        
        Parameters:
        -----------
        S : float
            当前股价
        knock_out_barrier : float  
            敲出障碍价格
        knock_in_barrier : float
            敲入障碍价格
        coupon_rate : float
            票息率（年化）
        T : int
            总天数
        r : float
            无风险利率
        sigma : float
            波动率
        observation_dates : List[int]
            观察日期列表（交易日）
        notional : float
            名义本金
        n_simulations : int
            模拟次数
        
        Returns:
        --------
        Tuple[float, Dict[str, float]]
            期权价格和希腊字母
        """
        dt = 1/252  # 每日时间步长
        payoffs = []
        
        # 将敲出和敲入障碍转换为绝对价格
        knockout_price = S * knock_out_barrier
        knockin_price = S * knock_in_barrier
        
        for sim in range(n_simulations):
            # 生成股价路径
            path = [S]
            Z = np.random.standard_normal(T)
            
            for t in range(1, T + 1):
                St = path[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[t-1])
                path.append(St)
            
            path = np.array(path)
            
            # 检查敲出和敲入
            knocked_out = False
            knocked_in = False
            knockout_date = None
            knockin_date = None
            
            # 检查敲入（任何时间点）
            knockin_indices = np.where(path <= knockin_price)[0]
            if len(knockin_indices) > 0:
                knocked_in = True
                knockin_date = knockin_indices[0]
            
            # 检查敲出（仅在观察日）
            for obs_date in observation_dates:
                if obs_date < len(path) and path[obs_date] >= knockout_price:
                    knocked_out = True
                    knockout_date = obs_date
                    break
            
            # 计算收益
            if knocked_out and (not knocked_in or knockout_date <= knockin_date):
                # 敲出且未敲入，或敲出在敲入之前
                payoff = coupon_rate * knockout_date / 252  # 按实际持有时间计算票息
                discount_factor = np.exp(-r * knockout_date / 252)
            elif not knocked_out and not knocked_in:
                # 未敲出未敲入，获得全额票息
                payoff = coupon_rate
                discount_factor = np.exp(-r * T / 252)
            else:
                # 敲入且未敲出，承担下跌损失
                final_return = path[-1] / S - 1  # 最终收益率
                payoff = min(final_return, 0)  # 只承担负收益
                discount_factor = np.exp(-r * T / 252)
            
            payoffs.append(payoff * discount_factor * notional)
        
        option_price = np.mean(payoffs)
        
        # 简化的希腊字母计算
        greeks = {
            'delta': -0.3,  # 雪球期权通常有负delta
            'gamma': 0.01,
            'theta': -option_price / (T / 252) / 365,
            'vega': option_price * 0.1
        }
        
        return option_price, greeks

class SnowballStrategy:
    """雪球期权策略"""
    
    def __init__(self, config: dict = None):
        self.config = config or SNOWBALL_CONFIG.__dict__
        self.positions = []
        
    def generate_signals(self, market_data: pd.DataFrame, 
                        signal_frequency: int = 66) -> pd.DataFrame:  # 每季度一个信号
        """
        生成雪球期权交易信号
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            市场数据
        signal_frequency : int
            信号频率（交易日）
            
        Returns:
        --------
        pd.DataFrame
            交易信号
        """
        signals = []
        dates = market_data.index
        
        for i in range(0, len(dates), signal_frequency):
            if i + 252 > len(dates):  # 确保有足够的到期时间
                break
                
            current_date = dates[i]
            current_price = market_data.loc[current_date, 'close']
            expiry_date = dates[min(i + 252, len(dates) - 1)]
            
            # 计算最近30天的波动率
            if i >= 30:
                recent_returns = market_data['close'].iloc[i-30:i].pct_change().dropna()
                realized_vol = recent_returns.std() * np.sqrt(252)
            else:
                realized_vol = 0.2  # 默认波动率
            
            # 根据波动率和趋势调整参数
            vol_adj = min(max(realized_vol / 0.2, 0.5), 2.0)  # 波动率调整因子
            
            # 动态调整敲出障碍
            knock_out_barrier = self.config['knock_out_barrier'] * vol_adj
            knock_in_barrier = self.config['knock_in_barrier'] / vol_adj
            
            signals.append({
                'date': current_date,
                'action': 'buy',
                'symbol': f'SNOWBALL_{current_date.strftime("%Y%m%d")}',
                'option_type': 'snowball',
                'strike': current_price,
                'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                'knock_out_barrier': knock_out_barrier,
                'knock_in_barrier': knock_in_barrier,
                'coupon_rate': self.config['coupon_rate'],
                'notional': self.config['notional'],
                'quantity': 1,
                'option_price': None,  # 将在回测中计算
                'volatility': realized_vol
            })
        
        df = pd.DataFrame(signals)
        if not df.empty:
            df = df.set_index('date')
        return df

class AdvancedOptionBacktest(OptionBacktestEngine):
    """增强型期权回测引擎"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snowball_pricer = SnowballPricer()
        
    def calculate_option_value(self, contract: OptionContract, current_price: float,
                             current_date: str, risk_free_rate: float = 0.03,
                             volatility: float = 0.2) -> Tuple[float, Dict[str, float]]:
        """增强的期权估值方法，支持雪球期权"""
        
        # 计算到期时间
        expiry = pd.to_datetime(contract.expiry_date)
        current = pd.to_datetime(current_date)
        days_to_expiry = (expiry - current).days
        
        if days_to_expiry <= 0:
            # 期权已到期
            return 0, {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        if contract.option_type == 'snowball':
            # 使用雪球期权定价模型
            observation_dates = SNOWBALL_CONFIG.observation_dates
            # 调整观察日期到剩余时间
            remaining_obs_dates = [d for d in observation_dates if d <= days_to_expiry]
            
            if remaining_obs_dates:
                price, greeks = self.snowball_pricer.snowball_mc(
                    S=current_price,
                    knock_out_barrier=contract.barrier_up or SNOWBALL_CONFIG.knock_out_barrier,
                    knock_in_barrier=contract.barrier_down or SNOWBALL_CONFIG.knock_in_barrier,
                    coupon_rate=contract.rebate or SNOWBALL_CONFIG.coupon_rate,
                    T=days_to_expiry,
                    r=risk_free_rate,
                    sigma=volatility,
                    observation_dates=remaining_obs_dates,
                    notional=contract.notional,
                    n_simulations=5000  # 减少模拟次数以提高速度
                )
                return price / contract.notional, greeks  # 返回单位价格
            else:
                return 0, {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        else:
            # 使用父类方法处理其他期权类型
            return super().calculate_option_value(contract, current_price, current_date, 
                                                risk_free_rate, volatility)
    
    def run_snowball_backtest(self, market_data: pd.DataFrame, 
                            strategy: SnowballStrategy = None):
        """运行雪球期权回测"""
        
        if strategy is None:
            strategy = SnowballStrategy()
        
        # 生成策略信号
        signals = strategy.generate_signals(market_data)
        
        print(f"生成了 {len(signals)} 个雪球期权交易信号")
        
        # 预计算期权价格
        for date, signal in signals.iterrows():
            current_price = market_data.loc[date, 'close']
            
            # 创建临时合约用于定价
            temp_contract = OptionContract(
                symbol=signal['symbol'],
                option_type='snowball',
                strike=signal['strike'],
                expiry_date=signal['expiry_date'],
                barrier_up=signal['knock_out_barrier'],
                barrier_down=signal['knock_in_barrier'],
                rebate=signal['coupon_rate'],
                notional=signal['notional']
            )
            
            option_price, _ = self.calculate_option_value(
                temp_contract, current_price, date.strftime('%Y-%m-%d'), 
                volatility=signal['volatility']
            )
            
            signals.loc[date, 'option_price'] = option_price
        
        # 运行回测
        self.run_backtest(signals)
    
    def get_detailed_analysis(self) -> Dict:
        """获取详细分析报告"""
        basic_summary = self.get_performance_summary()
        
        # 分析交易记录
        trades_df = pd.DataFrame([{
            'date': t.date,
            'symbol': t.symbol, 
            'action': t.action,
            'quantity': t.quantity,
            'price': t.price,
            'trade_type': t.trade_type,
            'pnl': t.pnl,
            'commission': t.commission
        } for t in self.trade_records])
        
        # 分析持仓记录
        positions_df = pd.DataFrame([{
            'date': p.date,
            'symbol': p.symbol,
            'quantity': p.quantity,
            'market_value': p.market_value,
            'unrealized_pnl': p.unrealized_pnl,
            'delta': p.delta
        } for p in self.position_records])
        
        detailed_analysis = {
            **basic_summary,
            'trade_analysis': self._analyze_trades(trades_df),
            'position_analysis': self._analyze_positions(positions_df),
            'risk_metrics': self._calculate_risk_metrics()
        }
        
        return detailed_analysis
    
    def _analyze_trades(self, trades_df: pd.DataFrame) -> Dict:
        """分析交易记录"""
        if trades_df.empty:
            return {}
        
        return {
            'total_trades': len(trades_df),
            'avg_trade_pnl': trades_df['pnl'].mean(),
            'trade_pnl_std': trades_df['pnl'].std(),
            'profitable_trades': len(trades_df[trades_df['pnl'] > 0]),
            'losing_trades': len(trades_df[trades_df['pnl'] < 0]),
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),
            'total_commission': trades_df['commission'].sum()
        }
    
    def _analyze_positions(self, positions_df: pd.DataFrame) -> Dict:
        """分析持仓记录"""
        if positions_df.empty:
            return {}
        
        positions_df['date'] = pd.to_datetime(positions_df['date'])
        daily_exposure = positions_df.groupby('date')['market_value'].sum()
        
        return {
            'max_exposure': daily_exposure.max(),
            'avg_exposure': daily_exposure.mean(),
            'exposure_std': daily_exposure.std(),
            'max_positions': positions_df.groupby('date').size().max(),
            'avg_positions': positions_df.groupby('date').size().mean()
        }
    
    def _calculate_risk_metrics(self) -> Dict:
        """计算风险指标"""
        if not self.daily_pnl:
            return {}
        
        df_pnl = pd.DataFrame(self.daily_pnl)
        df_pnl['date'] = pd.to_datetime(df_pnl['date'])
        
        returns = df_pnl['daily_return'].dropna()
        
        if len(returns) == 0:
            return {}
        
        # VaR计算
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # 最大连续亏损天数
        losing_streaks = []
        current_streak = 0
        for ret in returns:
            if ret < 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    losing_streaks.append(current_streak)
                current_streak = 0
        
        max_losing_streak = max(losing_streaks) if losing_streaks else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'max_losing_streak': max_losing_streak,
            'downside_deviation': returns[returns < 0].std(),
            'calmar_ratio': returns.mean() * 252 / abs(self._calculate_max_drawdown(df_pnl)) if self._calculate_max_drawdown(df_pnl) != 0 else 0
        }

def run_comprehensive_backtest():
    """运行综合回测示例"""
    print("开始综合期权回测...")
    print("="*60)
    
    # 创建更真实的市场数据
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    
    # 生成具有趋势和波动聚集的股价数据
    np.random.seed(42)
    returns = []
    volatility = 0.2
    
    for i in range(len(dates)):
        # 波动率聚集效应
        if i > 0:
            volatility = 0.9 * volatility + 0.1 * abs(returns[-1]) * 10
            volatility = np.clip(volatility, 0.1, 0.5)
        
        # 生成收益率
        daily_return = np.random.normal(0.0005, volatility/np.sqrt(252))  # 年化0.12%收益，动态波动率
        returns.append(daily_return)
    
    # 计算价格
    prices = [100]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    market_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates)),
        'high': np.array(prices) * (1 + np.random.uniform(0, 0.02, len(prices))),
        'low': np.array(prices) * (1 - np.random.uniform(0, 0.02, len(prices)))
    })
    
    # 创建增强回测引擎
    engine = AdvancedOptionBacktest(
        initial_capital=10000000,  # 1000万初始资金
        commission_rate=0.001
    )
    
    # 添加市场数据
    engine.add_market_data(market_data)
    
    # 创建雪球策略
    strategy = SnowballStrategy()
    
    # 运行雪球期权回测
    engine.run_snowball_backtest(market_data, strategy)
    
    # 获取详细分析
    analysis = engine.get_detailed_analysis()
    
    # 打印结果
    print("\n基础绩效指标:")
    print("-" * 40)
    for key, value in analysis.items():
        if key not in ['trade_analysis', 'position_analysis', 'risk_metrics']:
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    print("\n交易分析:")
    print("-" * 40)
    if 'trade_analysis' in analysis:
        for key, value in analysis['trade_analysis'].items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    print("\n风险指标:")
    print("-" * 40)
    if 'risk_metrics' in analysis:
        for key, value in analysis['risk_metrics'].items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    # 绘制结果
    if BACKTEST_CONFIG.plot_results:
        plot_comprehensive_results(engine, market_data)
    
    # 导出记录
    if BACKTEST_CONFIG.export_records:
        engine.export_records(BACKTEST_CONFIG.file_prefix + '_snowball')
    
    return engine, analysis

def plot_comprehensive_results(engine: AdvancedOptionBacktest, market_data: pd.DataFrame):
    """绘制综合回测结果"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 准备数据
    df_pnl = pd.DataFrame(engine.daily_pnl)
    df_pnl['date'] = pd.to_datetime(df_pnl['date'])
    df_pnl = df_pnl.set_index('date')
    
    market_data = market_data.set_index('date')
    
    # 1. 标的价格走势
    axes[0, 0].plot(market_data.index, market_data['close'], label='标的价格', color='blue')
    axes[0, 0].set_title('标的资产价格走势')
    axes[0, 0].set_ylabel('价格')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # 2. 资金曲线
    axes[0, 1].plot(df_pnl.index, df_pnl['total_value'], label='总资产', color='green')
    axes[0, 1].axhline(y=engine.initial_capital, color='red', linestyle='--', label='初始资金')
    axes[0, 1].set_title('资金曲线')
    axes[0, 1].set_ylabel('资产价值')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # 3. 每日收益率分布
    returns = df_pnl['daily_return'].dropna()
    axes[1, 0].hist(returns * 100, bins=50, alpha=0.7, color='purple')
    axes[1, 0].axvline(returns.mean() * 100, color='red', linestyle='--', label=f'均值: {returns.mean()*100:.3f}%')
    axes[1, 0].set_title('每日收益率分布')
    axes[1, 0].set_xlabel('收益率 (%)')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. 回撤分析
    cumulative = df_pnl['total_value']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    axes[1, 1].fill_between(df_pnl.index, drawdown, 0, alpha=0.3, color='red')
    axes[1, 1].plot(df_pnl.index, drawdown, color='red')
    axes[1, 1].set_title('回撤分析')
    axes[1, 1].set_ylabel('回撤 (%)')
    axes[1, 1].grid(True)
    
    # 5. 持仓价值
    axes[2, 0].plot(df_pnl.index, df_pnl['market_value'], label='持仓市值', color='orange')
    axes[2, 0].set_title('持仓市值变化')
    axes[2, 0].set_ylabel('市值')
    axes[2, 0].grid(True)
    axes[2, 0].legend()
    
    # 6. Delta暴露
    axes[2, 1].plot(df_pnl.index, df_pnl['portfolio_delta'], label='组合Delta', color='brown')
    axes[2, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2, 1].set_title('Delta暴露')
    axes[2, 1].set_ylabel('Delta')
    axes[2, 1].grid(True)
    axes[2, 1].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 运行综合回测
    engine, analysis = run_comprehensive_backtest()
    
    print(f"\n回测完成！生成了以下文件:")
    print(f"- {BACKTEST_CONFIG.file_prefix}_snowball_trades.csv")
    print(f"- {BACKTEST_CONFIG.file_prefix}_snowball_positions.csv") 
    print(f"- {BACKTEST_CONFIG.file_prefix}_snowball_daily_pnl.csv")