#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权价差策略回测框架
专注于场内期权策略组合，如sell call spread, ratio spreads等
主要目标：获得时间价值和隐含波动率下降的收益
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class OptionLeg:
    """期权组合中的单腿"""
    option_type: str  # 'call' or 'put'
    strike: float
    expiry_date: str
    position: int  # 正数为买入，负数为卖出
    premium: float = 0.0
    iv: float = 0.2  # 隐含波动率
    
@dataclass
class SpreadStrategy:
    """价差策略定义"""
    name: str
    legs: List[OptionLeg]
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven_points: List[float] = field(default_factory=list)
    
    def calculate_payoff(self, spot_price: float, time_to_expiry: float = 0) -> float:
        """计算在给定现货价格下的策略收益"""
        total_payoff = 0.0
        
        for leg in self.legs:
            if time_to_expiry > 0:
                # 期权未到期，使用BS定价
                option_value = black_scholes_price(
                    spot_price, leg.strike, time_to_expiry, 
                    0.03, leg.iv, leg.option_type
                )
            else:
                # 期权到期，使用内在价值
                if leg.option_type == 'call':
                    option_value = max(spot_price - leg.strike, 0)
                else:
                    option_value = max(leg.strike - spot_price, 0)
            
            # 计算该腿的盈亏
            leg_pnl = leg.position * (option_value - leg.premium)
            total_payoff += leg_pnl
            
        return total_payoff

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, 
                       option_type: str = 'call') -> float:
    """Black-Scholes期权定价"""
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    
    return price

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                    option_type: str = 'call') -> Dict[str, float]:
    """计算期权希腊字母"""
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = stats.norm.cdf(d1)
        theta = -(S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * stats.norm.cdf(d2))
    else:
        delta = -stats.norm.cdf(-d1)
        theta = -(S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * stats.norm.cdf(-d2))
    
    gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * stats.norm.pdf(d1) * np.sqrt(T)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta / 365,  # 日theta
        'vega': vega / 100     # 1%波动率变化的vega
    }

class IVEstimator:
    """隐含波动率估计器"""
    
    @staticmethod
    def historical_volatility(prices: pd.Series, window: int = 30) -> float:
        """计算历史波动率"""
        returns = prices.pct_change().dropna()
        if len(returns) < window:
            return 0.2  # 默认值
        
        recent_returns = returns.tail(window)
        return recent_returns.std() * np.sqrt(252)
    
    @staticmethod
    def garch_volatility(prices: pd.Series, window: int = 60) -> float:
        """简化的GARCH波动率预测"""
        returns = prices.pct_change().dropna()
        if len(returns) < window:
            return 0.2
        
        recent_returns = returns.tail(window)
        
        # 简化的GARCH(1,1)
        alpha = 0.1
        beta = 0.85
        omega = 0.05
        
        var_forecast = omega
        for ret in recent_returns:
            var_forecast = omega + alpha * ret**2 + beta * var_forecast
        
        return np.sqrt(var_forecast * 252)

class SellCallSpreadStrategy:
    """卖出看涨价差策略"""
    
    def __init__(self, short_strike_delta: float = 0.3, long_strike_delta: float = 0.15,
                 days_to_expiry: int = 30, profit_target: float = 0.5, 
                 stop_loss: float = 2.0):
        """
        初始化卖出看涨价差策略
        
        Parameters:
        -----------
        short_strike_delta : float
            卖出期权的delta目标（通常0.2-0.4）
        long_strike_delta : float  
            买入期权的delta目标（通常0.1-0.2）
        days_to_expiry : int
            期权到期天数
        profit_target : float
            止盈比例（相对于收到的净权利金）
        stop_loss : float
            止损比例（相对于收到的净权利金）
        """
        self.short_strike_delta = short_strike_delta
        self.long_strike_delta = long_strike_delta
        self.days_to_expiry = days_to_expiry
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        
    def find_strikes_by_delta(self, spot_price: float, time_to_expiry: float, 
                             iv: float, r: float = 0.03) -> Tuple[float, float]:
        """根据目标delta找到行权价"""
        
        def delta_objective(strike, target_delta):
            greeks = calculate_greeks(spot_price, strike, time_to_expiry, r, iv, 'call')
            return (greeks['delta'] - target_delta) ** 2
        
        # 使用简单搜索找到目标行权价
        strikes = np.linspace(spot_price * 0.8, spot_price * 1.5, 100)
        
        short_strike = None
        long_strike = None
        
        min_error_short = float('inf')
        min_error_long = float('inf')
        
        for strike in strikes:
            greeks = calculate_greeks(spot_price, strike, time_to_expiry, r, iv, 'call')
            
            # 寻找卖出行权价
            error_short = abs(greeks['delta'] - self.short_strike_delta)
            if error_short < min_error_short:
                min_error_short = error_short
                short_strike = strike
                
            # 寻找买入行权价
            error_long = abs(greeks['delta'] - self.long_strike_delta)
            if error_long < min_error_long:
                min_error_long = error_long
                long_strike = strike
        
        return short_strike, long_strike
    
    def create_spread(self, spot_price: float, current_date: str, iv: float) -> SpreadStrategy:
        """创建卖出看涨价差组合"""
        
        time_to_expiry = self.days_to_expiry / 365.0
        expiry_date = (pd.to_datetime(current_date) + timedelta(days=self.days_to_expiry)).strftime('%Y-%m-%d')
        
        # 根据delta找到行权价
        short_strike, long_strike = self.find_strikes_by_delta(spot_price, time_to_expiry, iv)
        
        # 计算期权价格
        short_call_price = black_scholes_price(spot_price, short_strike, time_to_expiry, 0.03, iv, 'call')
        long_call_price = black_scholes_price(spot_price, long_strike, time_to_expiry, 0.03, iv, 'call')
        
        # 创建期权腿
        legs = [
            OptionLeg(
                option_type='call',
                strike=short_strike,
                expiry_date=expiry_date,
                position=-1,  # 卖出
                premium=short_call_price,
                iv=iv
            ),
            OptionLeg(
                option_type='call', 
                strike=long_strike,
                expiry_date=expiry_date,
                position=1,   # 买入
                premium=long_call_price,
                iv=iv
            )
        ]
        
        # 计算策略参数
        net_credit = short_call_price - long_call_price  # 收到的净权利金
        max_profit = net_credit
        max_loss = (long_strike - short_strike) - net_credit
        breakeven = short_strike + net_credit
        
        strategy = SpreadStrategy(
            name=f"SellCallSpread_{current_date}",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven]
        )
        
        return strategy

class SpreadBacktestEngine:
    """期权价差策略回测引擎"""
    
    def __init__(self, initial_capital: float = 100000, commission_per_contract: float = 1.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_per_contract = commission_per_contract
        
        self.positions = []  # 当前持仓
        self.trade_history = []  # 交易历史
        self.daily_pnl = []  # 每日盈亏
        self.iv_estimator = IVEstimator()
        
    def add_market_data(self, data: pd.DataFrame):
        """添加市场数据"""
        self.market_data = data.copy()
        self.market_data['date'] = pd.to_datetime(self.market_data['date'])
        self.market_data = self.market_data.set_index('date').sort_index()
        
        # 计算隐含波动率
        self._calculate_iv_series()
    
    def _calculate_iv_series(self):
        """计算隐含波动率时间序列"""
        iv_series = []
        
        for i, (date, row) in enumerate(self.market_data.iterrows()):
            if i < 30:
                iv = 0.25  # 默认IV
            else:
                # 使用历史波动率作为IV代理
                hist_prices = self.market_data['close'].iloc[:i+1]
                iv = self.iv_estimator.historical_volatility(hist_prices, window=30)
                
                # 添加一些随机性模拟IV的变化
                iv_shock = np.random.normal(0, 0.02)
                iv = max(0.1, iv + iv_shock)
                
            iv_series.append(iv)
        
        self.market_data['iv'] = iv_series
    
    def open_spread_position(self, strategy: SpreadStrategy, date: str, contracts: int = 1):
        """开立价差头寸"""
        
        total_premium = 0
        total_commission = 0
        
        # 计算总权利金和手续费
        for leg in strategy.legs:
            leg_premium = leg.position * leg.premium * contracts
            leg_commission = abs(leg.position) * contracts * self.commission_per_contract
            
            total_premium += leg_premium
            total_commission += leg_commission
        
        # 记录开仓交易
        trade_record = {
            'date': date,
            'action': 'open',
            'strategy_name': strategy.name,
            'contracts': contracts,
            'net_premium': total_premium,
            'commission': total_commission,
            'max_profit': strategy.max_profit * contracts,
            'max_loss': strategy.max_loss * contracts,
            'iv_at_entry': self.market_data.loc[date, 'iv']
        }
        
        self.trade_history.append(trade_record)
        
        # 添加到持仓
        position = {
            'strategy': strategy,
            'contracts': contracts,
            'entry_date': date,
            'entry_premium': total_premium,
            'entry_commission': total_commission,
            'days_held': 0
        }
        
        self.positions.append(position)
        
        # 更新资金（卖出价差收到权利金）
        self.current_capital += total_premium - total_commission
        
        print(f"{date}: 开立 {strategy.name} {contracts}手，收到净权利金: {total_premium:.2f}")
    
    def close_spread_position(self, position_idx: int, date: str, reason: str = 'expiry'):
        """平仓价差头寸"""
        
        if position_idx >= len(self.positions):
            return
            
        position = self.positions[position_idx]
        strategy = position['strategy']
        contracts = position['contracts']
        
        current_price = self.market_data.loc[date, 'close']
        current_iv = self.market_data.loc[date, 'iv']
        
        # 计算到期时间
        expiry_date = pd.to_datetime(strategy.legs[0].expiry_date)
        current_date = pd.to_datetime(date)
        time_to_expiry = max(0, (expiry_date - current_date).days / 365.0)
        
        # 计算当前策略价值
        current_value = 0
        total_commission = 0
        
        for leg in strategy.legs:
            if time_to_expiry > 0:
                # 使用当前IV重新定价
                leg_value = black_scholes_price(
                    current_price, leg.strike, time_to_expiry, 0.03, current_iv, leg.option_type
                )
            else:
                # 到期行权价值
                if leg.option_type == 'call':
                    leg_value = max(current_price - leg.strike, 0)
                else:
                    leg_value = max(leg.strike - current_price, 0)
            
            current_value += leg.position * leg_value
            total_commission += abs(leg.position) * contracts * self.commission_per_contract
        
        current_value *= contracts
        
        # 计算盈亏
        entry_value = position['entry_premium']
        total_pnl = entry_value - current_value  # 对于卖出价差，入场收钱，平仓付钱
        net_pnl = total_pnl - total_commission
        
        # 记录平仓交易
        trade_record = {
            'date': date,
            'action': 'close',
            'strategy_name': strategy.name,
            'contracts': contracts,
            'close_value': current_value,
            'commission': total_commission,
            'total_pnl': total_pnl,
            'net_pnl': net_pnl,
            'days_held': position['days_held'],
            'close_reason': reason,
            'iv_at_exit': current_iv,
            'iv_change': current_iv - trade_record.get('iv_at_entry', current_iv)
        }
        
        self.trade_history.append(trade_record)
        
        # 更新资金
        self.current_capital -= current_value + total_commission
        
        # 移除持仓
        self.positions.pop(position_idx)
        
        print(f"{date}: 平仓 {strategy.name} {contracts}手，净盈亏: {net_pnl:.2f}")
        
        return net_pnl
    
    def update_positions(self, date: str):
        """更新持仓状态"""
        
        current_price = self.market_data.loc[date, 'close']
        current_iv = self.market_data.loc[date, 'iv']
        
        positions_to_close = []
        total_unrealized_pnl = 0
        
        for i, position in enumerate(self.positions):
            strategy = position['strategy']
            contracts = position['contracts']
            
            # 更新持有天数
            entry_date = pd.to_datetime(position['entry_date'])
            current_date = pd.to_datetime(date)
            position['days_held'] = (current_date - entry_date).days
            
            # 计算到期时间
            expiry_date = pd.to_datetime(strategy.legs[0].expiry_date)
            time_to_expiry = max(0, (expiry_date - current_date).days / 365.0)
            
            # 检查是否到期
            if time_to_expiry <= 0:
                positions_to_close.append((i, 'expiry'))
                continue
            
            # 计算当前价值和盈亏
            current_value = 0
            for leg in strategy.legs:
                leg_value = black_scholes_price(
                    current_price, leg.strike, time_to_expiry, 0.03, current_iv, leg.option_type
                )
                current_value += leg.position * leg_value
            
            current_value *= contracts
            unrealized_pnl = position['entry_premium'] - current_value
            total_unrealized_pnl += unrealized_pnl
            
            # 检查止盈止损
            entry_credit = position['entry_premium']
            if entry_credit > 0:  # 卖出价差
                profit_target = entry_credit * 0.5  # 50%利润目标
                stop_loss_level = entry_credit * -2.0  # 200%止损
                
                if unrealized_pnl >= profit_target:
                    positions_to_close.append((i, 'profit_target'))
                elif unrealized_pnl <= stop_loss_level:
                    positions_to_close.append((i, 'stop_loss'))
        
        # 平仓到期或触发条件的头寸
        for i, reason in sorted(positions_to_close, reverse=True):
            self.close_spread_position(i, date, reason)
        
        # 记录每日数据
        total_capital = self.current_capital + total_unrealized_pnl
        
        daily_record = {
            'date': date,
            'spot_price': current_price,
            'iv': current_iv,
            'cash': self.current_capital,
            'unrealized_pnl': total_unrealized_pnl,
            'total_value': total_capital,
            'active_positions': len(self.positions),
            'daily_return': 0  # 将在后处理中计算
        }
        
        self.daily_pnl.append(daily_record)
    
    def run_sell_call_spread_backtest(self, strategy_params: dict = None):
        """运行卖出看涨价差回测"""
        
        if strategy_params is None:
            strategy_params = {
                'signal_frequency': 15,  # 每15个交易日开新仓
                'contracts_per_trade': 1,
                'max_positions': 5
            }
        
        sell_call_strategy = SellCallSpreadStrategy()
        
        print("开始卖出看涨价差策略回测...")
        print("="*50)
        
        signal_count = 0
        
        for i, (date, row) in enumerate(self.market_data.iterrows()):
            date_str = date.strftime('%Y-%m-%d')
            
            # 生成开仓信号
            if (i % strategy_params['signal_frequency'] == 0 and 
                len(self.positions) < strategy_params['max_positions'] and
                i >= 30):  # 确保有足够的历史数据计算IV
                
                spot_price = row['close']
                iv = row['iv']
                
                # 只在IV相对较高时开仓
                if iv > 0.2:  # IV阈值
                    try:
                        spread = sell_call_strategy.create_spread(spot_price, date_str, iv)
                        self.open_spread_position(spread, date_str, strategy_params['contracts_per_trade'])
                        signal_count += 1
                    except Exception as e:
                        print(f"创建价差失败 {date_str}: {e}")
            
            # 更新持仓
            self.update_positions(date_str)
        
        # 计算每日收益率
        self._calculate_daily_returns()
        
        print(f"回测完成！总共开立了 {signal_count} 个价差头寸")
    
    def _calculate_daily_returns(self):
        """计算每日收益率"""
        if len(self.daily_pnl) <= 1:
            return
        
        for i in range(1, len(self.daily_pnl)):
            prev_value = self.daily_pnl[i-1]['total_value']
            curr_value = self.daily_pnl[i]['total_value']
            if prev_value != 0:
                self.daily_pnl[i]['daily_return'] = (curr_value - prev_value) / prev_value
    
    def get_performance_summary(self) -> Dict:
        """获取绩效摘要"""
        
        if not self.daily_pnl:
            return {}
        
        df_pnl = pd.DataFrame(self.daily_pnl)
        df_trades = pd.DataFrame([t for t in self.trade_history if t['action'] == 'close'])
        
        if df_trades.empty:
            return {'message': '没有已完成的交易'}
        
        total_return = (df_pnl['total_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        daily_returns = df_pnl['daily_return'].dropna()
        
        # 交易统计
        win_trades = df_trades[df_trades['net_pnl'] > 0]
        lose_trades = df_trades[df_trades['net_pnl'] <= 0]
        
        summary = {
            'initial_capital': self.initial_capital,
            'final_value': df_pnl['total_value'].iloc[-1],
            'total_return': total_return,
            'total_trades': len(df_trades),
            'win_rate': len(win_trades) / len(df_trades) if len(df_trades) > 0 else 0,
            'avg_win': win_trades['net_pnl'].mean() if len(win_trades) > 0 else 0,
            'avg_loss': lose_trades['net_pnl'].mean() if len(lose_trades) > 0 else 0,
            'profit_factor': abs(win_trades['net_pnl'].sum() / lose_trades['net_pnl'].sum()) if len(lose_trades) > 0 and lose_trades['net_pnl'].sum() != 0 else np.inf,
            'sharpe_ratio': daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(df_pnl),
            'avg_days_held': df_trades['days_held'].mean(),
            'avg_iv_change': df_trades['iv_change'].mean() if 'iv_change' in df_trades.columns else 0
        }
        
        return summary
    
    def _calculate_max_drawdown(self, df_pnl: pd.DataFrame) -> float:
        """计算最大回撤"""
        cumulative = df_pnl['total_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def plot_results(self):
        """绘制回测结果"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 准备数据
        df_pnl = pd.DataFrame(self.daily_pnl)
        df_pnl['date'] = pd.to_datetime(df_pnl['date'])
        df_trades = pd.DataFrame([t for t in self.trade_history if t['action'] == 'close'])
        
        # 1. 标的价格和IV
        axes[0, 0].plot(df_pnl['date'], df_pnl['spot_price'], label='标的价格', color='blue')
        ax_iv = axes[0, 0].twinx()
        ax_iv.plot(df_pnl['date'], df_pnl['iv'] * 100, label='隐含波动率(%)', color='red', alpha=0.7)
        axes[0, 0].set_title('标的价格 vs 隐含波动率')
        axes[0, 0].set_ylabel('价格')
        ax_iv.set_ylabel('IV (%)')
        axes[0, 0].legend(loc='upper left')
        ax_iv.legend(loc='upper right')
        axes[0, 0].grid(True)
        
        # 2. 资金曲线
        axes[0, 1].plot(df_pnl['date'], df_pnl['total_value'], label='总资产', color='green')
        axes[0, 1].axhline(y=self.initial_capital, color='red', linestyle='--', label='初始资金')
        axes[0, 1].set_title('资金曲线')
        axes[0, 1].set_ylabel('资产价值')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. 每笔交易盈亏
        if not df_trades.empty:
            axes[0, 2].bar(range(len(df_trades)), df_trades['net_pnl'], 
                          color=['green' if x > 0 else 'red' for x in df_trades['net_pnl']])
            axes[0, 2].set_title('每笔交易盈亏')
            axes[0, 2].set_xlabel('交易序号')
            axes[0, 2].set_ylabel('盈亏')
            axes[0, 2].grid(True)
        
        # 4. 持仓数量
        axes[1, 0].plot(df_pnl['date'], df_pnl['active_positions'], label='活跃持仓数', color='orange')
        axes[1, 0].set_title('持仓数量变化')
        axes[1, 0].set_ylabel('持仓数')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 5. 收益率分布
        if not df_pnl.empty:
            returns = df_pnl['daily_return'].dropna()
            if len(returns) > 0:
                axes[1, 1].hist(returns * 100, bins=30, alpha=0.7, color='purple')
                axes[1, 1].axvline(returns.mean() * 100, color='red', linestyle='--', 
                                  label=f'均值: {returns.mean()*100:.3f}%')
                axes[1, 1].set_title('每日收益率分布')
                axes[1, 1].set_xlabel('收益率 (%)')
                axes[1, 1].set_ylabel('频数')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
        
        # 6. IV变化 vs 盈亏关系
        if not df_trades.empty and 'iv_change' in df_trades.columns:
            axes[1, 2].scatter(df_trades['iv_change'] * 100, df_trades['net_pnl'], alpha=0.6)
            axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 2].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 2].set_title('IV变化 vs 交易盈亏')
            axes[1, 2].set_xlabel('IV变化 (%)')
            axes[1, 2].set_ylabel('净盈亏')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, file_prefix: str = 'sell_call_spread'):
        """导出回测结果"""
        
        # 导出交易记录
        df_trades = pd.DataFrame(self.trade_history)
        df_trades.to_csv(f'{file_prefix}_trades.csv', index=False, encoding='utf-8-sig')
        
        # 导出每日盈亏
        df_daily = pd.DataFrame(self.daily_pnl)
        df_daily.to_csv(f'{file_prefix}_daily.csv', index=False, encoding='utf-8-sig')
        
        print(f"结果已导出到 {file_prefix}_*.csv 文件")

def create_realistic_market_data(start_date: str = '2022-01-01', 
                               end_date: str = '2024-12-31') -> pd.DataFrame:
    """创建更真实的市场数据，包含趋势和波动率聚集"""
    
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # 使用真实的VIX风格波动率数据
    np.random.seed(42)
    n_days = len(dates)
    
    # 生成具有均值回归特性的波动率
    volatility = [0.2]  # 初始波动率
    for i in range(1, n_days):
        # 波动率均值回归模型
        vol_mean = 0.2
        vol_speed = 0.1
        vol_vol = 0.05
        
        dv = vol_speed * (vol_mean - volatility[-1]) + vol_vol * np.random.normal()
        new_vol = max(0.08, volatility[-1] + dv / 252)
        volatility.append(new_vol)
    
    # 生成股价路径
    prices = [100]
    for i in range(1, n_days):
        # 股价具有轻微上升趋势
        mu = 0.08 / 252  # 年化8%漂移
        sigma = volatility[i]
        
        daily_return = np.random.normal(mu, sigma / np.sqrt(252))
        new_price = prices[-1] * np.exp(daily_return)
        prices.append(new_price)
    
    # 构造数据框
    market_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': np.array(prices) * (1 + np.random.uniform(0, 0.015, len(prices))),
        'low': np.array(prices) * (1 - np.random.uniform(0, 0.015, len(prices))),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    return market_data

def run_sell_call_spread_demo():
    """运行卖出看涨价差策略演示"""
    
    print("卖出看涨价差策略回测演示")
    print("="*60)
    print("策略说明：")
    print("- 卖出近月看涨期权（较低行权价，Delta约0.3）")
    print("- 买入近月看涨期权（较高行权价，Delta约0.15）")
    print("- 目标：获得时间价值衰减和隐含波动率下降的收益")
    print("- 最适合：横盘或小幅上涨的市场环境")
    print("="*60)
    
    # 创建市场数据
    market_data = create_realistic_market_data('2022-01-01', '2024-12-31')
    
    # 创建回测引擎
    engine = SpreadBacktestEngine(initial_capital=100000, commission_per_contract=2.0)
    engine.add_market_data(market_data)
    
    # 设置策略参数
    strategy_params = {
        'signal_frequency': 10,  # 每10个交易日评估开仓
        'contracts_per_trade': 2,  # 每次开2手
        'max_positions': 8  # 最多同时持有8个头寸
    }
    
    # 运行回测
    engine.run_sell_call_spread_backtest(strategy_params)
    
    # 获取绩效摘要
    summary = engine.get_performance_summary()
    
    print("\n绩效摘要:")
    print("-" * 50)
    for key, value in summary.items():
        if isinstance(value, float):
            if 'rate' in key or 'return' in key or 'ratio' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # 绘制结果图表
    engine.plot_results()
    
    # 导出结果
    engine.export_results('sell_call_spread_demo')
    
    return engine, summary

if __name__ == "__main__":
    # 运行演示
    engine, summary = run_sell_call_spread_demo()
    
    print("\n策略分析:")
    print("-" * 50)
    print("✓ 时间价值衰减收益：期权卖方受益于Theta")
    print("✓ 隐含波动率下降收益：高IV时开仓，低IV时平仓")
    print("✓ 有限风险：最大损失 = 行权价差 - 收到的净权利金")
    print("✓ 有限收益：最大收益 = 收到的净权利金")
    print("\n建议优化方向:")
    print("- 结合VIX指标选择开仓时机")
    print("- 根据隐含波动率排名动态调整开仓频率")
    print("- 考虑Delta中性对冲降低方向性风险")
    print("- 优化止盈止损参数")