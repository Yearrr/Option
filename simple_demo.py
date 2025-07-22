#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的卖出看涨价差策略演示脚本（无图形输出）
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OptionLeg:
    """期权组合中的单腿"""
    option_type: str  # 'call' or 'put'
    strike: float
    expiry_date: str
    position: int  # 正数为买入，负数为卖出
    premium: float = 0.0
    iv: float = 0.2  # 隐含波动率

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

class SimpleSellCallSpread:
    """简化的卖出看涨价差策略"""
    
    def __init__(self):
        self.initial_capital = 100000
        self.current_capital = 100000
        self.trades = []
        self.daily_records = []
        
    def create_market_data(self):
        """创建示例市场数据"""
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        
        # 生成股价路径
        np.random.seed(42)
        prices = [100]
        volatility = [0.25]
        
        for i in range(1, len(dates)):
            # 波动率均值回归
            vol_mean = 0.25
            vol_speed = 0.1
            vol_shock = np.random.normal(0, 0.03)
            new_vol = max(0.1, volatility[-1] * 0.99 + vol_speed * (vol_mean - volatility[-1]) + vol_shock)
            volatility.append(new_vol)
            
            # 股价变化
            daily_return = np.random.normal(0.0002, new_vol / np.sqrt(252))
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        self.market_data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'iv': volatility
        })
        
        print(f"生成了 {len(self.market_data)} 天的市场数据")
        print(f"价格范围: {min(prices):.2f} - {max(prices):.2f}")
        print(f"IV范围: {min(volatility):.2%} - {max(volatility):.2%}")
    
    def create_call_spread(self, spot_price: float, iv: float, date: str):
        """创建卖出看涨价差"""
        
        # 选择行权价
        short_strike = spot_price * 1.03  # 卖出3%虚值看涨
        long_strike = spot_price * 1.08   # 买入8%虚值看涨
        
        time_to_expiry = 30 / 365.0  # 30天到期
        
        # 计算期权价格
        short_call_price = black_scholes_price(spot_price, short_strike, time_to_expiry, 0.03, iv, 'call')
        long_call_price = black_scholes_price(spot_price, long_strike, time_to_expiry, 0.03, iv, 'call')
        
        # 净收入
        net_credit = short_call_price - long_call_price
        max_profit = net_credit
        max_loss = (long_strike - short_strike) - net_credit
        
        return {
            'date': date,
            'spot_price': spot_price,
            'short_strike': short_strike,
            'long_strike': long_strike,
            'short_call_price': short_call_price,
            'long_call_price': long_call_price,
            'net_credit': net_credit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'iv': iv,
            'expiry_date': (pd.to_datetime(date) + timedelta(days=30)).strftime('%Y-%m-%d')
        }
    
    def calculate_spread_value(self, spread: dict, current_price: float, current_iv: float, current_date: str):
        """计算价差当前价值"""
        
        expiry_date = pd.to_datetime(spread['expiry_date'])
        current_date = pd.to_datetime(current_date)
        days_to_expiry = (expiry_date - current_date).days
        
        if days_to_expiry <= 0:
            # 到期价值
            short_value = max(current_price - spread['short_strike'], 0)
            long_value = max(current_price - spread['long_strike'], 0)
        else:
            # 当前市场价值
            time_to_expiry = days_to_expiry / 365.0
            short_value = black_scholes_price(current_price, spread['short_strike'], 
                                            time_to_expiry, 0.03, current_iv, 'call')
            long_value = black_scholes_price(current_price, spread['long_strike'], 
                                           time_to_expiry, 0.03, current_iv, 'call')
        
        current_spread_value = long_value - short_value  # 对于卖出价差
        pnl = spread['net_credit'] - current_spread_value
        
        return current_spread_value, pnl
    
    def run_backtest(self):
        """运行回测"""
        
        print("\n开始卖出看涨价差策略回测...")
        print("="*60)
        
        active_spreads = []
        trade_count = 0
        
        for i, row in self.market_data.iterrows():
            date = row['date'].strftime('%Y-%m-%d')
            spot_price = row['close']
            iv = row['iv']
            
            # 每20天开一个新仓位
            if i % 20 == 0 and i >= 30 and iv > 0.2:  # IV>20%时开仓
                spread = self.create_call_spread(spot_price, iv, date)
                active_spreads.append(spread)
                trade_count += 1
                
                print(f"{date}: 开立价差 #{trade_count}")
                print(f"  现价: {spot_price:.2f}, IV: {iv:.2%}")
                print(f"  卖出行权价: {spread['short_strike']:.2f} (权利金: {spread['short_call_price']:.2f})")
                print(f"  买入行权价: {spread['long_strike']:.2f} (权利金: {spread['long_call_price']:.2f})")
                print(f"  净收入: {spread['net_credit']:.2f}")
                print(f"  最大盈利: {spread['max_profit']:.2f}")
                print(f"  最大损失: {spread['max_loss']:.2f}")
            
            # 检查现有持仓
            spreads_to_close = []
            total_unrealized_pnl = 0
            
            for j, spread in enumerate(active_spreads):
                current_value, pnl = self.calculate_spread_value(spread, spot_price, iv, date)
                total_unrealized_pnl += pnl
                
                # 检查平仓条件
                days_to_expiry = (pd.to_datetime(spread['expiry_date']) - pd.to_datetime(date)).days
                profit_pct = pnl / spread['net_credit'] if spread['net_credit'] > 0 else 0
                
                should_close = False
                close_reason = ""
                
                if days_to_expiry <= 0:
                    should_close = True
                    close_reason = "到期"
                elif profit_pct >= 0.5:  # 50%止盈
                    should_close = True
                    close_reason = "止盈"
                elif profit_pct <= -2.0:  # 200%止损
                    should_close = True
                    close_reason = "止损"
                
                if should_close:
                    spreads_to_close.append((j, pnl, close_reason))
                    print(f"{date}: 平仓价差 (原因: {close_reason})")
                    print(f"  盈亏: {pnl:.2f}, 收益率: {profit_pct:.1%}")
                    
                    self.trades.append({
                        'open_date': spread['date'],
                        'close_date': date,
                        'days_held': (pd.to_datetime(date) - pd.to_datetime(spread['date'])).days,
                        'entry_credit': spread['net_credit'],
                        'pnl': pnl,
                        'return_pct': profit_pct,
                        'close_reason': close_reason,
                        'entry_iv': spread['iv'],
                        'exit_iv': iv
                    })
            
            # 移除已平仓的价差
            for j, _, _ in reversed(spreads_to_close):
                active_spreads.pop(j)
            
            # 记录每日数据
            total_capital = self.current_capital + total_unrealized_pnl
            self.daily_records.append({
                'date': date,
                'spot_price': spot_price,
                'iv': iv,
                'active_positions': len(active_spreads),
                'unrealized_pnl': total_unrealized_pnl,
                'total_value': total_capital
            })
        
        print(f"\n回测完成！共完成 {len(self.trades)} 笔交易")
    
    def analyze_results(self):
        """分析回测结果"""
        
        if not self.trades:
            print("没有完成的交易可供分析")
            return
        
        df_trades = pd.DataFrame(self.trades)
        df_daily = pd.DataFrame(self.daily_records)
        
        # 计算统计指标
        win_trades = df_trades[df_trades['pnl'] > 0]
        lose_trades = df_trades[df_trades['pnl'] <= 0]
        
        total_pnl = df_trades['pnl'].sum()
        win_rate = len(win_trades) / len(df_trades)
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = lose_trades['pnl'].mean() if len(lose_trades) > 0 else 0
        profit_factor = abs(win_trades['pnl'].sum() / lose_trades['pnl'].sum()) if len(lose_trades) > 0 and lose_trades['pnl'].sum() != 0 else np.inf
        
        print("\n" + "="*60)
        print("回测结果分析")
        print("="*60)
        print(f"总交易次数: {len(df_trades)}")
        print(f"胜率: {win_rate:.1%}")
        print(f"总盈亏: {total_pnl:.2f}")
        print(f"平均获利: {avg_win:.2f}")
        print(f"平均亏损: {avg_loss:.2f}")
        print(f"盈亏比: {profit_factor:.2f}")
        print(f"平均持仓天数: {df_trades['days_held'].mean():.1f}")
        
        # 按平仓原因分类
        print(f"\n平仓原因统计:")
        close_reasons = df_trades['close_reason'].value_counts()
        for reason, count in close_reasons.items():
            reason_trades = df_trades[df_trades['close_reason'] == reason]
            avg_pnl = reason_trades['pnl'].mean()
            print(f"  {reason}: {count}次, 平均盈亏: {avg_pnl:.2f}")
        
        # IV变化分析
        df_trades['iv_change'] = df_trades['exit_iv'] - df_trades['entry_iv']
        iv_down_trades = df_trades[df_trades['iv_change'] < 0]
        iv_up_trades = df_trades[df_trades['iv_change'] > 0]
        
        print(f"\nIV变化影响分析:")
        print(f"IV下降交易: {len(iv_down_trades)}次, 平均盈亏: {iv_down_trades['pnl'].mean():.2f}")
        print(f"IV上升交易: {len(iv_up_trades)}次, 平均盈亏: {iv_up_trades['pnl'].mean():.2f}")
        
        # 最终资金
        final_value = df_daily['total_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        print(f"\n资金变化:")
        print(f"初始资金: {self.initial_capital:,.2f}")
        print(f"最终价值: {final_value:,.2f}")
        print(f"总收益率: {total_return:.2%}")
        
        return {
            'total_trades': len(df_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'final_value': final_value
        }

def main():
    """主函数"""
    
    print("=" * 80)
    print("卖出看涨价差（Sell Call Spread）策略回测演示")
    print("=" * 80)
    print()
    print("策略说明:")
    print("• 卖出3%虚值看涨期权 + 买入8%虚值看涨期权")
    print("• 目标：获得时间价值衰减和隐含波动率下降收益")
    print("• 止盈：50%最大利润")
    print("• 止损：200%最大损失")
    print("• 开仓条件：IV > 20%")
    print()
    
    # 创建并运行回测
    strategy = SimpleSellCallSpread()
    strategy.create_market_data()
    strategy.run_backtest()
    results = strategy.analyze_results()
    
    print("\n" + "="*80)
    print("策略总结")
    print("="*80)
    print("✅ 卖出看涨价差适合以下市场:")
    print("  • 横盘或小幅上涨的趋势")
    print("  • 隐含波动率高位回落")
    print("  • 时间价值衰减明显的短期期权")
    print()
    print("⚠️  主要风险:")
    print("  • 标的价格大幅上涨突破卖出行权价")
    print("  • 隐含波动率意外飙升")
    print("  • 流动性不足导致无法及时平仓")
    print()
    print("📈 优化建议:")
    print("  • 结合VIX等波动率指标选择开仓时机")
    print("  • 考虑Delta对冲降低方向性风险")
    print("  • 建立动态止损机制")
    print("  • 分散到期时间降低集中风险")
    print("="*80)

if __name__ == "__main__":
    main()