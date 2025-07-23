#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IV宽跨式期权策略回测示例
演示基于隐含波动率分位数的宽跨式期权策略与Delta对冲
"""

import sys
import os
sys.path.append('options_backtest_framework')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from trading_engine import TradingEngine
from iv_straddle_strategy import IVPercentileCalculator, StraddleStrategy

class MarketDataGenerator:
    """市场数据生成器"""
    
    def __init__(self, start_date: datetime, end_date: datetime, 
                 underlying_symbol: str = "AAPL"):
        self.start_date = start_date
        self.end_date = end_date
        self.underlying_symbol = underlying_symbol
        self.current_price = 150.0  # 初始价格
        
    def generate_market_data(self) -> pd.DataFrame:
        """生成模拟市场数据"""
        
        # 生成日期序列
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        date_range = [d for d in date_range if d.weekday() < 5]  # 只包含工作日
        
        n_days = len(date_range)
        
        # 生成标的价格（几何布朗运动）
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, n_days)  # 日收益率
        prices = [self.current_price]
        
        for i in range(1, n_days):
            price = prices[-1] * (1 + returns[i])
            prices.append(max(price, 10))  # 价格下限
        
        # 生成隐含波动率（均值回归过程）
        iv_mean = 0.25
        iv_current = iv_mean
        iv_values = [iv_current]
        
        for i in range(1, n_days):
            # 均值回归 + 随机扰动
            iv_change = 0.05 * (iv_mean - iv_current) + np.random.normal(0, 0.02)
            iv_current = max(0.1, min(0.8, iv_current + iv_change))  # 限制范围
            iv_values.append(iv_current)
        
        # 创建DataFrame
        data = pd.DataFrame({
            'date': date_range,
            'underlying_price': prices,
            'iv': iv_values
        })
        
        return data
    
    def generate_options_data(self, underlying_price: float, iv: float, 
                             days_to_expiry: int = 30) -> Dict:
        """生成期权数据"""
        
        # 简化的Black-Scholes计算
        K = underlying_price  # ATM期权
        S = underlying_price
        T = days_to_expiry / 365.0
        r = 0.05  # 无风险利率
        sigma = iv
        
        # 计算d1, d2
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # 标准正态分布累积分布函数
        from scipy.stats import norm
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)
        
        # 期权价格
        call_price = S*N_d1 - K*np.exp(-r*T)*N_d2
        put_price = K*np.exp(-r*T)*N_neg_d2 - S*N_neg_d1
        
        # Greeks
        call_delta = N_d1
        put_delta = N_d1 - 1
        
        # Gamma (相同)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        call_theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*N_d2
        put_theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*N_neg_d2
        
        # Vega (相同)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        options_data = {
            f"{self.underlying_symbol}_CALL_ATM": {
                'price': max(0.01, call_price),
                'greeks': {
                    'delta': call_delta,
                    'gamma': gamma,
                    'theta': call_theta / 365,  # 日Theta
                    'vega': vega / 100  # Vega/1%
                }
            },
            f"{self.underlying_symbol}_PUT_ATM": {
                'price': max(0.01, put_price),
                'greeks': {
                    'delta': put_delta,
                    'gamma': gamma,
                    'theta': put_theta / 365,
                    'vega': vega / 100
                }
            },
            f"{self.underlying_symbol}_FUTURE": {
                'price': underlying_price * 1.02  # 期货略有升水
            }
        }
        
        return options_data

def run_iv_straddle_backtest():
    """运行IV宽跨式策略回测"""
    
    print("=== IV宽跨式期权策略回测 ===\n")
    
    # 设置回测参数
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    underlying_symbol = "AAPL"
    initial_cash = 500000  # 50万初始资金
    
    # 生成市场数据
    print("1. 生成模拟市场数据...")
    data_generator = MarketDataGenerator(start_date, end_date, underlying_symbol)
    market_df = data_generator.generate_market_data()
    print(f"生成了 {len(market_df)} 天的市场数据")
    
    # 创建回测引擎
    print("\n2. 初始化回测引擎...")
    trading_engine = TradingEngine(initial_cash=initial_cash)
    iv_calculator = IVPercentileCalculator(lookback_days=60)  # 60天历史
    
    # 创建策略
    strategy = StraddleStrategy(
        trading_engine=trading_engine,
        iv_calculator=iv_calculator,
        iv_percentile_threshold=85.0,   # IV分位数阈值85%
        delta_hedge_threshold=0.15,     # Delta对冲阈值0.15
        position_size=5                 # 每次交易5手
    )
    
    print(f"策略参数:")
    print(f"  IV分位数阈值: {strategy.iv_percentile_threshold}%")
    print(f"  Delta对冲阈值: {strategy.delta_hedge_threshold}")
    print(f"  头寸大小: {strategy.position_size} 手")
    
    # 回测循环
    print("\n3. 开始回测...")
    portfolio_values = []
    daily_summary = []
    
    for i, row in market_df.iterrows():
        current_date = row['date']
        underlying_price = row['underlying_price']
        iv = row['iv']
        
        trading_engine.current_date = current_date
        
        # 生成当日期权数据
        options_data = data_generator.generate_options_data(underlying_price, iv)
        
        # 添加标的数据
        market_data = {
            underlying_symbol: {
                'price': underlying_price,
                'iv': iv
            }
        }
        market_data.update(options_data)
        
        # 更新持仓价格
        trading_engine.update_positions(market_data)
        
        # 运行策略
        strategy.run_strategy(underlying_symbol, market_data, current_date)
        
        # 记录投资组合价值
        portfolio_value = trading_engine.get_portfolio_value()
        portfolio_values.append(portfolio_value)
        
        # 记录每日摘要
        breakdown = trading_engine.get_portfolio_breakdown()
        strategy_summary = strategy.get_strategy_summary()
        
        daily_summary.append({
            'date': current_date,
            'underlying_price': underlying_price,
            'iv': iv,
            'iv_percentile': iv_calculator.get_iv_percentile(underlying_symbol),
            'portfolio_value': portfolio_value,
            'cash': breakdown['cash'],
            'options_value': breakdown['options_value'],
            'futures_value': breakdown['futures_value'],
            'active_positions': strategy_summary['active_positions'],
            'hedge_positions': strategy_summary['hedge_positions']
        })
        
        # 记录每日投资组合
        trading_engine.record_daily_portfolio(current_date)
        
        # 定期输出进度
        if i % 50 == 0 or i == len(market_df) - 1:
            iv_pct = iv_calculator.get_iv_percentile(underlying_symbol)
            iv_pct_str = f"{iv_pct:.1f}%" if iv_pct else "N/A"
            print(f"  {current_date.strftime('%Y-%m-%d')}: 价格=${underlying_price:.2f}, "
                  f"IV={iv:.2f}({iv_pct_str}), 组合=${portfolio_value:,.0f}")
    
    # 分析结果
    print("\n4. 分析回测结果...")
    
    # 基本统计
    final_value = portfolio_values[-1]
    total_return = final_value - initial_cash
    return_pct = total_return / initial_cash * 100
    
    # 交易统计
    trade_history = trading_engine.get_trade_history()
    if not trade_history.empty:
        option_trades = trade_history[trade_history['strategy'].str.contains('straddle', na=False)]
        hedge_trades = trade_history[trade_history['strategy'].str.contains('hedge', na=False)]
        
        print(f"\n=== 回测结果摘要 ===")
        print(f"回测期间: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
        print(f"初始资金: ${initial_cash:,.2f}")
        print(f"最终价值: ${final_value:,.2f}")
        print(f"总收益: ${total_return:,.2f}")
        print(f"收益率: {return_pct:.2f}%")
        print(f"总交易数: {len(trade_history)}")
        print(f"期权交易: {len(option_trades)}")
        print(f"对冲交易: {len(hedge_trades)}")
        
        # 风险指标
        portfolio_series = pd.Series(portfolio_values)
        returns = portfolio_series.pct_change().dropna()
        
        max_drawdown = ((portfolio_series.cummax() - portfolio_series) / portfolio_series.cummax()).max()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        sharpe = return_pct / 100 / volatility if volatility > 0 else 0
        
        print(f"\n=== 风险指标 ===")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"年化波动率: {volatility:.2%}")
        print(f"夏普比率: {sharpe:.2f}")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(daily_summary)
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df.set_index('date', inplace=True)
    
    return results_df, trading_engine, strategy

def plot_backtest_results(results_df: pd.DataFrame):
    """绘制回测结果图表"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 图1: 投资组合价值
    axes[0].plot(results_df.index, results_df['portfolio_value'], 
                label='Portfolio Value', color='blue', linewidth=2)
    axes[0].set_title('投资组合价值变化')
    axes[0].set_ylabel('价值 ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 图2: 标的价格和IV
    ax2_twin = axes[1].twinx()
    axes[1].plot(results_df.index, results_df['underlying_price'], 
                label='标的价格', color='black', linewidth=1)
    ax2_twin.plot(results_df.index, results_df['iv'], 
                 label='隐含波动率', color='red', linewidth=1)
    
    # 标记IV分位数>=85%的点
    high_iv = results_df[results_df['iv_percentile'] >= 85.0]
    if not high_iv.empty:
        axes[1].scatter(high_iv.index, high_iv['underlying_price'], 
                       color='red', s=30, alpha=0.7, label='IV>=85%')
    
    axes[1].set_title('标的价格 & 隐含波动率')
    axes[1].set_ylabel('价格 ($)')
    ax2_twin.set_ylabel('隐含波动率')
    axes[1].legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # 图3: 持仓数量
    axes[2].plot(results_df.index, results_df['active_positions'], 
                label='活跃期权仓位', color='green', linewidth=2)
    axes[2].plot(results_df.index, results_df['hedge_positions'], 
                label='期货对冲仓位', color='orange', linewidth=2)
    axes[2].set_title('持仓数量')
    axes[2].set_ylabel('仓位数')
    axes[2].set_xlabel('日期')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iv_straddle_backtest_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    try:
        # 运行回测
        results_df, trading_engine, strategy = run_iv_straddle_backtest()
        
        # 显示详细的策略摘要
        print(f"\n=== 策略持仓详情 ===")
        strategy_summary = strategy.get_strategy_summary()
        print(f"活跃期权仓位: {strategy_summary['active_positions']}")
        print(f"期货对冲仓位: {strategy_summary['hedge_positions']}")
        
        if strategy_summary['positions_detail']:
            print("\n活跃仓位详情:")
            for symbol, pos_info in strategy_summary['positions_detail'].items():
                print(f"  {symbol}:")
                print(f"    入场日期: {pos_info['entry_date'].strftime('%Y-%m-%d')}")
                print(f"    仓位大小: {pos_info['position_size']} 手")
                print(f"    收到权利金: ${pos_info['total_premium']:,.2f}")
        
        if strategy_summary['hedge_detail']:
            print("\n对冲仓位详情:")
            for symbol, quantity in strategy_summary['hedge_detail'].items():
                print(f"  {symbol}_FUTURE: {quantity} 手")
        
        # 显示最近的交易记录
        trade_history = trading_engine.get_trade_history()
        if not trade_history.empty:
            print(f"\n=== 最近10笔交易 ===")
            recent_trades = trade_history.tail(10)
            for _, trade in recent_trades.iterrows():
                print(f"{trade['timestamp'].strftime('%Y-%m-%d')}: "
                      f"{trade['action']} {trade['quantity']} {trade['symbol']} "
                      f"@ ${trade['price']:.2f} [{trade['strategy']}]")
        
        # 绘制图表（如果可用）
        try:
            plot_backtest_results(results_df)
            print(f"\n图表已保存为 'iv_straddle_backtest_results.png'")
        except ImportError:
            print("\nMatplotlib未安装，跳过图表绘制")
        
        print(f"\n=== 回测完成 ===")
        
    except Exception as e:
        print(f"回测执行出错: {e}")
        import traceback
        traceback.print_exc()