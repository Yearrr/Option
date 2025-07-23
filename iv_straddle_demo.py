#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IV宽跨式期权策略演示
基于隐含波动率分位数的宽跨式期权策略与Delta对冲
完全独立实现，不依赖外部库
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class PortfolioTracker:
    """简化的投资组合跟踪器"""
    
    def __init__(self, initial_cash: float = 100000):
        self.cash = initial_cash
        self.positions = {}  # symbol -> {'quantity': int, 'entry_price': float, 'current_price': float}
        self.trades = []
        
    def execute_trade(self, symbol: str, action: str, quantity: int, price: float, 
                     instrument_type: str = "option"):
        """执行交易"""
        
        # 计算现金流
        if instrument_type == "option":
            if action == "buy":
                cash_flow = -(quantity * price * 100)  # 支付权利金
            else:  # sell
                cash_flow = quantity * price * 100     # 收到权利金
        else:  # future
            cash_flow = 0  # 期货只支付保证金，简化处理
        
        self.cash += cash_flow
        
        # 更新持仓
        if symbol in self.positions:
            old_quantity = self.positions[symbol]['quantity']
            if action == "buy":
                new_quantity = old_quantity + quantity
            else:  # sell
                new_quantity = old_quantity - quantity
            
            if new_quantity == 0:
                del self.positions[symbol]
            else:
                self.positions[symbol]['quantity'] = new_quantity
        else:
            if action == "buy":
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': price,
                    'current_price': price,
                    'instrument_type': instrument_type
                }
            else:  # sell
                self.positions[symbol] = {
                    'quantity': -quantity,
                    'entry_price': price,
                    'current_price': price,
                    'instrument_type': instrument_type
                }
        
        # 记录交易
        self.trades.append({
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'cash_flow': cash_flow,
            'instrument_type': instrument_type
        })
        
        return True
    
    def update_prices(self, market_data: Dict):
        """更新持仓价格"""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position['current_price'] = market_data[symbol]['price']
    
    def get_portfolio_value(self) -> float:
        """计算投资组合总价值"""
        positions_value = 0
        for symbol, position in self.positions.items():
            if position['instrument_type'] == "option":
                multiplier = 100
            else:
                multiplier = 1
            
            market_value = position['quantity'] * position['current_price'] * multiplier
            positions_value += market_value
        
        return self.cash + positions_value
    
    def get_unrealized_pnl(self) -> float:
        """计算未实现盈亏"""
        total_pnl = 0
        for symbol, position in self.positions.items():
            if position['instrument_type'] == "option":
                multiplier = 100
            else:
                multiplier = 1
            
            pnl = (position['current_price'] - position['entry_price']) * position['quantity'] * multiplier
            total_pnl += pnl
        
        return total_pnl

class IVStraddleStrategy:
    """IV宽跨式期权策略演示"""
    
    def __init__(self, portfolio_tracker, iv_percentile_threshold=85.0, 
                 delta_hedge_threshold=0.15, position_size=3):
        self.portfolio = portfolio_tracker
        self.iv_percentile_threshold = iv_percentile_threshold
        self.delta_hedge_threshold = delta_hedge_threshold
        self.position_size = position_size
        
        # IV历史数据
        self.iv_history = []
        self.lookback_days = 30
        
        # 策略状态
        self.active_positions = {}  # underlying -> position_info
        self.hedge_positions = {}   # underlying -> hedge_quantity
        
    def update_iv_history(self, iv: float):
        """更新IV历史"""
        self.iv_history.append(iv)
        if len(self.iv_history) > self.lookback_days:
            self.iv_history = self.iv_history[-self.lookback_days:]
    
    def get_iv_percentile(self) -> Optional[float]:
        """计算IV分位数"""
        if len(self.iv_history) < 10:
            return None
        
        current_iv = self.iv_history[-1]
        count_below = sum(1 for iv in self.iv_history if iv <= current_iv)
        percentile = (count_below / len(self.iv_history)) * 100
        
        return percentile
    
    def check_entry_signal(self, underlying: str) -> bool:
        """检查入场信号"""
        if underlying in self.active_positions:
            return False
        
        iv_percentile = self.get_iv_percentile()
        if iv_percentile is None or iv_percentile < self.iv_percentile_threshold:
            return False
        
        print(f"{underlying}: IV分位数 {iv_percentile:.1f}% >= {self.iv_percentile_threshold}%，触发卖出信号")
        return True
    
    def open_straddle(self, underlying: str, call_price: float, put_price: float, 
                     current_date: datetime) -> bool:
        """开立宽跨式期权仓位"""
        
        call_symbol = f"{underlying}_CALL_ATM"
        put_symbol = f"{underlying}_PUT_ATM"
        
        # 卖出看涨期权
        success1 = self.portfolio.execute_trade(
            call_symbol, "sell", self.position_size, call_price, "option"
        )
        
        # 卖出看跌期权
        success2 = self.portfolio.execute_trade(
            put_symbol, "sell", self.position_size, put_price, "option"
        )
        
        if success1 and success2:
            total_premium = (call_price + put_price) * self.position_size * 100
            
            self.active_positions[underlying] = {
                'call_symbol': call_symbol,
                'put_symbol': put_symbol,
                'entry_date': current_date,
                'position_size': self.position_size,
                'entry_call_price': call_price,
                'entry_put_price': put_price,
                'total_premium': total_premium
            }
            
            print(f"成功开立宽跨式期权仓位:")
            print(f"  卖出 {self.position_size} 手 {call_symbol} @ ${call_price:.2f}")
            print(f"  卖出 {self.position_size} 手 {put_symbol} @ ${put_price:.2f}")
            print(f"  收到总权利金: ${total_premium:,.2f}")
            
            return True
        
        return False
    
    def check_delta_hedge(self, underlying: str, call_delta: float, put_delta: float) -> Optional[int]:
        """检查Delta对冲需求"""
        if underlying not in self.active_positions:
            return None
        
        # 计算组合Delta（卖出期权，取负值）
        portfolio_delta = -(call_delta + put_delta) * self.position_size
        
        print(f"{underlying} 组合Delta: {portfolio_delta:.3f}")
        
        if abs(portfolio_delta) > self.delta_hedge_threshold:
            hedge_quantity = -int(round(portfolio_delta))
            print(f"Delta {portfolio_delta:.3f} 超过阈值 {self.delta_hedge_threshold}")
            print(f"需要{hedge_quantity}手期货对冲")
            return hedge_quantity
        
        return None
    
    def execute_hedge(self, underlying: str, hedge_quantity: int, future_price: float) -> bool:
        """执行期货对冲"""
        
        future_symbol = f"{underlying}_FUTURE"
        current_hedge = self.hedge_positions.get(underlying, 0)
        net_hedge = hedge_quantity - current_hedge
        
        if net_hedge == 0:
            return True
        
        action = "buy" if net_hedge > 0 else "sell"
        quantity = abs(net_hedge)
        
        success = self.portfolio.execute_trade(
            future_symbol, action, quantity, future_price, "future"
        )
        
        if success:
            self.hedge_positions[underlying] = hedge_quantity
            print(f"Delta对冲执行成功: {action} {quantity} 手 {future_symbol} @ ${future_price:.2f}")
            return True
        
        return False
    
    def check_exit_signal(self, underlying: str, current_date: datetime) -> bool:
        """检查出场信号"""
        if underlying not in self.active_positions:
            return False
        
        position_info = self.active_positions[underlying]
        days_held = (current_date - position_info['entry_date']).days
        
        if days_held >= 20:
            print(f"{underlying}: 持有{days_held}天，触发平仓信号")
            return True
        
        return False
    
    def close_straddle(self, underlying: str, call_price: float, put_price: float) -> bool:
        """平仓宽跨式期权"""
        
        if underlying not in self.active_positions:
            return False
        
        position_info = self.active_positions[underlying]
        call_symbol = position_info['call_symbol']
        put_symbol = position_info['put_symbol']
        
        # 买回看涨期权
        success1 = self.portfolio.execute_trade(
            call_symbol, "buy", self.position_size, call_price, "option"
        )
        
        # 买回看跌期权
        success2 = self.portfolio.execute_trade(
            put_symbol, "buy", self.position_size, put_price, "option"
        )
        
        # 平仓期货对冲
        if underlying in self.hedge_positions:
            hedge_quantity = self.hedge_positions[underlying]
            if hedge_quantity != 0:
                future_symbol = f"{underlying}_FUTURE"
                action = "sell" if hedge_quantity > 0 else "buy"
                quantity = abs(hedge_quantity)
                
                # 假设期货价格与标的相近
                future_price = call_price + put_price  # 简化处理
                
                self.portfolio.execute_trade(
                    future_symbol, action, quantity, future_price, "future"
                )
                
                print(f"平仓期货对冲: {action} {quantity} 手")
            
            del self.hedge_positions[underlying]
        
        if success1 and success2:
            exit_cost = (call_price + put_price) * self.position_size * 100
            profit = position_info['total_premium'] - exit_cost
            
            print(f"宽跨式期权平仓完成:")
            print(f"  买回 {self.position_size} 手 {call_symbol} @ ${call_price:.2f}")
            print(f"  买回 {self.position_size} 手 {put_symbol} @ ${put_price:.2f}")
            print(f"  平仓成本: ${exit_cost:,.2f}")
            print(f"  期权盈亏: ${profit:,.2f}")
            
            del self.active_positions[underlying]
            return True
        
        return False

def calculate_option_price_and_greeks(S, K, T, r, sigma, option_type='call'):
    """简化的期权定价和Greeks计算"""
    
    # 简化的Black-Scholes近似
    moneyness = S / K
    time_value = max(0.01, T * sigma * math.sqrt(T))
    
    if option_type == 'call':
        intrinsic = max(0, S - K)
        price = intrinsic + time_value * (1 + (moneyness - 1) * 0.5)
        delta = 0.5 + (moneyness - 1) * 0.3
    else:  # put
        intrinsic = max(0, K - S)
        price = intrinsic + time_value * (1 - (moneyness - 1) * 0.5)
        delta = -0.5 + (moneyness - 1) * 0.3
    
    price = max(0.01, price)
    
    return price, delta

def run_strategy_demo():
    """运行策略演示"""
    
    print("=== IV宽跨式期权策略演示 ===\n")
    
    # 初始化
    initial_cash = 100000
    portfolio = PortfolioTracker(initial_cash)
    strategy = IVStraddleStrategy(
        portfolio, 
        iv_percentile_threshold=80.0,  # 降低阈值便于触发
        delta_hedge_threshold=0.20,
        position_size=3
    )
    
    print(f"初始资金: ${initial_cash:,.2f}")
    print(f"策略参数:")
    print(f"  IV分位数阈值: {strategy.iv_percentile_threshold}%")
    print(f"  Delta对冲阈值: {strategy.delta_hedge_threshold}")
    print(f"  头寸大小: {strategy.position_size} 手")
    
    # 模拟50天的交易
    start_date = datetime(2024, 1, 1)
    underlying = "AAPL"
    base_price = 150.0
    base_iv = 0.25
    
    print(f"\n开始模拟交易...")
    
    for day in range(50):
        current_date = start_date + timedelta(days=day)
        
        # 生成市场数据
        price_change = (day % 7 - 3) * 2
        price = base_price + price_change
        
        # IV变化模拟
        if day < 10:
            iv = base_iv + 0.1
        elif day < 20:
            iv = base_iv + 0.2  # IV高峰期
        elif day < 30:
            iv = base_iv + 0.05
        else:
            iv = base_iv
        
        # 更新IV历史
        strategy.update_iv_history(iv)
        
        # 计算期权价格和Greeks
        call_price, call_delta = calculate_option_price_and_greeks(
            price, price, 30/365, 0.05, iv, 'call'
        )
        put_price, put_delta = calculate_option_price_and_greeks(
            price, price, 30/365, 0.05, iv, 'put'
        )
        
        # 构建市场数据
        market_data = {
            f"{underlying}_CALL_ATM": {'price': call_price},
            f"{underlying}_PUT_ATM": {'price': put_price},
            f"{underlying}_FUTURE": {'price': price * 1.01}
        }
        
        # 更新持仓价格
        portfolio.update_prices(market_data)
        
        # 策略决策
        # 1. 检查出场信号
        if strategy.check_exit_signal(underlying, current_date):
            strategy.close_straddle(underlying, call_price, put_price)
        
        # 2. 检查入场信号
        elif strategy.check_entry_signal(underlying):
            strategy.open_straddle(underlying, call_price, put_price, current_date)
        
        # 3. 检查Delta对冲
        hedge_quantity = strategy.check_delta_hedge(underlying, call_delta, put_delta)
        if hedge_quantity is not None:
            future_price = price * 1.01
            strategy.execute_hedge(underlying, hedge_quantity, future_price)
        
        # 记录状态
        portfolio_value = portfolio.get_portfolio_value()
        unrealized_pnl = portfolio.get_unrealized_pnl()
        
        # 每10天输出一次状态
        if day % 10 == 0 or day == 49:
            iv_pct = strategy.get_iv_percentile()
            iv_pct_str = f"{iv_pct:.1f}%" if iv_pct else "N/A"
            
            print(f"\n第{day+1}天 ({current_date.strftime('%Y-%m-%d')}):")
            print(f"  标的价格: ${price:.2f}")
            print(f"  隐含波动率: {iv:.2f} (分位数: {iv_pct_str})")
            print(f"  看涨期权: ${call_price:.2f} (Delta: {call_delta:.3f})")
            print(f"  看跌期权: ${put_price:.2f} (Delta: {put_delta:.3f})")
            print(f"  投资组合价值: ${portfolio_value:,.2f}")
            print(f"  现金: ${portfolio.cash:,.2f}")
            print(f"  未实现盈亏: ${unrealized_pnl:,.2f}")
            print(f"  活跃策略仓位: {len(strategy.active_positions)}")
            print(f"  对冲仓位: {len(strategy.hedge_positions)}")
    
    # 最终结果
    print(f"\n=== 策略演示结果 ===")
    
    final_value = portfolio.get_portfolio_value()
    total_return = final_value - initial_cash
    return_pct = total_return / initial_cash * 100
    
    print(f"初始资金: ${initial_cash:,.2f}")
    print(f"最终价值: ${final_value:,.2f}")
    print(f"总收益: ${total_return:,.2f}")
    print(f"收益率: {return_pct:.2f}%")
    print(f"总交易数: {len(portfolio.trades)}")
    
    # 交易明细
    print(f"\n=== 交易记录 ===")
    option_trades = [t for t in portfolio.trades if t['instrument_type'] == 'option']
    future_trades = [t for t in portfolio.trades if t['instrument_type'] == 'future']
    
    print(f"期权交易: {len(option_trades)} 笔")
    print(f"期货交易: {len(future_trades)} 笔")
    
    if len(portfolio.trades) > 0:
        print(f"\n最近交易:")
        recent_trades = portfolio.trades[-8:] if len(portfolio.trades) >= 8 else portfolio.trades
        for i, trade in enumerate(recent_trades, 1):
            print(f"  {i}. {trade['action'].upper()} {trade['quantity']} {trade['symbol']} "
                  f"@ ${trade['price']:.2f} (现金流: ${trade['cash_flow']:,.2f})")
    
    # 当前持仓
    print(f"\n=== 当前持仓 ===")
    if portfolio.positions:
        for symbol, pos in portfolio.positions.items():
            multiplier = 100 if pos['instrument_type'] == 'option' else 1
            market_value = pos['quantity'] * pos['current_price'] * multiplier
            pnl = (pos['current_price'] - pos['entry_price']) * pos['quantity'] * multiplier
            
            print(f"{symbol}: {pos['quantity']} 手")
            print(f"  类型: {pos['instrument_type']}")
            print(f"  入场价: ${pos['entry_price']:.2f}")
            print(f"  当前价: ${pos['current_price']:.2f}")
            print(f"  市值: ${market_value:,.2f}")
            print(f"  盈亏: ${pnl:,.2f}")
    else:
        print("无持仓")
    
    print(f"\n=== 策略演示完成 ===")
    
    return portfolio, strategy

if __name__ == "__main__":
    try:
        portfolio, strategy = run_strategy_demo()
        
        print(f"\n本演示展示了以下功能:")
        print(f"✓ IV分位数计算和高IV信号识别")
        print(f"✓ 宽跨式期权开仓和平仓")
        print(f"✓ Delta中性对冲机制")
        print(f"✓ 完整的投资组合管理")
        print(f"✓ 期权和期货的混合交易")
        
    except Exception as e:
        print(f"演示运行出错: {e}")
        import traceback
        traceback.print_exc()