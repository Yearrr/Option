#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的IV宽跨式期权策略测试
演示基于隐含波动率分位数的宽跨式期权策略与Delta对冲
"""

import sys
sys.path.append('options_backtest_framework')

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from trading_engine import TradingEngine

class SimpleIVCalculator:
    """简化的IV分位数计算器"""
    
    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self.iv_history = {}
    
    def update_iv(self, symbol: str, date: datetime, iv: float):
        """更新IV历史数据"""
        if symbol not in self.iv_history:
            self.iv_history[symbol] = []
        
        self.iv_history[symbol].append({
            'date': date,
            'iv': iv
        })
        
        # 只保留lookback_days的数据
        if len(self.iv_history[symbol]) > self.lookback_days:
            self.iv_history[symbol] = self.iv_history[symbol][-self.lookback_days:]
    
    def get_iv_percentile(self, symbol: str) -> Optional[float]:
        """计算当前IV的历史分位数"""
        if symbol not in self.iv_history or len(self.iv_history[symbol]) < 10:
            return None
        
        iv_values = [item['iv'] for item in self.iv_history[symbol]]
        current_iv = iv_values[-1]
        
        # 计算分位数
        count_below = sum(1 for iv in iv_values if iv <= current_iv)
        percentile = (count_below / len(iv_values)) * 100
        
        return percentile

class SimpleStraddleStrategy:
    """简化的宽跨式期权策略"""
    
    def __init__(self, trading_engine, iv_calculator, 
                 iv_percentile_threshold: float = 85.0,
                 delta_hedge_threshold: float = 0.15,
                 position_size: int = 5):
        self.trading_engine = trading_engine
        self.iv_calculator = iv_calculator
        self.iv_percentile_threshold = iv_percentile_threshold
        self.delta_hedge_threshold = delta_hedge_threshold
        self.position_size = position_size
        
        # 策略状态
        self.active_positions = {}  # symbol -> position_info
        self.hedge_positions = {}   # symbol -> hedge_quantity
        
    def check_entry_signal(self, underlying_symbol: str, market_data: Dict) -> bool:
        """检查入场信号"""
        # 检查是否已有该标的的持仓
        if underlying_symbol in self.active_positions:
            return False
        
        # 获取IV分位数
        iv_percentile = self.iv_calculator.get_iv_percentile(underlying_symbol)
        if iv_percentile is None or iv_percentile < self.iv_percentile_threshold:
            return False
        
        print(f"{underlying_symbol}: IV分位数 {iv_percentile:.1f}% >= {self.iv_percentile_threshold}%，触发卖出信号")
        return True
    
    def open_straddle_position(self, underlying_symbol: str, market_data: Dict, 
                              current_date: datetime) -> bool:
        """开立宽跨式期权仓位"""
        try:
            # 获取期权链数据
            call_symbol = f"{underlying_symbol}_CALL_ATM"
            put_symbol = f"{underlying_symbol}_PUT_ATM"
            
            call_data = market_data.get(call_symbol, {})
            put_data = market_data.get(put_symbol, {})
            
            if not call_data or not put_data:
                print(f"缺少期权数据: {call_symbol} 或 {put_symbol}")
                return False
            
            # 卖出看涨期权
            call_order_id = self.trading_engine.place_order(
                call_symbol, "sell", self.position_size,
                strategy="iv_straddle", instrument_type="option"
            )
            call_order = self.trading_engine.orders[-1]
            call_success = self.trading_engine.execute_order(
                call_order, call_data['price'], call_data.get('greeks', {})
            )
            
            # 卖出看跌期权
            put_order_id = self.trading_engine.place_order(
                put_symbol, "sell", self.position_size,
                strategy="iv_straddle", instrument_type="option"
            )
            put_order = self.trading_engine.orders[-1]
            put_success = self.trading_engine.execute_order(
                put_order, put_data['price'], put_data.get('greeks', {})
            )
            
            if call_success and put_success:
                # 记录策略持仓
                self.active_positions[underlying_symbol] = {
                    'call_symbol': call_symbol,
                    'put_symbol': put_symbol,
                    'entry_date': current_date,
                    'position_size': self.position_size,
                    'entry_call_price': call_data['price'],
                    'entry_put_price': put_data['price'],
                    'total_premium': (call_data['price'] + put_data['price']) * self.position_size * 100
                }
                
                print(f"成功开立宽跨式期权仓位:")
                print(f"  卖出 {self.position_size} 手 {call_symbol} @ ${call_data['price']:.2f}")
                print(f"  卖出 {self.position_size} 手 {put_symbol} @ ${put_data['price']:.2f}")
                print(f"  收到总权利金: ${self.active_positions[underlying_symbol]['total_premium']:,.2f}")
                
                return True
            else:
                print(f"期权订单执行失败")
                return False
                
        except Exception as e:
            print(f"开立宽跨式仓位失败: {e}")
            return False
    
    def check_delta_hedge(self, underlying_symbol: str, market_data: Dict) -> Optional[int]:
        """检查是否需要Delta对冲"""
        if underlying_symbol not in self.active_positions:
            return None
        
        position_info = self.active_positions[underlying_symbol]
        call_symbol = position_info['call_symbol']
        put_symbol = position_info['put_symbol']
        
        # 获取当前Greeks
        call_data = market_data.get(call_symbol, {})
        put_data = market_data.get(put_symbol, {})
        
        if not call_data or not put_data:
            return None
        
        call_greeks = call_data.get('greeks', {})
        put_greeks = put_data.get('greeks', {})
        
        call_delta = call_greeks.get('delta', 0)
        put_delta = put_greeks.get('delta', 0)
        
        # 计算组合Delta（卖出期权，所以要取负值）
        portfolio_delta = -(call_delta + put_delta) * self.position_size
        
        print(f"{underlying_symbol} 组合Delta: {portfolio_delta:.3f}")
        
        # 检查是否超过阈值
        if abs(portfolio_delta) > self.delta_hedge_threshold:
            # 计算需要的期货对冲数量
            hedge_quantity = -int(round(portfolio_delta))  # 期货数量取整
            
            print(f"Delta {portfolio_delta:.3f} 超过阈值 {self.delta_hedge_threshold}")
            print(f"需要{hedge_quantity}手期货对冲")
            
            return hedge_quantity
        
        return None
    
    def execute_delta_hedge(self, underlying_symbol: str, hedge_quantity: int,
                           market_data: Dict) -> bool:
        """执行Delta对冲"""
        try:
            future_symbol = f"{underlying_symbol}_FUTURE"
            future_data = market_data.get(future_symbol, {})
            
            if not future_data:
                print(f"缺少期货数据: {future_symbol}")
                return False
            
            # 计算当前对冲仓位
            current_hedge = self.hedge_positions.get(underlying_symbol, 0)
            net_hedge_quantity = hedge_quantity - current_hedge
            
            if net_hedge_quantity == 0:
                return True
            
            # 执行期货对冲交易
            action = "buy" if net_hedge_quantity > 0 else "sell"
            quantity = abs(net_hedge_quantity)
            
            hedge_order_id = self.trading_engine.place_order(
                future_symbol, action, quantity,
                strategy="delta_hedge", instrument_type="future", 
                contract_multiplier=1
            )
            hedge_order = self.trading_engine.orders[-1]
            hedge_success = self.trading_engine.execute_order(
                hedge_order, future_data['price']
            )
            
            if hedge_success:
                self.hedge_positions[underlying_symbol] = hedge_quantity
                print(f"Delta对冲执行成功: {action} {quantity} 手 {future_symbol} @ ${future_data['price']:.2f}")
                return True
            else:
                print(f"Delta对冲执行失败")
                return False
                
        except Exception as e:
            print(f"Delta对冲执行错误: {e}")
            return False
    
    def check_exit_signal(self, underlying_symbol: str, market_data: Dict,
                         current_date: datetime) -> bool:
        """检查出场信号"""
        if underlying_symbol not in self.active_positions:
            return False
        
        position_info = self.active_positions[underlying_symbol]
        
        # 简单的出场规则：持有20天后平仓
        days_held = (current_date - position_info['entry_date']).days
        if days_held >= 20:
            print(f"{underlying_symbol}: 持有{days_held}天，触发平仓信号")
            return True
        
        return False
    
    def close_straddle_position(self, underlying_symbol: str, market_data: Dict) -> bool:
        """平仓宽跨式期权仓位"""
        try:
            if underlying_symbol not in self.active_positions:
                return False
            
            position_info = self.active_positions[underlying_symbol]
            call_symbol = position_info['call_symbol']
            put_symbol = position_info['put_symbol']
            
            call_data = market_data.get(call_symbol, {})
            put_data = market_data.get(put_symbol, {})
            
            if not call_data or not put_data:
                print(f"缺少期权数据，无法平仓")
                return False
            
            # 买回看涨期权
            call_order_id = self.trading_engine.place_order(
                call_symbol, "buy", self.position_size,
                strategy="iv_straddle_close", instrument_type="option"
            )
            call_order = self.trading_engine.orders[-1]
            call_success = self.trading_engine.execute_order(
                call_order, call_data['price']
            )
            
            # 买回看跌期权
            put_order_id = self.trading_engine.place_order(
                put_symbol, "buy", self.position_size,
                strategy="iv_straddle_close", instrument_type="option"
            )
            put_order = self.trading_engine.orders[-1]
            put_success = self.trading_engine.execute_order(
                put_order, put_data['price']
            )
            
            # 平仓期货对冲仓位
            if underlying_symbol in self.hedge_positions:
                hedge_quantity = self.hedge_positions[underlying_symbol]
                if hedge_quantity != 0:
                    future_symbol = f"{underlying_symbol}_FUTURE"
                    future_data = market_data.get(future_symbol, {})
                    
                    if future_data:
                        action = "sell" if hedge_quantity > 0 else "buy"
                        quantity = abs(hedge_quantity)
                        
                        hedge_close_order_id = self.trading_engine.place_order(
                            future_symbol, action, quantity,
                            strategy="hedge_close", instrument_type="future",
                            contract_multiplier=1
                        )
                        hedge_close_order = self.trading_engine.orders[-1]
                        self.trading_engine.execute_order(
                            hedge_close_order, future_data['price']
                        )
                        
                        print(f"平仓期货对冲: {action} {quantity} 手")
                
                del self.hedge_positions[underlying_symbol]
            
            if call_success and put_success:
                # 计算盈亏
                exit_cost = (call_data['price'] + put_data['price']) * self.position_size * 100
                profit = position_info['total_premium'] - exit_cost
                
                print(f"宽跨式期权平仓完成:")
                print(f"  买回 {self.position_size} 手 {call_symbol} @ ${call_data['price']:.2f}")
                print(f"  买回 {self.position_size} 手 {put_symbol} @ ${put_data['price']:.2f}")
                print(f"  平仓成本: ${exit_cost:,.2f}")
                print(f"  期权盈亏: ${profit:,.2f}")
                
                del self.active_positions[underlying_symbol]
                return True
            else:
                print(f"期权平仓失败")
                return False
                
        except Exception as e:
            print(f"平仓宽跨式仓位失败: {e}")
            return False
    
    def run_strategy(self, underlying_symbol: str, market_data: Dict, 
                    current_date: datetime):
        """运行策略主逻辑"""
        # 更新IV历史
        underlying_data = market_data.get(underlying_symbol, {})
        if underlying_data and 'iv' in underlying_data:
            self.iv_calculator.update_iv(underlying_symbol, current_date, 
                                       underlying_data['iv'])
        
        # 检查平仓信号
        if self.check_exit_signal(underlying_symbol, market_data, current_date):
            self.close_straddle_position(underlying_symbol, market_data)
        
        # 检查开仓信号
        elif self.check_entry_signal(underlying_symbol, market_data):
            self.open_straddle_position(underlying_symbol, market_data, current_date)
        
        # 检查Delta对冲
        hedge_quantity = self.check_delta_hedge(underlying_symbol, market_data)
        if hedge_quantity is not None:
            self.execute_delta_hedge(underlying_symbol, hedge_quantity, market_data)

def generate_simple_market_data():
    """生成简单的市场数据"""
    
    # 创建50天的模拟数据
    start_date = datetime(2024, 1, 1)
    data = []
    
    # 基础参数
    base_price = 150.0
    base_iv = 0.25
    
    for i in range(50):
        current_date = start_date + timedelta(days=i)
        
        # 价格变化（简单随机游走）
        price_change = (i % 7 - 3) * 2  # 简单的周期性变化
        price = base_price + price_change
        
        # IV变化（模拟波动率聚集）
        if i < 10:
            iv = base_iv + 0.1  # 开始时IV较高
        elif i < 20:
            iv = base_iv + 0.2  # 中期IV进一步升高
        elif i < 30:
            iv = base_iv + 0.05  # 后期IV回落
        else:
            iv = base_iv  # 恢复正常水平
        
        data.append({
            'date': current_date,
            'price': price,
            'iv': iv
        })
    
    return data

def calculate_simple_option_price(S, K, T, r, sigma, option_type='call'):
    """简化的期权定价（无需外部库）"""
    
    # 简化的Black-Scholes近似
    # 这里使用简化公式，实际中应使用更精确的计算
    
    moneyness = S / K
    time_value = max(0.01, T * sigma * math.sqrt(T))
    
    if option_type == 'call':
        intrinsic = max(0, S - K)
        price = intrinsic + time_value * (1 + (moneyness - 1) * 0.5)
        delta = 0.5 + (moneyness - 1) * 0.3  # 简化Delta
    else:  # put
        intrinsic = max(0, K - S)
        price = intrinsic + time_value * (1 - (moneyness - 1) * 0.5)
        delta = -0.5 + (moneyness - 1) * 0.3  # 简化Delta
    
    # 确保价格不为负且有最小值
    price = max(0.01, price)
    
    return {
        'price': price,
        'greeks': {
            'delta': delta,
            'gamma': 0.05,  # 简化Gamma
            'theta': -0.02,  # 简化Theta
            'vega': 0.1   # 简化Vega
        }
    }

def run_simple_backtest():
    """运行简化的回测"""
    
    print("=== 简化IV宽跨式期权策略回测 ===\n")
    
    # 初始化
    initial_cash = 100000
    underlying_symbol = "AAPL"
    
    trading_engine = TradingEngine(initial_cash=initial_cash)
    iv_calculator = SimpleIVCalculator(lookback_days=30)
    strategy = SimpleStraddleStrategy(
        trading_engine=trading_engine,
        iv_calculator=iv_calculator,
        iv_percentile_threshold=80.0,  # 降低阈值以便触发
        delta_hedge_threshold=0.20,     # Delta对冲阈值
        position_size=3                 # 每次交易3手
    )
    
    print(f"初始资金: ${initial_cash:,.2f}")
    print(f"策略参数:")
    print(f"  IV分位数阈值: {strategy.iv_percentile_threshold}%")
    print(f"  Delta对冲阈值: {strategy.delta_hedge_threshold}")
    print(f"  头寸大小: {strategy.position_size} 手")
    
    # 生成市场数据
    market_data_list = generate_simple_market_data()
    print(f"\n生成了 {len(market_data_list)} 天的模拟数据")
    
    # 回测循环
    print(f"\n开始回测...")
    portfolio_values = []
    
    for i, data in enumerate(market_data_list):
        current_date = data['date']
        price = data['price']
        iv = data['iv']
        
        trading_engine.current_date = current_date
        
        # 生成期权数据
        call_data = calculate_simple_option_price(price, price, 30/365, 0.05, iv, 'call')
        put_data = calculate_simple_option_price(price, price, 30/365, 0.05, iv, 'put')
        
        # 构建市场数据
        market_data = {
            underlying_symbol: {
                'price': price,
                'iv': iv
            },
            f"{underlying_symbol}_CALL_ATM": call_data,
            f"{underlying_symbol}_PUT_ATM": put_data,
            f"{underlying_symbol}_FUTURE": {
                'price': price * 1.01  # 期货略有升水
            }
        }
        
        # 更新持仓价格
        trading_engine.update_positions(market_data)
        
        # 运行策略
        strategy.run_strategy(underlying_symbol, market_data, current_date)
        
        # 记录投资组合价值
        portfolio_value = trading_engine.get_portfolio_value()
        portfolio_values.append(portfolio_value)
        
        # 每10天输出一次进度
        if i % 10 == 0 or i == len(market_data_list) - 1:
            iv_pct = iv_calculator.get_iv_percentile(underlying_symbol)
            iv_pct_str = f"{iv_pct:.1f}%" if iv_pct else "N/A"
            breakdown = trading_engine.get_portfolio_breakdown()
            print(f"  第{i+1}天 {current_date.strftime('%Y-%m-%d')}: 价格=${price:.2f}, "
                  f"IV={iv:.2f}({iv_pct_str}), 组合=${portfolio_value:,.0f}, "
                  f"现金=${breakdown['cash']:,.0f}")
    
    # 分析结果
    print(f"\n=== 回测结果分析 ===")
    
    final_value = portfolio_values[-1]
    total_return = final_value - initial_cash
    return_pct = total_return / initial_cash * 100
    
    print(f"初始资金: ${initial_cash:,.2f}")
    print(f"最终价值: ${final_value:,.2f}")
    print(f"总收益: ${total_return:,.2f}")
    print(f"收益率: {return_pct:.2f}%")
    
    # 交易统计
    trade_history = trading_engine.get_trade_history()
    if not trade_history.empty:
        option_trades = trade_history[trade_history['strategy'].str.contains('straddle', na=False)]
        hedge_trades = trade_history[trade_history['strategy'].str.contains('hedge', na=False)]
        
        print(f"总交易数: {len(trade_history)}")
        print(f"期权交易: {len(option_trades)}")
        print(f"对冲交易: {len(hedge_trades)}")
        
        print(f"\n=== 最近交易记录 ===")
        recent_trades = trade_history.tail(8)
        for _, trade in recent_trades.iterrows():
            print(f"{trade['timestamp'].strftime('%Y-%m-%d')}: "
                  f"{trade['action']} {trade['quantity']} {trade['symbol']} "
                  f"@ ${trade['price']:.2f} [{trade['strategy']}]")
    
    # 持仓摘要
    print(f"\n=== 当前持仓 ===")
    positions = trading_engine.get_position_summary()
    if not positions.empty:
        for _, pos in positions.iterrows():
            print(f"{pos['symbol']}: {pos['position_type']} {pos['quantity']} 手, "
                  f"价值${pos['current_market_value']:,.2f}, "
                  f"PnL${pos['unrealized_pnl']:,.2f}")
    else:
        print("无持仓")
    
    breakdown = trading_engine.get_portfolio_breakdown()
    print(f"\n=== 投资组合分解 ===")
    print(f"现金: ${breakdown['cash']:,.2f}")
    print(f"期权价值: ${breakdown['options_value']:,.2f}")
    print(f"期货价值: ${breakdown['futures_value']:,.2f}")
    print(f"总价值: ${breakdown['total_portfolio_value']:,.2f}")
    
    return trading_engine, strategy

if __name__ == "__main__":
    try:
        trading_engine, strategy = run_simple_backtest()
        print(f"\n=== 回测完成 ===")
        
    except Exception as e:
        print(f"回测执行出错: {e}")
        import traceback
        traceback.print_exc()