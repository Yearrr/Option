#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于隐含波动率分位数的宽跨式期权策略
当IV达到历史分位数85%以上时卖出宽跨式期权
当Delta达到阈值时进行期货对冲
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class IVPercentileCalculator:
    """隐含波动率分位数计算器"""
    
    def __init__(self, lookback_days: int = 252):
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
        if symbol not in self.iv_history or len(self.iv_history[symbol]) < 20:
            return None
        
        iv_values = [item['iv'] for item in self.iv_history[symbol]]
        current_iv = iv_values[-1]
        
        # 计算分位数
        percentile = (np.sum(np.array(iv_values) <= current_iv) / len(iv_values)) * 100
        
        return percentile

class StraddleStrategy:
    """宽跨式期权策略"""
    
    def __init__(self, trading_engine, iv_calculator, 
                 iv_percentile_threshold: float = 85.0,
                 delta_hedge_threshold: float = 0.15,
                 position_size: int = 10):
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
            future_symbol = f"{underlying_symbol}_FUTURE"
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
        
        # 简单的出场规则：持有30天后平仓
        days_held = (current_date - position_info['entry_date']).days
        if days_held >= 30:
            print(f"{underlying_symbol}: 持有{days_held}天，触发平仓信号")
            return True
        
        # 其他出场条件可以在这里添加
        # 例如：盈利目标、止损等
        
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
    
    def get_strategy_summary(self) -> Dict:
        """获取策略摘要"""
        summary = {
            'active_positions': len(self.active_positions),
            'hedge_positions': len(self.hedge_positions),
            'positions_detail': self.active_positions.copy(),
            'hedge_detail': self.hedge_positions.copy()
        }
        return summary