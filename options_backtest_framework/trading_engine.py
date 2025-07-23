#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权回测框架 - 交易引擎模块
负责订单执行、持仓管理和交易记录
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import uuid
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')


class Order:
    """订单类"""
    
    def __init__(self, order_id: str, symbol: str, action: str, quantity: int,
                 order_type: str = "market", price: float = None, 
                 strategy: str = None, timestamp: datetime = None,
                 instrument_type: str = "option", contract_multiplier: int = None):
        self.order_id = order_id
        self.symbol = symbol
        self.action = action  # "buy" or "sell"
        self.quantity = quantity
        self.order_type = order_type  # "market", "limit", "stop"
        self.price = price
        self.strategy = strategy
        self.instrument_type = instrument_type.lower()
        self.contract_multiplier = contract_multiplier
        self.timestamp = timestamp or datetime.now()
        self.status = "pending"  # "pending", "filled", "canceled", "rejected"
        self.filled_price = None
        self.filled_quantity = 0
        self.commission = 0
        
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'action': self.action,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'price': self.price,
            'strategy': self.strategy,
            'instrument_type': self.instrument_type,
            'contract_multiplier': self.contract_multiplier,
            'timestamp': self.timestamp,
            'status': self.status,
            'filled_price': self.filled_price,
            'filled_quantity': self.filled_quantity,
            'commission': self.commission
        }


class Position:
    """持仓类"""
    
    def __init__(self, symbol: str, quantity: int, entry_price: float,
                 entry_date: datetime, strategy: str = None, 
                 instrument_type: str = "option", contract_multiplier: int = None):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.strategy = strategy
        self.instrument_type = instrument_type.lower()  # "option", "future", "stock", etc.
        
        # 设置默认合约乘数
        if contract_multiplier is None:
            if self.instrument_type == "option":
                self.contract_multiplier = 100  # 期权默认100
            elif self.instrument_type == "future":
                self.contract_multiplier = 1    # 期货默认1，具体需要根据品种设置
            else:
                self.contract_multiplier = 1    # 其他金融工具默认1
        else:
            self.contract_multiplier = contract_multiplier
            
        self.current_price = entry_price
        self.unrealized_pnl = 0
        self.realized_pnl = 0
        self.total_commission = 0
        self.greeks = {}
        self.position_id = str(uuid.uuid4())
        
    def update_price(self, new_price: float, greeks: Dict = None):
        """更新价格和Greeks"""
        self.current_price = new_price
        if greeks:
            self.greeks = greeks
        # 正确计算未实现盈亏：(当前价格 - 入场价格) * 持仓数量 * 合约乘数
        # 对于多头：quantity > 0，价格上涨盈利
        # 对于空头：quantity < 0，价格下跌盈利
        self.unrealized_pnl = (new_price - self.entry_price) * self.quantity * self.contract_multiplier
        
    def add_trade(self, quantity: int, price: float, commission: float = 0):
        """添加交易"""
        if self.quantity == 0:
            # 新开仓
            self.quantity = quantity
            self.entry_price = price
        else:
            # 加仓或减仓
            if np.sign(quantity) == np.sign(self.quantity):
                # 加仓
                total_value = self.quantity * self.entry_price + quantity * price
                self.quantity += quantity
                self.entry_price = total_value / self.quantity if self.quantity != 0 else 0
            else:
                # 减仓或平仓
                close_quantity = min(abs(quantity), abs(self.quantity))
                # 平仓盈亏 = (平仓价格 - 开仓价格) * 平仓数量 * 持仓方向 * 合约乘数
                self.realized_pnl += (price - self.entry_price) * close_quantity * np.sign(self.quantity) * self.contract_multiplier
                self.quantity += quantity
                
                if self.quantity == 0:
                    self.entry_price = 0
        
        self.total_commission += commission
        
    def to_dict(self) -> Dict:
        """转换为字典"""
        # 计算当前市值
        current_market_value = self.quantity * self.current_price * self.contract_multiplier
        # 计算总盈亏
        total_pnl = self.unrealized_pnl + self.realized_pnl
        # 计算现金流（入场时）
        cash_flow = self.quantity * self.entry_price * self.contract_multiplier
        
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'entry_date': self.entry_date,
            'strategy': self.strategy,
            'instrument_type': self.instrument_type,
            'contract_multiplier': self.contract_multiplier,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': total_pnl,
            'current_market_value': current_market_value,
            'cash_flow': cash_flow,
            'total_commission': self.total_commission,
            'greeks': self.greeks,
            'position_type': 'Long' if self.quantity > 0 else 'Short'
        }


class TradingEngine:
    """交易引擎 - 负责订单执行和持仓管理"""
    
    def __init__(self, initial_cash: float = 100000, commission_rate: float = 0.001):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.positions = {}  # symbol -> Position
        self.orders = []  # 订单历史
        self.trades = []  # 交易历史
        self.daily_portfolio = []  # 每日投资组合价值
        self.current_date = None
        
    def place_order(self, symbol: str, action: str, quantity: int,
                   order_type: str = "market", price: float = None,
                   strategy: str = None, instrument_type: str = "option",
                   contract_multiplier: int = None) -> str:
        """
        下单
        
        Args:
            symbol: 金融工具代码
            action: 买卖方向
            quantity: 数量
            order_type: 订单类型
            price: 价格
            strategy: 策略名称
            instrument_type: 金融工具类型 ("option", "future", "stock", 等)
            contract_multiplier: 合约乘数
            
        Returns:
            订单ID
        """
        order_id = str(uuid.uuid4())
        order = Order(order_id, symbol, action, quantity, order_type, price, strategy,
                     instrument_type=instrument_type, contract_multiplier=contract_multiplier)
        self.orders.append(order)
        return order_id
    
    def execute_order(self, order: Order, market_price: float, 
                     greeks: Dict = None) -> bool:
        """
        执行订单
        
        Args:
            order: 订单对象
            market_price: 市场价格
            greeks: Greeks值
            
        Returns:
            是否执行成功
        """
        # 检查订单类型和价格条件
        if order.order_type == "market":
            execution_price = market_price
        elif order.order_type == "limit":
            if order.action == "buy" and market_price <= order.price:
                execution_price = order.price
            elif order.action == "sell" and market_price >= order.price:
                execution_price = order.price
            else:
                return False  # 限价单未成交
        else:
            execution_price = market_price
        
        # 计算佣金
        trade_value = abs(order.quantity) * execution_price * 100  # 期权乘数为100
        commission = trade_value * self.commission_rate
        
        # 检查资金是否充足（买入时）
        if order.action == "buy":
            required_cash = trade_value + commission
            if self.cash < required_cash:
                order.status = "rejected"
                return False
        
        # 执行交易
        signed_quantity = order.quantity if order.action == "buy" else -order.quantity
        
        # 更新持仓
        if order.symbol in self.positions:
            self.positions[order.symbol].add_trade(signed_quantity, execution_price, commission)
        else:
            self.positions[order.symbol] = Position(
                order.symbol, signed_quantity, execution_price,
                self.current_date or datetime.now(), order.strategy,
                order.instrument_type, order.contract_multiplier
            )
        
        # 更新现金 - 根据金融工具类型计算现金流
        # 确定合约乘数
        multiplier = order.contract_multiplier
        if multiplier is None:
            if order.instrument_type == "option":
                multiplier = 100
            elif order.instrument_type == "future":
                multiplier = 1  # 期货通常是1，具体根据品种而定
            else:
                multiplier = 1
        
        if order.instrument_type == "option":
            # 期权交易现金流
            if order.action == "buy":
                # 买入期权：支付权利金 + 手续费
                cash_flow = -(order.quantity * execution_price * multiplier) - commission
            else:  # sell
                # 卖出期权：收到权利金 - 手续费
                cash_flow = (order.quantity * execution_price * multiplier) - commission
        else:
            # 期货或其他金融工具：只有手续费，无权利金
            # 期货交易通常只支付保证金，这里简化处理
            cash_flow = -commission
        
        self.cash += cash_flow
        
        # 更新订单状态
        order.status = "filled"
        order.filled_price = execution_price
        order.filled_quantity = order.quantity
        order.commission = commission
        
        # 记录交易
        trade = {
            'timestamp': self.current_date or datetime.now(),
            'order_id': order.order_id,
            'symbol': order.symbol,
            'action': order.action,
            'quantity': order.quantity,
            'price': execution_price,
            'commission': commission,
            'strategy': order.strategy,
            'cash_flow': cash_flow
        }
        self.trades.append(trade)
        
        # 清理空仓位
        if order.symbol in self.positions and self.positions[order.symbol].quantity == 0:
            del self.positions[order.symbol]
        
        return True
    
    def update_positions(self, market_data: Dict[str, Dict]):
        """
        更新持仓的市价和Greeks
        
        Args:
            market_data: 市场数据 {symbol: {'price': price, 'greeks': greeks}}
        """
        for symbol, position in self.positions.items():
            if symbol in market_data:
                data = market_data[symbol]
                position.update_price(data.get('price', position.current_price),
                                    data.get('greeks', {}))
    
    def close_position(self, symbol: str, quantity: int = None) -> bool:
        """
        平仓
        
        Args:
            symbol: 期权代码
            quantity: 平仓数量，None表示全部平仓
            
        Returns:
            是否成功
        """
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        close_quantity = quantity or abs(position.quantity)
        
        # 确定平仓方向
        action = "sell" if position.quantity > 0 else "buy"
        
        # 下单平仓
        order_id = self.place_order(symbol, action, close_quantity, 
                                  strategy=f"close_{position.strategy}")
        
        return True
    
    def close_expired_positions(self, expired_symbols: List[str]):
        """
        关闭到期持仓
        
        Args:
            expired_symbols: 到期期权代码列表
        """
        for symbol in expired_symbols:
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # 计算到期价值（简化处理，实际需要根据期权类型计算内在价值）
                intrinsic_value = max(0, position.current_price)  # 简化假设
                
                # 记录到期结算
                trade = {
                    'timestamp': self.current_date,
                    'order_id': 'expiry',
                    'symbol': symbol,
                    'action': 'expire',
                    'quantity': position.quantity,
                    'price': intrinsic_value,
                    'commission': 0,
                    'strategy': position.strategy,
                    'instrument_type': position.instrument_type,
                    'cash_flow': position.quantity * intrinsic_value * position.contract_multiplier
                }
                self.trades.append(trade)
                
                # 更新现金
                self.cash += trade['cash_flow']
                
                # 删除持仓
                del self.positions[symbol]
    
    def get_portfolio_value(self) -> float:
        """
        计算投资组合总价值
        
        金融工具持仓价值计算：
        - 多头持仓 (quantity > 0): 拥有权利/资产，价值 = quantity × current_price × contract_multiplier
        - 空头持仓 (quantity < 0): 承担义务/负债，价值 = quantity × current_price × contract_multiplier (为负数)
        
        现金流已在交易时正确处理：
        - 期权：买入支付权利金，卖出收到权利金
        - 期货：主要支付保证金和手续费
        
        投资组合总价值 = 现金 + 所有金融工具持仓的当前市值
        """
        positions_value = 0
        for pos in self.positions.values():
            # 持仓当前市值 = 数量 × 当前价格 × 合约乘数
            # 多头持仓：quantity > 0，当前价值为正（资产）
            # 空头持仓：quantity < 0，当前价值为负（负债）
            position_current_value = pos.quantity * pos.current_price * pos.contract_multiplier
            positions_value += position_current_value
            
        return self.cash + positions_value
    
    def get_portfolio_greeks(self) -> Dict[str, float]:
        """计算投资组合Greeks"""
        total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        for position in self.positions.values():
            greeks = position.greeks
            multiplier = position.quantity
            
            for greek in total_greeks:
                if greek in greeks:
                    total_greeks[greek] += greeks[greek] * multiplier
        
        return total_greeks
    
    def get_position_summary(self) -> pd.DataFrame:
        """获取持仓摘要"""
        if not self.positions:
            return pd.DataFrame()
        
        position_data = []
        for position in self.positions.values():
            data = position.to_dict()
            # 金融工具持仓当前市值（已在to_dict中计算）
            # 多头持仓：正值（资产）
            # 空头持仓：负值（负债）
            # 计算持仓的绝对价值（用于风险分析）
            data['absolute_market_value'] = abs(data['current_market_value'])
            position_data.append(data)
        
        return pd.DataFrame(position_data)
    
    def get_portfolio_breakdown(self) -> Dict[str, float]:
        """
        获取投资组合详细分解
        
        Returns:
            包含各项资产负债的详细分解
        """
        breakdown = {
            'cash': self.cash,
            'long_positions_value': 0,     # 多头持仓总价值（资产）
            'short_positions_value': 0,    # 空头持仓总价值（负债）
            'options_value': 0,            # 期权持仓总价值
            'futures_value': 0,            # 期货持仓总价值
            'other_instruments_value': 0,  # 其他金融工具价值
            'net_positions_value': 0,      # 净持仓价值
            'total_portfolio_value': 0     # 总投资组合价值
        }
        
        for pos in self.positions.values():
            position_current_value = pos.quantity * pos.current_price * pos.contract_multiplier
            
            # 按多空分类
            if pos.quantity > 0:
                breakdown['long_positions_value'] += position_current_value
            else:
                breakdown['short_positions_value'] += position_current_value  # 这是负数
                
            # 按金融工具类型分类
            if pos.instrument_type == "option":
                breakdown['options_value'] += position_current_value
            elif pos.instrument_type == "future":
                breakdown['futures_value'] += position_current_value
            else:
                breakdown['other_instruments_value'] += position_current_value
        
        breakdown['net_positions_value'] = (breakdown['long_positions_value'] + 
                                          breakdown['short_positions_value'])
        breakdown['total_portfolio_value'] = breakdown['cash'] + breakdown['net_positions_value']
        
        return breakdown
    
    def get_trade_history(self) -> pd.DataFrame:
        """获取交易历史"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def record_daily_portfolio(self, date: datetime):
        """记录每日投资组合状态"""
        self.current_date = date
        
        portfolio_value = self.get_portfolio_value()
        portfolio_greeks = self.get_portfolio_greeks()
        
        daily_record = {
            'date': date,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'total_value': portfolio_value,
            'return_pct': (portfolio_value - self.initial_cash) / self.initial_cash,
            'num_positions': len(self.positions),
            **{f'portfolio_{greek}': value for greek, value in portfolio_greeks.items()}
        }
        
        self.daily_portfolio.append(daily_record)
    
    def get_performance_metrics(self) -> Dict:
        """计算绩效指标"""
        if not self.daily_portfolio:
            return {}
        
        df = pd.DataFrame(self.daily_portfolio)
        
        # 基础指标
        final_value = df['total_value'].iloc[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        # 计算日收益率
        df['daily_return'] = df['total_value'].pct_change()
        daily_returns = df['daily_return'].dropna()
        
        if len(daily_returns) == 0:
            return {'total_return': total_return}
        
        # 年化收益率
        trading_days = len(daily_returns)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        
        # 波动率
        volatility = daily_returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        running_max = df['total_value'].expanding().max()
        drawdown = (df['total_value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        winning_trades = sum(1 for trade in self.trades if trade.get('cash_flow', 0) > 0)
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_value': final_value
        }
    
    def get_strategy_performance(self) -> Dict[str, Dict]:
        """按策略分析绩效"""
        strategy_trades = {}
        
        for trade in self.trades:
            strategy = trade.get('strategy', 'unknown')
            if strategy not in strategy_trades:
                strategy_trades[strategy] = []
            strategy_trades[strategy].append(trade)
        
        strategy_performance = {}
        
        for strategy, trades in strategy_trades.items():
            if not trades:
                continue
            
            total_pnl = sum(trade.get('cash_flow', 0) for trade in trades)
            trade_count = len(trades)
            winning_trades = sum(1 for trade in trades if trade.get('cash_flow', 0) > 0)
            
            strategy_performance[strategy] = {
                'total_pnl': total_pnl,
                'trade_count': trade_count,
                'win_rate': winning_trades / trade_count if trade_count > 0 else 0,
                'avg_pnl_per_trade': total_pnl / trade_count if trade_count > 0 else 0
            }
        
        return strategy_performance


class BacktestEngine:
    """回测引擎 - 整合交易引擎进行回测"""
    
    def __init__(self, trading_engine: TradingEngine):
        self.trading_engine = trading_engine
        self.market_data = {}
        self.results = {}
        
    def add_market_data(self, symbol: str, data: pd.DataFrame):
        """添加市场数据"""
        self.market_data[symbol] = data
        
    def run_backtest(self, start_date: str, end_date: str, 
                    signal_generator, strategy_executor) -> Dict:
        """
        运行回测
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            signal_generator: 信号生成器
            strategy_executor: 策略执行器
            
        Returns:
            回测结果
        """
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        for current_date in date_range:
            self.trading_engine.current_date = current_date
            
            # 获取当日市场数据
            daily_market_data = {}
            for symbol, data in self.market_data.items():
                if current_date in data.index:
                    daily_market_data[symbol] = {
                        'price': data.loc[current_date, 'Close'],
                        'data': data.loc[current_date]
                    }
            
            # 更新持仓价格
            self.trading_engine.update_positions(daily_market_data)
            
            # 生成信号（这里需要用户自定义实现）
            # signals = signal_generator.generate_signals(daily_market_data)
            
            # 执行策略（这里需要用户自定义实现）
            # strategy_executor.execute_strategies(signals, self.trading_engine)
            
            # 记录每日投资组合
            self.trading_engine.record_daily_portfolio(current_date)
        
        # 计算绩效
        self.results = self.trading_engine.get_performance_metrics()
        return self.results


if __name__ == "__main__":
    # 测试交易引擎 - 支持期权和期货
    
    # 创建交易引擎
    engine = TradingEngine(initial_cash=100000)
    print(f"Initial cash: ${engine.cash:.2f}")
    
    # 测试1: 买入期权（多头持仓）
    print("\n=== 测试期权交易 ===")
    order_id1 = engine.place_order("AAPL_CALL_150_2024_01", "buy", 10, 
                                   strategy="long_call", instrument_type="option")
    order1 = engine.orders[0]
    success1 = engine.execute_order(order1, 5.50)  # 买入价格5.50
    print(f"Buy option executed: {success1}")
    
    # 测试2: 卖出期权（空头持仓）
    order_id2 = engine.place_order("AAPL_PUT_140_2024_01", "sell", 5, 
                                   strategy="short_put", instrument_type="option")
    order2 = engine.orders[1]
    success2 = engine.execute_order(order2, 3.20)  # 卖出价格3.20
    print(f"Sell option executed: {success2}")
    
    # 测试3: 期货交易
    print("\n=== 测试期货交易 ===")
    order_id3 = engine.place_order("CU2405", "buy", 2, 
                                   strategy="long_future", instrument_type="future",
                                   contract_multiplier=5)  # 铜期货，合约乘数5吨
    order3 = engine.orders[2]
    success3 = engine.execute_order(order3, 74500)  # 价格74,500元/吨
    print(f"Buy future executed: {success3}")
    
    print(f"\nCash after trades: ${engine.cash:.2f}")
    
    # 更新市场价格
    market_data = {
        "AAPL_CALL_150_2024_01": {
            'price': 6.00,  # 看涨期权价格上涨
            'greeks': {'delta': 0.6, 'gamma': 0.03, 'theta': -2, 'vega': 8}
        },
        "AAPL_PUT_140_2024_01": {
            'price': 2.80,  # 看跌期权价格下跌
            'greeks': {'delta': -0.4, 'gamma': 0.02, 'theta': -1.5, 'vega': 6}
        },
        "CU2405": {
            'price': 75200,  # 铜期货价格上涨
        }
    }
    engine.update_positions(market_data)
    
    # 查看详细的投资组合分解
    print("\n=== 投资组合分解 ===")
    breakdown = engine.get_portfolio_breakdown()
    for key, value in breakdown.items():
        print(f"{key}: ${value:.2f}")
    
    # 查看持仓摘要
    print("\n=== 持仓摘要 ===")
    positions = engine.get_position_summary()
    if not positions.empty:
        for _, pos in positions.iterrows():
            print(f"Symbol: {pos['symbol']}")
            print(f"  Instrument Type: {pos['instrument_type']}")
            print(f"  Position Type: {pos['position_type']}")
            print(f"  Quantity: {pos['quantity']}")
            print(f"  Contract Multiplier: {pos['contract_multiplier']}")
            print(f"  Entry Price: ${pos['entry_price']:.2f}")
            print(f"  Current Price: ${pos['current_price']:.2f}")
            print(f"  Cash Flow: ${pos['cash_flow']:.2f}")
            print(f"  Current Market Value: ${pos['current_market_value']:.2f}")
            print(f"  Unrealized PnL: ${pos['unrealized_pnl']:.2f}")
            print()
    
    # 查看投资组合Greeks
    portfolio_greeks = engine.get_portfolio_greeks()
    print("=== 投资组合Greeks ===")
    for greek, value in portfolio_greeks.items():
        print(f"{greek.upper()}: {value:.4f}")
    
    # 验证投资组合价值计算
    portfolio_value = engine.get_portfolio_value()
    print(f"\n=== 投资组合总价值 ===")
    print(f"Portfolio value: ${portfolio_value:.2f}")
    
    # 手动验证计算
    manual_calc = (engine.cash + 
                  10 * 6.00 * 100 +    # 多头call持仓
                  (-5) * 2.80 * 100)    # 空头put持仓（负数）
    print(f"Manual calculation: ${manual_calc:.2f}")
    print(f"Calculation matches: {abs(portfolio_value - manual_calc) < 0.01}")