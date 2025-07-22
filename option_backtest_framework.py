#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权回测框架
包括止盈止损、对冲模块以及生成每日成交持仓记录
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
class OptionContract:
    """期权合约数据结构"""
    symbol: str
    option_type: str  # 'call', 'put', 'barrier', 'snowball'
    strike: float
    expiry_date: str
    barrier_up: Optional[float] = None
    barrier_down: Optional[float] = None
    rebate: Optional[float] = None
    notional: float = 100.0
    direction: int = 1  # 1为买入，-1为卖出

@dataclass
class TradeRecord:
    """交易记录数据结构"""
    date: str
    symbol: str
    action: str  # 'buy', 'sell', 'hedge'
    quantity: int
    price: float
    trade_type: str  # 'open', 'close', 'stop_loss', 'take_profit'
    pnl: float = 0.0
    commission: float = 0.0

@dataclass
class PositionRecord:
    """持仓记录数据结构"""
    date: str
    symbol: str
    quantity: int
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

class OptionPricer:
    """期权定价引擎"""
    
    @staticmethod
    def black_scholes(S: float, K: float, T: float, r: float, sigma: float, 
                     option_type: str = 'call') -> Tuple[float, Dict[str, float]]:
        """Black-Scholes期权定价模型"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
            delta = stats.norm.cdf(d1)
        else:  # put
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
            delta = -stats.norm.cdf(-d1)
        
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                 r * K * np.exp(-r * T) * stats.norm.cdf(d2 if option_type.lower() == 'call' else -d2))
        vega = S * stats.norm.pdf(d1) * np.sqrt(T)
        
        greeks = {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # 转换为每日theta
            'vega': vega / 100     # 转换为1%波动率变化的vega
        }
        
        return price, greeks
    
    @staticmethod
    def barrier_option_mc(S: float, K: float, T: float, r: float, sigma: float,
                         barrier_up: Optional[float] = None, barrier_down: Optional[float] = None,
                         rebate: float = 0.0, option_type: str = 'call',
                         n_simulations: int = 10000, n_steps: int = 252) -> float:
        """蒙特卡洛方法定价障碍期权"""
        dt = T / n_steps
        payoffs = []
        
        for _ in range(n_simulations):
            # 生成股价路径
            Z = np.random.standard_normal(n_steps)
            path = [S]
            
            for i in range(n_steps):
                St = path[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[i])
                path.append(St)
            
            path = np.array(path)
            
            # 检查是否触发障碍
            barrier_hit = False
            if barrier_up is not None and np.any(path >= barrier_up):
                barrier_hit = True
            if barrier_down is not None and np.any(path <= barrier_down):
                barrier_hit = True
            
            # 计算收益
            if barrier_hit:
                payoff = rebate
            else:
                if option_type.lower() == 'call':
                    payoff = max(path[-1] - K, 0)
                else:
                    payoff = max(K - path[-1], 0)
            
            payoffs.append(payoff)
        
        return np.mean(payoffs) * np.exp(-r * T)

class RiskManager:
    """风险管理模块"""
    
    def __init__(self, stop_loss_pct: float = 0.2, take_profit_pct: float = 0.5,
                 max_position_size: float = 1000000, delta_hedge_threshold: float = 0.1):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_size = max_position_size
        self.delta_hedge_threshold = delta_hedge_threshold
    
    def check_stop_loss(self, current_pnl: float, initial_cost: float) -> bool:
        """检查是否触发止损"""
        if initial_cost == 0:
            return False
        loss_pct = -current_pnl / abs(initial_cost)
        return loss_pct >= self.stop_loss_pct
    
    def check_take_profit(self, current_pnl: float, initial_cost: float) -> bool:
        """检查是否触发止盈"""
        if initial_cost == 0:
            return False
        profit_pct = current_pnl / abs(initial_cost)
        return profit_pct >= self.take_profit_pct
    
    def check_hedge_signal(self, portfolio_delta: float) -> bool:
        """检查是否需要对冲"""
        return abs(portfolio_delta) >= self.delta_hedge_threshold

class HedgeEngine:
    """对冲引擎"""
    
    def __init__(self, hedge_ratio: float = 1.0, hedge_frequency: str = 'daily'):
        self.hedge_ratio = hedge_ratio
        self.hedge_frequency = hedge_frequency
    
    def calculate_hedge_size(self, portfolio_delta: float, stock_price: float) -> int:
        """计算对冲数量"""
        # 计算需要买入/卖出的股票数量来对冲delta
        hedge_shares = -int(portfolio_delta * self.hedge_ratio * 100)  # 假设每个期权对应100股
        return hedge_shares
    
    def generate_hedge_trade(self, date: str, symbol: str, portfolio_delta: float,
                           stock_price: float) -> Optional[TradeRecord]:
        """生成对冲交易"""
        hedge_size = self.calculate_hedge_size(portfolio_delta, stock_price)
        
        if abs(hedge_size) > 0:
            action = 'buy' if hedge_size > 0 else 'sell'
            return TradeRecord(
                date=date,
                symbol=f"{symbol}_STOCK",
                action=action,
                quantity=abs(hedge_size),
                price=stock_price,
                trade_type='hedge'
            )
        return None

class OptionBacktestEngine:
    """期权回测引擎"""
    
    def __init__(self, initial_capital: float = 1000000, commission_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        
        # 组件
        self.pricer = OptionPricer()
        self.risk_manager = RiskManager()
        self.hedge_engine = HedgeEngine()
        
        # 记录
        self.trade_records: List[TradeRecord] = []
        self.position_records: List[PositionRecord] = []
        self.daily_pnl: List[Dict] = []
        
        # 持仓
        self.positions: Dict[str, Dict] = {}  # {symbol: {quantity, avg_cost, ...}}
        self.option_positions: Dict[str, OptionContract] = {}
        
    def add_market_data(self, data: pd.DataFrame):
        """添加市场数据"""
        self.market_data = data.copy()
        self.market_data['date'] = pd.to_datetime(self.market_data['date'])
        self.market_data = self.market_data.set_index('date').sort_index()
    
    def calculate_option_value(self, contract: OptionContract, current_price: float,
                             current_date: str, risk_free_rate: float = 0.03,
                             volatility: float = 0.2) -> Tuple[float, Dict[str, float]]:
        """计算期权价值和希腊字母"""
        
        # 计算到期时间
        expiry = pd.to_datetime(contract.expiry_date)
        current = pd.to_datetime(current_date)
        time_to_expiry = (expiry - current).days / 365.0
        
        if time_to_expiry <= 0:
            # 期权已到期
            if contract.option_type in ['call', 'put']:
                intrinsic_value = max(current_price - contract.strike, 0) if contract.option_type == 'call' else max(contract.strike - current_price, 0)
                return intrinsic_value, {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            else:
                return 0, {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        if contract.option_type in ['call', 'put']:
            return self.pricer.black_scholes(
                current_price, contract.strike, time_to_expiry,
                risk_free_rate, volatility, contract.option_type
            )
        elif contract.option_type == 'barrier':
            price = self.pricer.barrier_option_mc(
                current_price, contract.strike, time_to_expiry,
                risk_free_rate, volatility, contract.barrier_up, contract.barrier_down,
                contract.rebate or 0, 'call'
            )
            # 简化的希腊字母计算
            return price, {'delta': 0.5, 'gamma': 0, 'theta': -price/365, 'vega': 0}
        
        return 0, {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def open_position(self, date: str, contract: OptionContract, quantity: int, price: float):
        """开仓"""
        symbol = contract.symbol
        cost = quantity * price * contract.notional
        commission = cost * self.commission_rate
        
        # 记录交易
        trade = TradeRecord(
            date=date,
            symbol=symbol,
            action='buy' if contract.direction > 0 else 'sell',
            quantity=quantity,
            price=price,
            trade_type='open',
            commission=commission
        )
        self.trade_records.append(trade)
        
        # 更新持仓
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_cost': 0, 'total_cost': 0}
            self.option_positions[symbol] = contract
        
        current_pos = self.positions[symbol]
        new_quantity = current_pos['quantity'] + quantity * contract.direction
        new_total_cost = current_pos['total_cost'] + cost * contract.direction
        
        self.positions[symbol]['quantity'] = new_quantity
        self.positions[symbol]['total_cost'] = new_total_cost
        if new_quantity != 0:
            self.positions[symbol]['avg_cost'] = new_total_cost / new_quantity
        
        # 更新资金
        self.current_capital -= (cost + commission)
    
    def close_position(self, date: str, symbol: str, quantity: int, price: float, reason: str = 'manual'):
        """平仓"""
        if symbol not in self.positions or self.positions[symbol]['quantity'] == 0:
            return
        
        contract = self.option_positions[symbol]
        close_quantity = min(quantity, abs(self.positions[symbol]['quantity']))
        
        # 计算盈亏
        avg_cost = self.positions[symbol]['avg_cost']
        pnl = (price - avg_cost) * close_quantity * contract.notional
        if self.positions[symbol]['quantity'] < 0:  # 空头持仓
            pnl = -pnl
        
        commission = close_quantity * price * contract.notional * self.commission_rate
        
        # 记录交易
        trade = TradeRecord(
            date=date,
            symbol=symbol,
            action='sell' if self.positions[symbol]['quantity'] > 0 else 'buy',
            quantity=close_quantity,
            price=price,
            trade_type=reason,
            pnl=pnl,
            commission=commission
        )
        self.trade_records.append(trade)
        
        # 更新持仓
        direction = 1 if self.positions[symbol]['quantity'] > 0 else -1
        self.positions[symbol]['quantity'] -= close_quantity * direction
        self.positions[symbol]['total_cost'] -= avg_cost * close_quantity * direction
        
        # 更新资金
        proceeds = close_quantity * price * contract.notional
        self.current_capital += proceeds - commission + pnl
    
    def update_daily_positions(self, date: str):
        """更新每日持仓"""
        if date not in self.market_data.index:
            return
        
        current_price = self.market_data.loc[date, 'close']
        portfolio_delta = 0
        total_market_value = 0
        total_unrealized_pnl = 0
        
        for symbol, position in self.positions.items():
            if position['quantity'] == 0:
                continue
            
            contract = self.option_positions[symbol]
            option_value, greeks = self.calculate_option_value(contract, current_price, date)
            
            market_value = position['quantity'] * option_value * contract.notional
            unrealized_pnl = market_value - position['total_cost']
            
            total_market_value += market_value
            total_unrealized_pnl += unrealized_pnl
            portfolio_delta += position['quantity'] * greeks['delta']
            
            # 记录持仓
            pos_record = PositionRecord(
                date=date,
                symbol=symbol,
                quantity=position['quantity'],
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=0,  # 当日已实现盈亏在交易记录中
                delta=greeks['delta'],
                gamma=greeks['gamma'],
                theta=greeks['theta'],
                vega=greeks['vega']
            )
            self.position_records.append(pos_record)
            
            # 检查止盈止损
            if self.risk_manager.check_stop_loss(unrealized_pnl, abs(position['total_cost'])):
                self.close_position(date, symbol, abs(position['quantity']), option_value, 'stop_loss')
            elif self.risk_manager.check_take_profit(unrealized_pnl, abs(position['total_cost'])):
                self.close_position(date, symbol, abs(position['quantity']), option_value, 'take_profit')
        
        # 检查对冲
        if self.risk_manager.check_hedge_signal(portfolio_delta):
            hedge_trade = self.hedge_engine.generate_hedge_trade(date, 'INDEX', portfolio_delta, current_price)
            if hedge_trade:
                self.trade_records.append(hedge_trade)
        
        # 记录每日盈亏
        daily_record = {
            'date': date,
            'capital': self.current_capital,
            'market_value': total_market_value,
            'total_value': self.current_capital + total_market_value,
            'unrealized_pnl': total_unrealized_pnl,
            'portfolio_delta': portfolio_delta,
            'daily_return': 0  # 将在后处理中计算
        }
        self.daily_pnl.append(daily_record)
    
    def run_backtest(self, strategy_signals: pd.DataFrame):
        """运行回测"""
        print("开始期权回测...")
        
        for date in self.market_data.index:
            date_str = date.strftime('%Y-%m-%d')
            
            # 处理策略信号
            if date in strategy_signals.index:
                signals = strategy_signals.loc[date]
                if not pd.isna(signals.get('action')):
                    if signals['action'] == 'buy':
                        contract = OptionContract(
                            symbol=signals.get('symbol', 'OPTION'),
                            option_type=signals.get('option_type', 'call'),
                            strike=signals.get('strike', self.market_data.loc[date, 'close']),
                            expiry_date=signals.get('expiry_date', (date + timedelta(days=30)).strftime('%Y-%m-%d')),
                            direction=1
                        )
                        option_price = signals.get('option_price', 10)  # 默认期权价格
                        self.open_position(date_str, contract, signals.get('quantity', 1), option_price)
            
            # 更新持仓
            self.update_daily_positions(date_str)
        
        # 计算每日收益率
        self._calculate_daily_returns()
        print("回测完成!")
    
    def _calculate_daily_returns(self):
        """计算每日收益率"""
        if len(self.daily_pnl) <= 1:
            return
        
        for i in range(1, len(self.daily_pnl)):
            prev_value = self.daily_pnl[i-1]['total_value']
            curr_value = self.daily_pnl[i]['total_value']
            self.daily_pnl[i]['daily_return'] = (curr_value - prev_value) / prev_value if prev_value != 0 else 0
    
    def get_performance_summary(self) -> Dict:
        """获取绩效摘要"""
        if not self.daily_pnl:
            return {}
        
        df_pnl = pd.DataFrame(self.daily_pnl)
        df_pnl['date'] = pd.to_datetime(df_pnl['date'])
        
        total_return = (df_pnl['total_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        daily_returns = df_pnl['daily_return'].dropna()
        
        summary = {
            'initial_capital': self.initial_capital,
            'final_value': df_pnl['total_value'].iloc[-1],
            'total_return': total_return,
            'total_trades': len(self.trade_records),
            'sharpe_ratio': daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(df_pnl),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor()
        }
        
        return summary
    
    def _calculate_max_drawdown(self, df_pnl: pd.DataFrame) -> float:
        """计算最大回撤"""
        cumulative = df_pnl['total_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        profitable_trades = [t for t in self.trade_records if t.pnl > 0]
        total_trades = [t for t in self.trade_records if t.trade_type != 'hedge']
        return len(profitable_trades) / len(total_trades) if total_trades else 0
    
    def _calculate_profit_factor(self) -> float:
        """计算盈亏比"""
        profits = sum([t.pnl for t in self.trade_records if t.pnl > 0])
        losses = abs(sum([t.pnl for t in self.trade_records if t.pnl < 0]))
        return profits / losses if losses != 0 else np.inf
    
    def plot_results(self):
        """绘制回测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 资金曲线
        df_pnl = pd.DataFrame(self.daily_pnl)
        df_pnl['date'] = pd.to_datetime(df_pnl['date'])
        
        axes[0, 0].plot(df_pnl['date'], df_pnl['total_value'])
        axes[0, 0].set_title('资金曲线')
        axes[0, 0].set_ylabel('总资产')
        axes[0, 0].grid(True)
        
        # 每日收益率
        axes[0, 1].plot(df_pnl['date'], df_pnl['daily_return'] * 100)
        axes[0, 1].set_title('每日收益率')
        axes[0, 1].set_ylabel('收益率 (%)')
        axes[0, 1].grid(True)
        
        # 持仓价值
        axes[1, 0].plot(df_pnl['date'], df_pnl['market_value'])
        axes[1, 0].set_title('持仓市值')
        axes[1, 0].set_ylabel('市值')
        axes[1, 0].grid(True)
        
        # Delta暴露
        axes[1, 1].plot(df_pnl['date'], df_pnl['portfolio_delta'])
        axes[1, 1].set_title('组合Delta')
        axes[1, 1].set_ylabel('Delta')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def export_records(self, file_prefix: str = 'backtest'):
        """导出记录到文件"""
        # 导出交易记录
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
        trades_df.to_csv(f'{file_prefix}_trades.csv', index=False, encoding='utf-8-sig')
        
        # 导出持仓记录
        positions_df = pd.DataFrame([{
            'date': p.date,
            'symbol': p.symbol,
            'quantity': p.quantity,
            'market_value': p.market_value,
            'unrealized_pnl': p.unrealized_pnl,
            'realized_pnl': p.realized_pnl,
            'delta': p.delta,
            'gamma': p.gamma,
            'theta': p.theta,
            'vega': p.vega
        } for p in self.position_records])
        positions_df.to_csv(f'{file_prefix}_positions.csv', index=False, encoding='utf-8-sig')
        
        # 导出每日盈亏
        pnl_df = pd.DataFrame(self.daily_pnl)
        pnl_df.to_csv(f'{file_prefix}_daily_pnl.csv', index=False, encoding='utf-8-sig')
        
        print(f"回测记录已导出到 {file_prefix}_*.csv 文件")

def create_sample_strategy() -> pd.DataFrame:
    """创建示例策略信号"""
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='W')  # 每周一个信号
    
    signals = []
    for i, date in enumerate(dates[:10]):  # 只生成前10个信号
        signals.append({
            'date': date,
            'action': 'buy',
            'symbol': f'OPTION_{i}',
            'option_type': 'call',
            'strike': 100 + i * 5,
            'expiry_date': (date + timedelta(days=30)).strftime('%Y-%m-%d'),
            'quantity': 1,
            'option_price': 10 + np.random.normal(0, 2)
        })
    
    df = pd.DataFrame(signals)
    df = df.set_index('date')
    return df

def create_sample_market_data() -> pd.DataFrame:
    """创建示例市场数据"""
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    
    # 生成模拟股价数据
    np.random.seed(42)
    prices = [100]
    for _ in range(len(dates) - 1):
        change = np.random.normal(0, 0.02)  # 2%日波动率
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    market_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    return market_data

if __name__ == "__main__":
    # 示例使用
    print("期权回测框架示例")
    print("="*50)
    
    # 创建回测引擎
    engine = OptionBacktestEngine(initial_capital=1000000)
    
    # 添加市场数据
    market_data = create_sample_market_data()
    engine.add_market_data(market_data)
    
    # 创建策略信号
    strategy_signals = create_sample_strategy()
    
    # 运行回测
    engine.run_backtest(strategy_signals)
    
    # 获取绩效摘要
    summary = engine.get_performance_summary()
    print("\n绩效摘要:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # 绘制结果
    engine.plot_results()
    
    # 导出记录
    engine.export_records('option_backtest')