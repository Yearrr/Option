#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权回测框架 - 主框架模块
整合所有模块，提供完整的期权策略回测流程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
import os
import json
import uuid

# 导入自定义模块
from .data_manager import DataManager
from .option_pricing import OptionPricer, StrategyPricer
from .signal_generator import CompositeSignalGenerator, BaseSignalGenerator
from .risk_manager import RiskManager, PositionSizer
from .trading_engine import TradingEngine, Order

warnings.filterwarnings('ignore')


class OptionsBacktestFramework:
    """期权回测框架主类"""
    
    def __init__(self, config: Dict):
        """
        初始化回测框架
        
        Args:
            config: 配置字典，包含所有参数设置
        """
        self.config = config
        
        # 基础参数
        self.underlying_symbol = config['underlying_symbol']
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        self.initial_cash = config.get('initial_cash', 100000)
        self.commission_rate = config.get('commission_rate', 0.001)
        
        # 风险管理参数
        self.risk_config = config.get('risk_management', {})
        self.margin_config = config.get('margin_parameters', {})
        self.hedge_config = config.get('hedge_parameters', {})
        
        # 初始化组件
        self.data_manager = DataManager(
            data_source=config.get('data_source', 'yfinance'),
            cache_dir=config.get('cache_dir', 'data_cache')
        )
        
        self.option_pricer = OptionPricer()
        self.strategy_pricer = StrategyPricer(self.option_pricer)
        
        self.risk_manager = RiskManager(self.risk_config)
        self.position_sizer = PositionSizer(config.get('position_sizing', {}))
        
        self.trading_engine = TradingEngine(
            initial_cash=self.initial_cash,
            commission_rate=self.commission_rate
        )
        
        # 数据存储
        self.underlying_data = pd.DataFrame()
        self.signals = pd.DataFrame()
        self.results = {}
        
        # 风险参数
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        
    def load_data(self) -> pd.DataFrame:
        """
        加载标的资产数据
        
        Returns:
            标的数据DataFrame
        """
        print(f"正在加载 {self.underlying_symbol} 从 {self.start_date} 到 {self.end_date} 的数据...")
        
        self.underlying_data = self.data_manager.get_underlying_data(
            self.underlying_symbol,
            self.start_date,
            self.end_date
        )
        
        if self.underlying_data.empty:
            raise ValueError(f"无法获取 {self.underlying_symbol} 的数据")
        
        print(f"成功加载 {len(self.underlying_data)} 条数据记录")
        return self.underlying_data
    
    def setup_signal_generators(self, signal_generators: List[BaseSignalGenerator],
                               weights: Dict[str, float] = None) -> CompositeSignalGenerator:
        """
        设置信号生成器
        
        Args:
            signal_generators: 信号生成器列表
            weights: 信号权重
            
        Returns:
            组合信号生成器
        """
        self.signal_generator = CompositeSignalGenerator(signal_generators, weights)
        return self.signal_generator
    
    def generate_signals(self, **kwargs) -> pd.DataFrame:
        """
        生成交易信号
        
        Returns:
            信号DataFrame
        """
        if not hasattr(self, 'signal_generator'):
            raise ValueError("请先设置信号生成器")
        
        print("正在生成交易信号...")
        self.signals = self.signal_generator.generate_composite_signals(
            self.underlying_data, **kwargs
        )
        
        signal_summary = self.signal_generator.get_signal_summary(self.signals)
        print(f"信号生成完成: {signal_summary}")
        
        return self.signals
    
    def create_option_chains(self, expiry_dates: List[str], 
                           strike_range: Tuple[float, float] = None) -> Dict[str, pd.DataFrame]:
        """
        创建期权链数据（模拟或实际）
        
        Args:
            expiry_dates: 到期日列表
            strike_range: 行权价范围 (min_strike, max_strike)
            
        Returns:
            期权链数据字典
        """
        print("正在创建期权链数据...")
        
        option_chains = {}
        
        for expiry_date in expiry_dates:
            # 这里可以尝试获取真实期权数据，或生成模拟数据
            try:
                chain = self.data_manager.get_options_chain(self.underlying_symbol, expiry_date)
                if not chain.empty:
                    option_chains[expiry_date] = chain
                    continue
            except:
                pass
            
            # 生成模拟期权链
            option_chains[expiry_date] = self._generate_synthetic_option_chain(
                expiry_date, strike_range
            )
        
        return option_chains
    
    def _generate_synthetic_option_chain(self, expiry_date: str, 
                                       strike_range: Tuple[float, float] = None) -> pd.DataFrame:
        """
        生成模拟期权链
        
        Args:
            expiry_date: 到期日
            strike_range: 行权价范围
            
        Returns:
            模拟期权链DataFrame
        """
        # 获取当前标的价格
        current_price = self.underlying_data['Close'].iloc[-1]
        
        # 设置行权价范围
        if strike_range is None:
            min_strike = current_price * 0.8
            max_strike = current_price * 1.2
        else:
            min_strike, max_strike = strike_range
        
        # 生成行权价
        strikes = np.arange(
            round(min_strike / 5) * 5,  # 向下取整到5的倍数
            round(max_strike / 5) * 5 + 5,  # 向上取整到5的倍数
            5
        )
        
        # 计算到期时间
        expiry_dt = pd.to_datetime(expiry_date)
        today = pd.to_datetime(self.end_date)
        time_to_expiry = (expiry_dt - today).days / 365.0
        
        if time_to_expiry <= 0:
            time_to_expiry = 1/365  # 最少1天
        
        # 估算波动率
        returns = self.underlying_data['Returns'].dropna()
        historical_vol = returns.std() * np.sqrt(252)
        
        option_data = []
        
        for strike in strikes:
            # 计算看涨期权价格
            call_price = self.option_pricer.black_scholes(
                current_price, strike, time_to_expiry, 
                self.risk_free_rate, historical_vol, "call"
            )
            
            # 计算看跌期权价格
            put_price = self.option_pricer.black_scholes(
                current_price, strike, time_to_expiry,
                self.risk_free_rate, historical_vol, "put"
            )
            
            # 计算Greeks
            call_greeks = self.option_pricer.calculate_greeks(
                current_price, strike, time_to_expiry,
                self.risk_free_rate, historical_vol, "call"
            )
            
            put_greeks = self.option_pricer.calculate_greeks(
                current_price, strike, time_to_expiry,
                self.risk_free_rate, historical_vol, "put"
            )
            
            # 看涨期权
            option_data.append({
                'symbol': f"{self.underlying_symbol}_C_{strike}_{expiry_date.replace('-', '')}",
                'strike': strike,
                'expiry': expiry_date,
                'option_type': 'call',
                'price': call_price,
                'bid': call_price * 0.99,
                'ask': call_price * 1.01,
                'volume': np.random.randint(100, 1000),
                'open_interest': np.random.randint(500, 5000),
                'implied_volatility': historical_vol,
                **{f'call_{k}': v for k, v in call_greeks.items()}
            })
            
            # 看跌期权
            option_data.append({
                'symbol': f"{self.underlying_symbol}_P_{strike}_{expiry_date.replace('-', '')}",
                'strike': strike,
                'expiry': expiry_date,
                'option_type': 'put',
                'price': put_price,
                'bid': put_price * 0.99,
                'ask': put_price * 1.01,
                'volume': np.random.randint(100, 1000),
                'open_interest': np.random.randint(500, 5000),
                'implied_volatility': historical_vol,
                **{f'put_{k}': v for k, v in put_greeks.items()}
            })
        
        return pd.DataFrame(option_data)
    
    def execute_strategy(self, strategy_name: str, signal_date: datetime,
                        option_chains: Dict[str, pd.DataFrame]) -> List[Order]:
        """
        执行具体策略
        
        Args:
            strategy_name: 策略名称
            signal_date: 信号日期
            option_chains: 期权链数据
            
        Returns:
            订单列表
        """
        orders = []
        current_price = self.underlying_data.loc[signal_date, 'Close']
        
        # 根据策略类型执行不同逻辑
        if strategy_name == 'long_straddle':
            orders = self._execute_long_straddle(current_price, option_chains)
        elif strategy_name == 'short_straddle':
            orders = self._execute_short_straddle(current_price, option_chains)
        elif strategy_name == 'bull_call_spread':
            orders = self._execute_bull_call_spread(current_price, option_chains)
        elif strategy_name == 'bear_put_spread':
            orders = self._execute_bear_put_spread(current_price, option_chains)
        elif strategy_name == 'iron_condor':
            orders = self._execute_iron_condor(current_price, option_chains)
        elif strategy_name == 'ratio_spread':
            orders = self._execute_ratio_spread(current_price, option_chains)
        
        return orders
    
    def _execute_long_straddle(self, current_price: float, 
                              option_chains: Dict[str, pd.DataFrame]) -> List[Order]:
        """执行做多跨式策略"""
        orders = []
        
        # 选择最近到期的期权
        nearest_expiry = min(option_chains.keys())
        chain = option_chains[nearest_expiry]
        
        # 选择平值期权
        atm_strike = chain.loc[
            (chain['strike'] - current_price).abs().idxmin(), 'strike'
        ]
        
        # 买入ATM Call和Put
        call_option = chain[
            (chain['strike'] == atm_strike) & (chain['option_type'] == 'call')
        ].iloc[0]
        
        put_option = chain[
            (chain['strike'] == atm_strike) & (chain['option_type'] == 'put')
        ].iloc[0]
        
        # 计算仓位大小
        strategy_info = {
            'strategy': 'long_straddle',
            'option_price': call_option['price'] + put_option['price']
        }
        
        position_size = self.position_sizer.calculate_position_size(
            strategy_info, self.trading_engine.get_portfolio_value()
        )
        
        # 创建订单
        orders.append(Order(
            str(uuid.uuid4()), call_option['symbol'], 'buy', position_size,
            strategy='long_straddle'
        ))
        
        orders.append(Order(
            str(uuid.uuid4()), put_option['symbol'], 'buy', position_size,
            strategy='long_straddle'
        ))
        
        return orders
    
    def _execute_short_straddle(self, current_price: float,
                               option_chains: Dict[str, pd.DataFrame]) -> List[Order]:
        """执行做空跨式策略"""
        orders = []
        
        # 选择最近到期的期权
        nearest_expiry = min(option_chains.keys())
        chain = option_chains[nearest_expiry]
        
        # 选择平值期权
        atm_strike = chain.loc[
            (chain['strike'] - current_price).abs().idxmin(), 'strike'
        ]
        
        # 卖出ATM Call和Put
        call_option = chain[
            (chain['strike'] == atm_strike) & (chain['option_type'] == 'call')
        ].iloc[0]
        
        put_option = chain[
            (chain['strike'] == atm_strike) & (chain['option_type'] == 'put')
        ].iloc[0]
        
        # 计算仓位大小（卖方策略仓位较小）
        strategy_info = {
            'strategy': 'short_straddle',
            'option_price': call_option['price'] + put_option['price']
        }
        
        position_size = self.position_sizer.calculate_position_size(
            strategy_info, self.trading_engine.get_portfolio_value()
        ) // 2  # 卖方策略减半
        
        # 创建订单
        orders.append(Order(
            str(uuid.uuid4()), call_option['symbol'], 'sell', position_size,
            strategy='short_straddle'
        ))
        
        orders.append(Order(
            str(uuid.uuid4()), put_option['symbol'], 'sell', position_size,
            strategy='short_straddle'
        ))
        
        return orders
    
    def _execute_bull_call_spread(self, current_price: float,
                                 option_chains: Dict[str, pd.DataFrame]) -> List[Order]:
        """执行牛市看涨价差策略"""
        orders = []
        
        # 选择最近到期的期权
        nearest_expiry = min(option_chains.keys())
        chain = option_chains[nearest_expiry]
        calls = chain[chain['option_type'] == 'call'].copy()
        
        # 选择行权价：买入平值，卖出价外
        atm_strike = calls.loc[
            (calls['strike'] - current_price).abs().idxmin(), 'strike'
        ]
        
        otm_strike = atm_strike + 10  # 价外10点
        
        long_call = calls[calls['strike'] == atm_strike].iloc[0]
        short_call = calls[calls['strike'] == otm_strike]
        
        if short_call.empty:
            return orders
        
        short_call = short_call.iloc[0]
        
        # 计算仓位大小
        strategy_info = {
            'strategy': 'bull_call_spread',
            'option_price': long_call['price'] - short_call['price']
        }
        
        position_size = self.position_sizer.calculate_position_size(
            strategy_info, self.trading_engine.get_portfolio_value()
        )
        
        # 创建订单
        orders.append(Order(
            str(uuid.uuid4()), long_call['symbol'], 'buy', position_size,
            strategy='bull_call_spread'
        ))
        
        orders.append(Order(
            str(uuid.uuid4()), short_call['symbol'], 'sell', position_size,
            strategy='bull_call_spread'
        ))
        
        return orders
    
    def _execute_bear_put_spread(self, current_price: float,
                                option_chains: Dict[str, pd.DataFrame]) -> List[Order]:
        """执行熊市看跌价差策略"""
        # 类似牛市看涨价差的实现逻辑
        return []
    
    def _execute_iron_condor(self, current_price: float,
                            option_chains: Dict[str, pd.DataFrame]) -> List[Order]:
        """执行铁鹰策略"""
        # 铁鹰策略的实现逻辑
        return []
    
    def _execute_ratio_spread(self, current_price: float,
                             option_chains: Dict[str, pd.DataFrame]) -> List[Order]:
        """执行比例价差策略"""
        # 比例价差策略的实现逻辑
        return []
    
    def run_backtest(self, signal_generators: List[BaseSignalGenerator] = None,
                    expiry_dates: List[str] = None, **kwargs) -> Dict:
        """
        运行完整回测
        
        Args:
            signal_generators: 信号生成器列表
            expiry_dates: 期权到期日列表
            **kwargs: 其他参数
            
        Returns:
            回测结果字典
        """
        print("=" * 50)
        print("开始期权策略回测")
        print("=" * 50)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 设置信号生成器
        if signal_generators:
            self.setup_signal_generators(signal_generators)
            self.generate_signals(**kwargs)
        
        # 3. 创建期权链
        if expiry_dates is None:
            # 默认使用每月第三个周五作为到期日
            expiry_dates = self._generate_monthly_expiries()
        
        option_chains = self.create_option_chains(expiry_dates)
        
        # 4. 回测主循环
        print("\n开始回测循环...")
        
        trading_dates = self.underlying_data.index
        
        for i, date in enumerate(trading_dates):
            self.trading_engine.current_date = date
            
            # 更新持仓价格（这里需要实现期权价格更新逻辑）
            self._update_option_prices(date, option_chains)
            
            # 检查信号
            if hasattr(self, 'signals') and date in self.signals.index:
                signal_row = self.signals.loc[date]
                
                if signal_row['composite_signal'] != 0:
                    strategy = signal_row['primary_strategy']
                    
                    if strategy:
                        # 执行策略
                        orders = self.execute_strategy(strategy, date, option_chains)
                        
                        # 执行订单
                        for order in orders:
                            # 这里需要获取期权的当前价格
                            market_price = self._get_option_price(order.symbol, date, option_chains)
                            if market_price:
                                self.trading_engine.execute_order(order, market_price)
            
            # 风险管理
            self._apply_risk_management(date)
            
            # 记录每日状态
            self.trading_engine.record_daily_portfolio(date)
            
            # 进度显示
            if i % 50 == 0:
                print(f"进度: {i+1}/{len(trading_dates)} ({(i+1)/len(trading_dates)*100:.1f}%)")
        
        # 5. 计算结果
        print("\n计算回测结果...")
        self.results = self._calculate_results()
        
        # 6. 输出结果
        self._print_results()
        
        return self.results
    
    def _generate_monthly_expiries(self) -> List[str]:
        """生成每月期权到期日（第三个周五）"""
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        
        expiry_dates = []
        current = start.replace(day=1)  # 月初
        
        while current <= end:
            # 找到当月第三个周五
            month_start = current
            month_end = (current + pd.DateOffset(months=1) - pd.DateOffset(days=1))
            
            fridays = pd.date_range(month_start, month_end, freq='W-FRI')
            if len(fridays) >= 3:
                third_friday = fridays[2]
                expiry_dates.append(third_friday.strftime('%Y-%m-%d'))
            
            current += pd.DateOffset(months=1)
        
        return expiry_dates
    
    def _update_option_prices(self, date: datetime, option_chains: Dict[str, pd.DataFrame]):
        """更新期权价格"""
        # 这里需要实现期权价格的日常更新逻辑
        # 可以基于Black-Scholes公式和当前标的价格重新计算
        pass
    
    def _get_option_price(self, symbol: str, date: datetime, 
                         option_chains: Dict[str, pd.DataFrame]) -> Optional[float]:
        """获取期权价格"""
        # 从期权链中查找对应期权的价格
        for expiry, chain in option_chains.items():
            option_row = chain[chain['symbol'] == symbol]
            if not option_row.empty:
                return option_row.iloc[0]['price']
        return None
    
    def _apply_risk_management(self, date: datetime):
        """应用风险管理规则"""
        # 检查止盈止损
        positions_to_close = []
        
        for symbol, position in self.trading_engine.positions.items():
            current_pnl = position.unrealized_pnl
            
            # 检查止损
            if self.risk_manager.should_stop_loss(position.to_dict(), current_pnl):
                positions_to_close.append(symbol)
                continue
            
            # 检查止盈
            if self.risk_manager.should_take_profit(position.to_dict(), current_pnl):
                positions_to_close.append(symbol)
        
        # 执行平仓
        for symbol in positions_to_close:
            self.trading_engine.close_position(symbol)
    
    def _calculate_results(self) -> Dict:
        """计算回测结果"""
        results = {
            'performance_metrics': self.trading_engine.get_performance_metrics(),
            'strategy_performance': self.trading_engine.get_strategy_performance(),
            'position_summary': self.trading_engine.get_position_summary(),
            'trade_history': self.trading_engine.get_trade_history(),
            'daily_portfolio': pd.DataFrame(self.trading_engine.daily_portfolio)
        }
        
        return results
    
    def _print_results(self):
        """打印回测结果"""
        performance = self.results['performance_metrics']
        
        print("\n" + "=" * 50)
        print("回测结果摘要")
        print("=" * 50)
        
        print(f"总收益率: {performance.get('total_return', 0):.2%}")
        print(f"年化收益率: {performance.get('annualized_return', 0):.2%}")
        print(f"年化波动率: {performance.get('volatility', 0):.2%}")
        print(f"夏普比率: {performance.get('sharpe_ratio', 0):.3f}")
        print(f"最大回撤: {performance.get('max_drawdown', 0):.2%}")
        print(f"胜率: {performance.get('win_rate', 0):.2%}")
        print(f"总交易次数: {performance.get('total_trades', 0)}")
        print(f"最终价值: ${performance.get('final_value', 0):,.2f}")
    
    def save_results(self, output_dir: str = "backtest_results"):
        """保存回测结果"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存配置
        with open(f"{output_dir}/config.json", 'w') as f:
            # 将datetime对象转换为字符串
            config_copy = self.config.copy()
            json.dump(config_copy, f, indent=2, default=str)
        
        # 保存各种结果
        if 'trade_history' in self.results and not self.results['trade_history'].empty:
            self.results['trade_history'].to_csv(f"{output_dir}/trade_history.csv")
        
        if 'daily_portfolio' in self.results and not self.results['daily_portfolio'].empty:
            self.results['daily_portfolio'].to_csv(f"{output_dir}/daily_portfolio.csv")
        
        if 'position_summary' in self.results and not self.results['position_summary'].empty:
            self.results['position_summary'].to_csv(f"{output_dir}/positions.csv")
        
        # 保存绩效指标
        with open(f"{output_dir}/performance_metrics.json", 'w') as f:
            json.dump(self.results['performance_metrics'], f, indent=2, default=str)
        
        print(f"结果已保存到 {output_dir} 目录")
    
    def plot_results(self, save_plots: bool = True, output_dir: str = "backtest_results"):
        """绘制回测结果图表"""
        if 'daily_portfolio' not in self.results:
            print("没有可绘制的数据")
            return
        
        df = self.results['daily_portfolio']
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 组合价值曲线
        axes[0, 0].plot(df['date'], df['total_value'])
        axes[0, 0].set_title('投资组合价值曲线')
        axes[0, 0].set_ylabel('价值 ($)')
        axes[0, 0].grid(True)
        
        # 2. 收益率分布
        if 'return_pct' in df.columns:
            df['return_pct'].hist(bins=50, ax=axes[0, 1])
            axes[0, 1].set_title('收益率分布')
            axes[0, 1].set_xlabel('收益率')
            axes[0, 1].set_ylabel('频次')
        
        # 3. 持仓数量
        if 'num_positions' in df.columns:
            axes[1, 0].plot(df['date'], df['num_positions'])
            axes[1, 0].set_title('持仓数量')
            axes[1, 0].set_ylabel('持仓数量')
            axes[1, 0].grid(True)
        
        # 4. 组合Greeks
        greek_columns = [col for col in df.columns if col.startswith('portfolio_')]
        if greek_columns:
            for col in greek_columns[:3]:  # 只显示前3个Greeks
                axes[1, 1].plot(df['date'], df[col], label=col.replace('portfolio_', '').upper())
            axes[1, 1].set_title('投资组合Greeks')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(f"{output_dir}/backtest_results.png", dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    # 示例配置
    config = {
        'underlying_symbol': 'AAPL',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'initial_cash': 100000,
        'commission_rate': 0.001,
        'risk_management': {
            'max_loss_pct': 0.5,
            'take_profit_pct': 1.0,
            'max_positions': 10,
        },
        'position_sizing': {
            'max_position_size': 0.05,
            'kelly_fraction': 0.25
        }
    }
    
    # 创建回测框架
    framework = OptionsBacktestFramework(config)
    
    print("期权回测框架已初始化")
    print("请使用 framework.run_backtest() 开始回测")