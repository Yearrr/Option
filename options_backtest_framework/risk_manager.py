#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权回测框架 - 风险管理模块
实现止盈止损、对冲和保证金管理功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class RiskManager:
    """风险管理器 - 负责策略的风险控制"""
    
    def __init__(self, config: Dict):
        """
        初始化风险管理器
        
        Args:
            config: 风险管理配置
        """
        self.config = config
        self.max_loss_pct = config.get('max_loss_pct', 0.5)  # 最大亏损比例
        self.take_profit_pct = config.get('take_profit_pct', 1.0)  # 止盈比例
        self.max_positions = config.get('max_positions', 10)  # 最大持仓数
        self.margin_requirement = config.get('margin_requirement', 0.2)  # 保证金要求
        self.delta_hedge_threshold = config.get('delta_hedge_threshold', 0.1)  # Delta对冲阈值
        self.gamma_hedge_threshold = config.get('gamma_hedge_threshold', 0.05)  # Gamma对冲阈值
        self.vega_hedge_threshold = config.get('vega_hedge_threshold', 0.1)  # Vega对冲阈值
        
    def should_stop_loss(self, position: Dict, current_pnl: float) -> bool:
        """
        判断是否应该止损
        
        Args:
            position: 持仓信息
            current_pnl: 当前盈亏
            
        Returns:
            是否止损
        """
        initial_cost = position.get('initial_cost', 0)
        if initial_cost == 0:
            return False
            
        loss_pct = current_pnl / initial_cost if initial_cost > 0 else 0
        
        # 基本止损条件
        if loss_pct <= -self.max_loss_pct:
            return True
            
        # 基于时间价值衰减的止损
        days_to_expiry = position.get('days_to_expiry', 30)
        if days_to_expiry <= 5 and loss_pct <= -0.2:  # 临近到期且亏损20%
            return True
            
        # 基于Greeks的止损
        greeks = position.get('greeks', {})
        theta = greeks.get('theta', 0)
        
        # 如果Theta损失过大
        if theta < -initial_cost * 0.05:  # 日Theta损失超过初始成本5%
            return True
            
        return False
    
    def should_take_profit(self, position: Dict, current_pnl: float) -> bool:
        """
        判断是否应该止盈
        
        Args:
            position: 持仓信息
            current_pnl: 当前盈亏
            
        Returns:
            是否止盈
        """
        initial_cost = position.get('initial_cost', 0)
        if initial_cost == 0:
            return False
            
        profit_pct = current_pnl / initial_cost if initial_cost > 0 else 0
        
        # 基本止盈条件
        if profit_pct >= self.take_profit_pct:
            return True
            
        # 基于时间的动态止盈
        days_to_expiry = position.get('days_to_expiry', 30)
        
        # 临近到期时降低止盈标准
        if days_to_expiry <= 10:
            dynamic_profit_target = self.take_profit_pct * 0.5
            if profit_pct >= dynamic_profit_target:
                return True
                
        # 基于波动率的止盈
        if profit_pct >= 0.3:  # 已有30%利润
            current_iv = position.get('current_iv', 0.2)
            entry_iv = position.get('entry_iv', 0.2)
            
            # 如果是做多波动率策略，且IV下降超过20%
            if position.get('strategy_type') == 'long_volatility':
                if current_iv < entry_iv * 0.8:
                    return True
                    
            # 如果是做空波动率策略，且IV上升超过20%
            elif position.get('strategy_type') == 'short_volatility':
                if current_iv > entry_iv * 1.2:
                    return True
        
        return False
    
    def calculate_margin_requirement(self, position: Dict) -> float:
        """
        计算保证金要求
        
        Args:
            position: 持仓信息
            
        Returns:
            保证金要求
        """
        strategy_type = position.get('strategy', '')
        underlying_price = position.get('underlying_price', 100)
        
        # 基础保证金计算
        base_margin = underlying_price * self.margin_requirement
        
        if 'short' in strategy_type.lower():
            # 卖方策略需要更多保证金
            multiplier = 2.0
            
            # 根据具体策略调整
            if 'straddle' in strategy_type or 'strangle' in strategy_type:
                multiplier = 3.0  # 做空跨式策略风险更高
            elif 'naked' in strategy_type:
                multiplier = 4.0  # 裸卖期权风险最高
                
        else:
            # 买方策略最大损失为权利金
            multiplier = 0.1  # 只需要少量保证金
            
        return base_margin * multiplier
    
    def check_position_limits(self, current_positions: List[Dict], new_position: Dict) -> bool:
        """
        检查是否超过持仓限制
        
        Args:
            current_positions: 当前持仓列表
            new_position: 新增持仓
            
        Returns:
            是否可以开仓
        """
        # 检查总持仓数
        if len(current_positions) >= self.max_positions:
            return False
            
        # 检查单个标的持仓集中度
        underlying = new_position.get('underlying', '')
        same_underlying_count = sum(1 for pos in current_positions 
                                  if pos.get('underlying') == underlying)
        
        if same_underlying_count >= self.max_positions // 2:  # 单个标的不超过总持仓的一半
            return False
            
        # 检查同向策略集中度
        strategy_type = new_position.get('strategy_type', '')
        same_strategy_count = sum(1 for pos in current_positions 
                                if pos.get('strategy_type') == strategy_type)
        
        if same_strategy_count >= self.max_positions // 3:  # 同类策略不超过1/3
            return False
            
        return True
    
    def calculate_hedge_requirements(self, portfolio_greeks: Dict) -> Dict[str, float]:
        """
        计算对冲需求
        
        Args:
            portfolio_greeks: 投资组合Greeks
            
        Returns:
            对冲需求字典
        """
        hedge_requirements = {
            'delta_hedge': 0,
            'gamma_hedge': 0,
            'vega_hedge': 0,
            'theta_hedge': 0
        }
        
        # Delta对冲
        portfolio_delta = portfolio_greeks.get('delta', 0)
        if abs(portfolio_delta) > self.delta_hedge_threshold:
            hedge_requirements['delta_hedge'] = -portfolio_delta
            
        # Gamma对冲
        portfolio_gamma = portfolio_greeks.get('gamma', 0)
        if abs(portfolio_gamma) > self.gamma_hedge_threshold:
            hedge_requirements['gamma_hedge'] = -portfolio_gamma
            
        # Vega对冲
        portfolio_vega = portfolio_greeks.get('vega', 0)
        if abs(portfolio_vega) > self.vega_hedge_threshold:
            hedge_requirements['vega_hedge'] = -portfolio_vega
            
        return hedge_requirements
    
    def execute_delta_hedge(self, delta_exposure: float, underlying_price: float) -> Dict:
        """
        执行Delta对冲
        
        Args:
            delta_exposure: Delta敞口
            underlying_price: 标的价格
            
        Returns:
            对冲交易信息
        """
        # 通过买卖标的股票进行Delta中性化
        shares_to_trade = -delta_exposure  # 相反方向交易
        trade_value = shares_to_trade * underlying_price
        
        return {
            'hedge_type': 'delta',
            'shares': shares_to_trade,
            'trade_value': trade_value,
            'action': 'buy' if shares_to_trade > 0 else 'sell'
        }
    
    def execute_gamma_hedge(self, gamma_exposure: float, available_options: List[Dict]) -> Dict:
        """
        执行Gamma对冲
        
        Args:
            gamma_exposure: Gamma敞口
            available_options: 可用期权列表
            
        Returns:
            对冲交易信息
        """
        # 选择最适合的期权进行Gamma对冲
        best_option = None
        min_cost = float('inf')
        
        for option in available_options:
            option_gamma = option.get('gamma', 0)
            option_price = option.get('price', 0)
            
            if option_gamma != 0:
                # 计算需要的期权数量
                contracts_needed = -gamma_exposure / option_gamma
                cost = abs(contracts_needed) * option_price
                
                if cost < min_cost:
                    min_cost = cost
                    best_option = {
                        'option_info': option,
                        'contracts': contracts_needed,
                        'cost': cost
                    }
        
        if best_option:
            return {
                'hedge_type': 'gamma',
                'option': best_option['option_info'],
                'contracts': best_option['contracts'],
                'cost': best_option['cost'],
                'action': 'buy' if best_option['contracts'] > 0 else 'sell'
            }
        
        return {}
    
    def execute_vega_hedge(self, vega_exposure: float, available_options: List[Dict]) -> Dict:
        """
        执行Vega对冲
        
        Args:
            vega_exposure: Vega敞口
            available_options: 可用期权列表
            
        Returns:
            对冲交易信息
        """
        # 选择具有较高Vega的期权进行对冲
        best_option = None
        max_vega_efficiency = 0
        
        for option in available_options:
            option_vega = option.get('vega', 0)
            option_price = option.get('price', 0)
            
            if option_vega != 0 and option_price > 0:
                vega_efficiency = abs(option_vega) / option_price
                
                if vega_efficiency > max_vega_efficiency:
                    max_vega_efficiency = vega_efficiency
                    contracts_needed = -vega_exposure / option_vega
                    
                    best_option = {
                        'option_info': option,
                        'contracts': contracts_needed,
                        'cost': abs(contracts_needed) * option_price
                    }
        
        if best_option:
            return {
                'hedge_type': 'vega',
                'option': best_option['option_info'],
                'contracts': best_option['contracts'],
                'cost': best_option['cost'],
                'action': 'buy' if best_option['contracts'] > 0 else 'sell'
            }
        
        return {}
    
    def monitor_portfolio_risk(self, portfolio: Dict) -> Dict[str, Any]:
        """
        监控投资组合风险
        
        Args:
            portfolio: 投资组合信息
            
        Returns:
            风险监控报告
        """
        risk_report = {
            'total_value': 0,
            'total_margin': 0,
            'max_loss': 0,
            'concentration_risk': {},
            'greeks_exposure': {},
            'warnings': []
        }
        
        positions = portfolio.get('positions', [])
        
        # 计算总价值和保证金
        for position in positions:
            risk_report['total_value'] += position.get('current_value', 0)
            risk_report['total_margin'] += self.calculate_margin_requirement(position)
            risk_report['max_loss'] += position.get('max_loss', 0)
        
        # 集中度风险分析
        underlying_exposure = {}
        strategy_exposure = {}
        
        for position in positions:
            underlying = position.get('underlying', 'unknown')
            strategy = position.get('strategy', 'unknown')
            value = position.get('current_value', 0)
            
            underlying_exposure[underlying] = underlying_exposure.get(underlying, 0) + value
            strategy_exposure[strategy] = strategy_exposure.get(strategy, 0) + value
        
        # 检查集中度风险
        total_value = risk_report['total_value']
        if total_value > 0:
            for underlying, exposure in underlying_exposure.items():
                concentration = exposure / total_value
                if concentration > 0.3:  # 单个标的超过30%
                    risk_report['warnings'].append(
                        f"标的 {underlying} 集中度过高: {concentration:.1%}"
                    )
        
        risk_report['concentration_risk'] = {
            'underlying': underlying_exposure,
            'strategy': strategy_exposure
        }
        
        # Greeks敞口分析
        total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        for position in positions:
            greeks = position.get('greeks', {})
            for greek, value in greeks.items():
                if greek in total_greeks:
                    total_greeks[greek] += value
        
        risk_report['greeks_exposure'] = total_greeks
        
        # Greeks风险警告
        if abs(total_greeks['delta']) > 1.0:
            risk_report['warnings'].append(f"Delta敞口过大: {total_greeks['delta']:.2f}")
        
        if abs(total_greeks['gamma']) > 0.1:
            risk_report['warnings'].append(f"Gamma敞口过大: {total_greeks['gamma']:.3f}")
        
        if abs(total_greeks['vega']) > 10:
            risk_report['warnings'].append(f"Vega敞口过大: {total_greeks['vega']:.2f}")
        
        return risk_report
    
    def generate_risk_alerts(self, risk_report: Dict) -> List[str]:
        """
        生成风险提醒
        
        Args:
            risk_report: 风险报告
            
        Returns:
            风险提醒列表
        """
        alerts = []
        
        # 保证金使用率警告
        total_value = risk_report.get('total_value', 0)
        total_margin = risk_report.get('total_margin', 0)
        
        if total_value > 0:
            margin_usage = total_margin / total_value
            if margin_usage > 0.8:
                alerts.append(f"保证金使用率过高: {margin_usage:.1%}")
        
        # 最大损失警告
        max_loss = risk_report.get('max_loss', 0)
        if total_value > 0:
            max_loss_pct = max_loss / total_value
            if max_loss_pct > 0.2:
                alerts.append(f"最大潜在损失过大: {max_loss_pct:.1%}")
        
        # 添加已有的警告
        alerts.extend(risk_report.get('warnings', []))
        
        return alerts


class PositionSizer:
    """仓位管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_position_size = config.get('max_position_size', 0.05)  # 单个头寸最大5%
        self.kelly_fraction = config.get('kelly_fraction', 0.25)  # Kelly公式分数
        
    def calculate_position_size(self, strategy_info: Dict, portfolio_value: float, 
                              win_rate: float = 0.6, avg_win: float = 0.3, 
                              avg_loss: float = 0.15) -> int:
        """
        计算合适的仓位大小
        
        Args:
            strategy_info: 策略信息
            portfolio_value: 投资组合总价值
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损
            
        Returns:
            建议仓位大小
        """
        # Kelly公式计算最优仓位
        if avg_loss > 0:
            kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_pct = max(0, min(kelly_pct, self.kelly_fraction))  # 限制在合理范围
        else:
            kelly_pct = self.max_position_size
        
        # 基于策略类型调整
        strategy = strategy_info.get('strategy', '')
        risk_multiplier = 1.0
        
        if 'short' in strategy.lower():
            risk_multiplier = 0.5  # 卖方策略降低仓位
        elif 'ratio' in strategy.lower():
            risk_multiplier = 0.7  # 比例策略中等仓位
        
        # 计算最终仓位
        position_pct = kelly_pct * risk_multiplier
        position_pct = min(position_pct, self.max_position_size)
        
        option_price = strategy_info.get('option_price', 1)
        max_contracts = int((portfolio_value * position_pct) / (option_price * 100))
        
        return max(1, max_contracts)  # 至少1手


if __name__ == "__main__":
    # 测试风险管理器
    
    config = {
        'max_loss_pct': 0.5,
        'take_profit_pct': 1.0,
        'max_positions': 10,
        'margin_requirement': 0.2,
        'delta_hedge_threshold': 0.1,
        'gamma_hedge_threshold': 0.05,
        'vega_hedge_threshold': 0.1
    }
    
    risk_manager = RiskManager(config)
    
    # 测试持仓
    position = {
        'initial_cost': 1000,
        'days_to_expiry': 15,
        'strategy_type': 'long_volatility',
        'current_iv': 0.18,
        'entry_iv': 0.22,
        'underlying_price': 100,
        'strategy': 'long_straddle',
        'greeks': {'delta': 0.05, 'gamma': 0.03, 'theta': -5, 'vega': 15}
    }
    
    # 测试止损
    should_stop = risk_manager.should_stop_loss(position, -600)
    print(f"Should stop loss: {should_stop}")
    
    # 测试止盈
    should_profit = risk_manager.should_take_profit(position, 400)
    print(f"Should take profit: {should_profit}")
    
    # 测试保证金计算
    margin = risk_manager.calculate_margin_requirement(position)
    print(f"Margin requirement: ${margin:.2f}")
    
    # 测试对冲需求
    portfolio_greeks = {'delta': 0.15, 'gamma': 0.08, 'vega': 12, 'theta': -20}
    hedge_req = risk_manager.calculate_hedge_requirements(portfolio_greeks)
    print(f"Hedge requirements: {hedge_req}")