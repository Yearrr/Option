#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权回测框架 - 期权定价模块
实现期权定价和Greeks计算
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class OptionPricer:
    """期权定价器 - 实现各种期权定价模型"""
    
    @staticmethod
    def black_scholes(S: float, K: float, T: float, r: float, sigma: float, 
                     option_type: str = "call") -> float:
        """
        Black-Scholes期权定价公式
        
        Args:
            S: 标的价格
            K: 行权价
            T: 到期时间 (年)
            r: 无风险利率
            sigma: 波动率
            option_type: 期权类型 ("call" or "put")
            
        Returns:
            期权价格
        """
        if T <= 0:
            if option_type.lower() == "call":
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == "call":
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        
        return max(price, 0)
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                        option_type: str = "call") -> Dict[str, float]:
        """
        计算期权Greeks
        
        Args:
            S: 标的价格
            K: 行权价
            T: 到期时间 (年)
            r: 无风险利率
            sigma: 波动率
            option_type: 期权类型
            
        Returns:
            Greeks字典
        """
        if T <= 0:
            return {
                'delta': 0, 'gamma': 0, 'theta': 0, 
                'vega': 0, 'rho': 0
            }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type.lower() == "call":
            delta = stats.norm.cdf(d1)
        else:
            delta = stats.norm.cdf(d1) - 1
        
        # Gamma
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        theta_common = -(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type.lower() == "call":
            theta = theta_common - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            theta = theta_common + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
        theta = theta / 365  # 转换为每日Theta
        
        # Vega
        vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100  # 转换为1%波动率变化
        
        # Rho
        if option_type.lower() == "call":
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, 
                          r: float, option_type: str = "call") -> float:
        """
        计算隐含波动率
        
        Args:
            market_price: 市场价格
            S: 标的价格
            K: 行权价
            T: 到期时间
            r: 无风险利率
            option_type: 期权类型
            
        Returns:
            隐含波动率
        """
        if T <= 0:
            return 0
        
        def objective(sigma):
            return OptionPricer.black_scholes(S, K, T, r, sigma, option_type) - market_price
        
        try:
            iv = brentq(objective, 0.001, 5.0)
            return iv
        except:
            return 0.2  # 默认值
    
    @staticmethod
    def american_option_binomial(S: float, K: float, T: float, r: float, 
                               sigma: float, n_steps: int = 100, 
                               option_type: str = "call") -> float:
        """
        二项树美式期权定价
        
        Args:
            S: 标的价格
            K: 行权价
            T: 到期时间
            r: 无风险利率
            sigma: 波动率
            n_steps: 时间步数
            option_type: 期权类型
            
        Returns:
            期权价格
        """
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        # 构建价格树
        prices = np.zeros((n_steps + 1, n_steps + 1))
        for i in range(n_steps + 1):
            for j in range(i + 1):
                prices[j, i] = S * (u**j) * (d**(i - j))
        
        # 计算到期日期权价值
        option_values = np.zeros((n_steps + 1, n_steps + 1))
        for j in range(n_steps + 1):
            if option_type.lower() == "call":
                option_values[j, n_steps] = max(0, prices[j, n_steps] - K)
            else:
                option_values[j, n_steps] = max(0, K - prices[j, n_steps])
        
        # 向后递推
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                # 欧式价值
                european_value = np.exp(-r * dt) * (p * option_values[j + 1, i + 1] + 
                                                   (1 - p) * option_values[j, i + 1])
                
                # 内在价值 (美式期权可提前执行)
                if option_type.lower() == "call":
                    intrinsic_value = max(0, prices[j, i] - K)
                else:
                    intrinsic_value = max(0, K - prices[j, i])
                
                option_values[j, i] = max(european_value, intrinsic_value)
        
        return option_values[0, 0]


class StrategyPricer:
    """期权策略定价器 - 处理复合期权策略"""
    
    def __init__(self, pricer: OptionPricer):
        self.pricer = pricer
    
    def straddle_price(self, S: float, K: float, T: float, r: float, 
                      sigma: float) -> Dict[str, float]:
        """
        跨式组合定价 (Long Straddle)
        
        Args:
            S: 标的价格
            K: 行权价
            T: 到期时间
            r: 无风险利率
            sigma: 波动率
            
        Returns:
            策略价格和Greeks
        """
        call_price = self.pricer.black_scholes(S, K, T, r, sigma, "call")
        put_price = self.pricer.black_scholes(S, K, T, r, sigma, "put")
        
        call_greeks = self.pricer.calculate_greeks(S, K, T, r, sigma, "call")
        put_greeks = self.pricer.calculate_greeks(S, K, T, r, sigma, "put")
        
        # 策略总价格
        total_price = call_price + put_price
        
        # 策略Greeks
        strategy_greeks = {
            'delta': call_greeks['delta'] + put_greeks['delta'],
            'gamma': call_greeks['gamma'] + put_greeks['gamma'],
            'theta': call_greeks['theta'] + put_greeks['theta'],
            'vega': call_greeks['vega'] + put_greeks['vega'],
            'rho': call_greeks['rho'] + put_greeks['rho']
        }
        
        return {
            'total_price': total_price,
            'call_price': call_price,
            'put_price': put_price,
            'greeks': strategy_greeks
        }
    
    def strangle_price(self, S: float, K_call: float, K_put: float, T: float, 
                      r: float, sigma: float) -> Dict[str, float]:
        """
        宽跨式组合定价 (Long Strangle)
        """
        call_price = self.pricer.black_scholes(S, K_call, T, r, sigma, "call")
        put_price = self.pricer.black_scholes(S, K_put, T, r, sigma, "put")
        
        call_greeks = self.pricer.calculate_greeks(S, K_call, T, r, sigma, "call")
        put_greeks = self.pricer.calculate_greeks(S, K_put, T, r, sigma, "put")
        
        total_price = call_price + put_price
        
        strategy_greeks = {
            'delta': call_greeks['delta'] + put_greeks['delta'],
            'gamma': call_greeks['gamma'] + put_greeks['gamma'],
            'theta': call_greeks['theta'] + put_greeks['theta'],
            'vega': call_greeks['vega'] + put_greeks['vega'],
            'rho': call_greeks['rho'] + put_greeks['rho']
        }
        
        return {
            'total_price': total_price,
            'call_price': call_price,
            'put_price': put_price,
            'greeks': strategy_greeks
        }
    
    def bull_call_spread(self, S: float, K_long: float, K_short: float, 
                        T: float, r: float, sigma: float) -> Dict[str, float]:
        """
        牛市看涨价差 (Bull Call Spread)
        """
        long_call = self.pricer.black_scholes(S, K_long, T, r, sigma, "call")
        short_call = self.pricer.black_scholes(S, K_short, T, r, sigma, "call")
        
        long_greeks = self.pricer.calculate_greeks(S, K_long, T, r, sigma, "call")
        short_greeks = self.pricer.calculate_greeks(S, K_short, T, r, sigma, "call")
        
        net_cost = long_call - short_call
        max_profit = (K_short - K_long) - net_cost
        max_loss = net_cost
        
        strategy_greeks = {
            'delta': long_greeks['delta'] - short_greeks['delta'],
            'gamma': long_greeks['gamma'] - short_greeks['gamma'],
            'theta': long_greeks['theta'] - short_greeks['theta'],
            'vega': long_greeks['vega'] - short_greeks['vega'],
            'rho': long_greeks['rho'] - short_greeks['rho']
        }
        
        return {
            'net_cost': net_cost,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'long_call': long_call,
            'short_call': short_call,
            'greeks': strategy_greeks
        }
    
    def ratio_spread(self, S: float, K_long: float, K_short: float, T: float, 
                    r: float, sigma: float, ratio: int = 2, 
                    option_type: str = "call") -> Dict[str, float]:
        """
        比例价差 (Ratio Spread)
        
        Args:
            ratio: 卖出数量/买入数量比例
        """
        long_option = self.pricer.black_scholes(S, K_long, T, r, sigma, option_type)
        short_option = self.pricer.black_scholes(S, K_short, T, r, sigma, option_type)
        
        long_greeks = self.pricer.calculate_greeks(S, K_long, T, r, sigma, option_type)
        short_greeks = self.pricer.calculate_greeks(S, K_short, T, r, sigma, option_type)
        
        net_cost = long_option - ratio * short_option
        
        strategy_greeks = {
            'delta': long_greeks['delta'] - ratio * short_greeks['delta'],
            'gamma': long_greeks['gamma'] - ratio * short_greeks['gamma'],
            'theta': long_greeks['theta'] - ratio * short_greeks['theta'],
            'vega': long_greeks['vega'] - ratio * short_greeks['vega'],
            'rho': long_greeks['rho'] - ratio * short_greeks['rho']
        }
        
        return {
            'net_cost': net_cost,
            'long_option': long_option,
            'short_option': short_option,
            'ratio': ratio,
            'greeks': strategy_greeks
        }


if __name__ == "__main__":
    # 测试期权定价
    S = 100  # 标的价格
    K = 100  # 行权价
    T = 0.25  # 3个月到期
    r = 0.05  # 5%无风险利率
    sigma = 0.2  # 20%波动率
    
    pricer = OptionPricer()
    
    # 测试Black-Scholes定价
    call_price = pricer.black_scholes(S, K, T, r, sigma, "call")
    put_price = pricer.black_scholes(S, K, T, r, sigma, "put")
    
    print(f"Call Price: {call_price:.4f}")
    print(f"Put Price: {put_price:.4f}")
    
    # 测试Greeks
    call_greeks = pricer.calculate_greeks(S, K, T, r, sigma, "call")
    print(f"Call Greeks: {call_greeks}")
    
    # 测试策略定价
    strategy_pricer = StrategyPricer(pricer)
    straddle = strategy_pricer.straddle_price(S, K, T, r, sigma)
    print(f"Straddle Price: {straddle['total_price']:.4f}")