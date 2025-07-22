#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权回测框架 - 数据管理模块
负责历史数据的获取、存储和管理
"""

import pandas as pd
import numpy as np
import yfinance as yf
import tushare as ts
from datetime import datetime, timedelta
import os
import pickle
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataManager:
    """数据管理器 - 负责期权和标的资产数据的获取与管理"""
    
    def __init__(self, data_source: str = "yfinance", cache_dir: str = "data_cache"):
        """
        初始化数据管理器
        
        Args:
            data_source: 数据源 ("yfinance", "tushare")
            cache_dir: 缓存目录
        """
        self.data_source = data_source
        self.cache_dir = cache_dir
        self.underlying_data = {}
        self.options_data = {}
        
        # 创建缓存目录
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def get_underlying_data(self, symbol: str, start_date: str, end_date: str, 
                          force_update: bool = False) -> pd.DataFrame:
        """
        获取标的资产历史数据
        
        Args:
            symbol: 标的代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            force_update: 是否强制更新
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        cache_file = f"{self.cache_dir}/{symbol}_underlying_{start_date}_{end_date}.pkl"
        
        # 检查缓存
        if not force_update and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        try:
            if self.data_source == "yfinance":
                data = yf.download(symbol, start=start_date, end=end_date)
            elif self.data_source == "tushare":
                # 需要设置tushare token
                data = self._get_tushare_data(symbol, start_date, end_date)
            else:
                raise ValueError(f"不支持的数据源: {self.data_source}")
            
            # 数据清洗
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            
            # 添加收益率和波动率
            data['Returns'] = data['Adj Close'].pct_change()
            data['Log_Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
            
            # 计算历史波动率 (21日滚动)
            data['HV_21'] = data['Returns'].rolling(21).std() * np.sqrt(252)
            
            # 缓存数据
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.underlying_data[symbol] = data
            return data
            
        except Exception as e:
            print(f"获取标的数据失败: {e}")
            return pd.DataFrame()
    
    def _get_tushare_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取Tushare数据 (需要配置token)"""
        # 这里需要用户自己配置tushare token
        # ts.set_token('your_token_here')
        pro = ts.pro_api()
        
        # 转换日期格式
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        
        # 获取日线数据
        df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        
        # 数据格式转换
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date').sort_index()
        
        # 重命名列以匹配yfinance格式
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'vol': 'Volume'
        })
        df['Adj Close'] = df['Close']
        
        return df
    
    def get_options_chain(self, underlying: str, expiry_date: str, 
                         option_type: str = "both") -> pd.DataFrame:
        """
        获取期权链数据
        
        Args:
            underlying: 标的代码
            expiry_date: 到期日
            option_type: 期权类型 ("call", "put", "both")
            
        Returns:
            期权链数据
        """
        try:
            ticker = yf.Ticker(underlying)
            options_chain = ticker.option_chain(expiry_date)
            
            if option_type == "call":
                return options_chain.calls
            elif option_type == "put":
                return options_chain.puts
            else:
                # 合并看涨和看跌期权
                calls = options_chain.calls.copy()
                puts = options_chain.puts.copy()
                calls['option_type'] = 'call'
                puts['option_type'] = 'put'
                return pd.concat([calls, puts], ignore_index=True)
                
        except Exception as e:
            print(f"获取期权链失败: {e}")
            return pd.DataFrame()
    
    def get_risk_free_rate(self, date: str = None) -> float:
        """
        获取无风险利率 (使用美国10年期国债)
        
        Args:
            date: 指定日期
            
        Returns:
            无风险利率
        """
        try:
            treasury = yf.download("^TNX", period="1y")
            if date:
                rate = treasury.loc[date, 'Close'] / 100
            else:
                rate = treasury['Close'].iloc[-1] / 100
            return rate
        except:
            return 0.02  # 默认2%
    
    def calculate_implied_volatility(self, underlying_price: float, strike: float,
                                   time_to_expiry: float, risk_free_rate: float,
                                   option_price: float, option_type: str) -> float:
        """
        计算隐含波动率
        
        Args:
            underlying_price: 标的价格
            strike: 行权价
            time_to_expiry: 到期时间 (年)
            risk_free_rate: 无风险利率
            option_price: 期权价格
            option_type: 期权类型 ("call" or "put")
            
        Returns:
            隐含波动率
        """
        try:
            from py_vollib.black_scholes.implied_volatility import implied_volatility
            
            flag = 'c' if option_type.lower() == 'call' else 'p'
            iv = implied_volatility(option_price, underlying_price, strike, 
                                  time_to_expiry, risk_free_rate, flag)
            return iv
        except:
            return 0.2  # 默认20%波动率
    
    def save_to_excel(self, data_dict: Dict[str, pd.DataFrame], filename: str):
        """
        将数据保存到Excel文件
        
        Args:
            data_dict: 数据字典
            filename: 文件名
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name)
    
    def load_from_excel(self, filename: str) -> Dict[str, pd.DataFrame]:
        """
        从Excel文件加载数据
        
        Args:
            filename: 文件名
            
        Returns:
            数据字典
        """
        return pd.read_excel(filename, sheet_name=None, index_col=0)


if __name__ == "__main__":
    # 测试数据管理器
    dm = DataManager()
    
    # 获取标的数据
    data = dm.get_underlying_data("AAPL", "2023-01-01", "2023-12-31")
    print("标的数据:")
    print(data.head())
    
    # 获取期权链
    options = dm.get_options_chain("AAPL", "2024-01-19")
    if not options.empty:
        print("\n期权链数据:")
        print(options.head())