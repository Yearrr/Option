#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:23:09 2020

@author: yeer
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


class  interval_option():
    
    
    def  __init__(self, St,historical_prices,rebate_price,B1,B2,T,t,r,sigma,I):
        
        self.St = St  # underlying spot price
        self.historical_prices = historical_prices #历史价格
        self.rebate_price = rebate_price  # 绝对价格
        self.B1 = B1  #lower barrier price
        self.B2 = B2  #higher barrier price
        self.T = T  #expirtion period
        self.r = r # risk free rate
        self.I = I #number of replications
        self.t = t #current time 
        self.dt = (T-t)/252/(T-t) #dt
        self.sigma = sigma #volatility
         
        self.path = self.get_path_mc()

    
         
    def  get_path_mc(self):
        
        
        np.random.seed(2000000)
        Z = np.random.standard_normal((self.T-self.t+1,self.I))
        self.path = self.St * np.exp(np.cumsum((self.r - 0.5*self.sigma **2) *self.dt + self.sigma *np.sqrt(self.dt) *Z,axis=0))#生成价格路径
        self.path[0] = St
        
        return self.path
        
        
        
        
    def get_option_price_mc(self):
        
        
        df = pd.DataFrame(self.path)
        new_numin = df[(df.loc[1:,:]>=self.B1)&(df.loc[1:,:]<=self.B2)].count() #计算模拟价格在区间的日子
        his_numin = len([s for s in self.historical_prices if s>=self.B1 and s<=self.B2])#计算历史价格在区间的日子
        payoff = self.rebate_price *(new_numin+his_numin)/self.T #每个路径的payoff
        C = payoff.mean()*np.exp(-self.r *(self.T-self.t)/252)
         
              
        return C



    def  get_option_price_analysis(self):
    
        n=0
        #计算每天价格在区间内的概率然后累加
        for time in range(1,self.T+1-t):
            time = time/252
            d1 = (np.log(B1/self.St)-(r-0.5*self.sigma**2)*time)/(sigma*np.sqrt(time))
            d2 = (np.log(B2/self.St)-(r-0.5*self.sigma**2)*time)/(sigma*np.sqrt(time))
            p = stats.norm.cdf(d2, 0.0, 1.0)-stats.norm.cdf(d1, 0.0, 1.0)
            n+=p
        n = n + len([s for s in self.historical_prices if s>=self.B1 and s<=self.B2])
        C = rebate_price*n/T*np.exp(-self.r*(self.T-self.t)/252)
    
        return C


if __name__ == '__main__':  
    
    St = 100 
    B1 = 90
    B2 =110
    rebate_price = 5
    I = 100000
    T =  20  
    r = 0.05
    sigma = 0.2
    t = 3
    historical_prices=[100,98,95]
    engine = interval_option(St,historical_prices, rebate_price, B1, B2, T, t,r, sigma, I)
    C = engine.get_option_price_mc()
    


    
    
    
        
 
        
    
    
    


