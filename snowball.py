#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 20:54:49 2020

@author: yeer
"""


# coupon = 0.36 #票息，年化
# dt = T/M/252
# np.random.seed(20000)
# Z = np.random.standard_normal((I,M+1))
# # path = S0* np.exp(np.cumsum((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z,axis=0))
# # path[0] = S0
# f = np.zeros((I,1))
# notional = 100  #本金
# obs_out = [22,44,66,88,110,132] #敲出观察日   
  
# for i in range(I):    
#     Z = np.random.standard_normal(M+1)
#     path = S0*(np.exp(np.cumsum((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z))) 
#     path[0] =S0
#     df = pd.DataFrame({'path':path})
#     df['dis_factor'] = np.exp(-r*df.index/252)
#     index_in = list(df[df.path<=B1].index) #敲入日
#     index_out= list(df.loc[obs_out,:][df.path>=B2].index) #敲出日
#     df['coupon'] = df.loc[index_out,:].loc[:,'dis_factor']*coupon #敲出日的coupon贴现值
#     df.fillna(0,inplace=True)
    
#     #没有敲入
#     if len(index_in)==0:
#         payoff = coupon*df['dis_factor'].values[-1]

#     #有敲入有敲出
#     if len(index_in)>0 and len(index_out)>0 and index_out[-1]>index_in[0] :
#         payoff= df.loc[index_in[0]:,][df.coupon>0].coupon.values[0]
    
#     #有敲入没敲出
#     if len(index_in)>0 and len(index_out)==0:
#         payoff = 1 - min(df.path.values[-1]/S0,1)
       
#     #敲出在敲入前
#     if len(index_out)>0 and len(index_in)>0 and index_out[-1]<index_in[0]:
#         payoff = 1 - min(df.path.values[-1]/S0,1)

#     f[i] = payoff  
# price  = np.mean(f)     