# 千千量化
# 微信公众号  千千的量化世界    关注公众号 + 回复回测模板 = 获取源码下载链接
# 需要回测用的历史数据的朋友 可以加我微信 或者 在千千量化策略分享群中at我

import pandas as pd
import numpy as np

def s_dc(df):
    signal = []
    if df.iloc[-2]['close'] >= df.iloc[-2]['DC_upp']:
        signal.append('kd')
    if df.iloc[-2]['close'] <= df.iloc[-2]['DC_mid'] and df.iloc[-3]['close'] > df.iloc[-3]['DC_mid']:
        signal.append('pd')
    if df.iloc[-2]['close'] <= df.iloc[-2]['DC_low']:
        signal.append('kk')
    if df.iloc[-2]['close'] >= df.iloc[-2]['DC_mid'] and df.iloc[-3]['close'] < df.iloc[-3]['DC_mid']:
        signal.append('pk')
    return signal, 's_dc'
New
def backtest(df, m, name, symbol, rule_type):
    '''
        df: 历史数据
        m： 策略参数
        name: 策略名称
        symbol: 交易对
        rule_type: 时间周期
    '''
    # 深拷贝
    df_test = df.copy()

    # 计算指标
    df_test.loc[:,'DC_upp'] = df_test.loc[:, 'close'].rolling(m, min_periods=1).max()
    df_test.loc[:,'DC_low'] = df_test.loc[:, 'close'].rolling(m, min_periods=1).min()
    df_test.loc[:,'DC_mid'] = (df_test.loc[:, 'DC_upp'] + df_test.loc[:, 'DC_low'])/2

    ### 画图检查数据
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(df_test.loc[:,'DC_upp'])
    plt.plot(df_test.loc[:,'DC_low'])
    plt.plot(df_test.loc[:,'DC_mid'])
    plt.plot(df_test.loc[:,'close'])
    plt.grid()
    plt.show()

    ### 循环式回测（简单， 避免很多坑，接近实盘环境）  矢量式回测  向量式（快，未来函数）
    # 窗口宽度
    window = 20
    # K线数量
    nrows = df_test.shape[0]
    # 初始持仓 1 -1 0
    position = 0
    # 手续费
    fee = 2 / 1000
    # 初始资产
    coin = 0
    cash = 10000
    # 交易数量
    one_hand = 1
    # 净值列表
    total = []
    # 交易次数
    ntrade = 0
    # 
    for start_index in range(nrows - window):

        print(str(start_index * 100//(nrows - window))+'%')

        df_real = df_test.iloc[start_index:start_index+window]
        
        # 调用策略函数产生交易信号
        if name =='dc':signal, s_name = s_dc(df_real)

        # 获取当前开盘价格
        price = df_real.iloc[-1]['open']

        # 当持仓为0 且 存在开多信号时
        if 'kd' in signal and position == 0:
            coin += one_hand
            cash -= one_hand * price * (1 + fee)
            position = 1
            ntrade += 1
            print('kd' + str(price))

        # 当持仓为1 且 存在平多信号时
        if 'pd' in signal and position == 1:
            coin -= one_hand 
            cash += one_hand * price * (1 - fee)
            position = 0
            print('pd' + str(price) + '\n')

        # 当持仓为0 且 存在开空信号时
        if 'kk' in signal and position == 0:
            coin -= one_hand 
            cash += one_hand * price * (1 - fee)
            position = -1
            ntrade += 1
            print('kk' + str(price))

        # 当持仓为-1 且 存在平空信号时
        if 'pk' in signal and position == -1:
            coin += one_hand
            cash -= one_hand * price * (1 + fee)
            position = 0
            print('pk' + str(price) + '\n')
        
        total.append(cash + coin * price)

    ## 显示结果
    total = np.array(total)
    # 打开画布 图片布局为2行1列
    plt.figure()
    # 第一张图
    plt.subplot(211)
    # 画出开盘价时间序列
    plt.plot(df_test.loc[:,'open'])
    # 加网格
    plt.grid()
    # 第二张图
    plt.subplot(212)
    # 画出净值曲线
    plt.plot(total)
    # 加网格
    plt.grid()
    plt.title('ntrade:'+str(ntrade)+' total:'+str(round(total[-1],1))+' std:'+str(round(total.std(),1)))
    plt.show()
    # 保存结果
    # plt.savefig('backtest_%s_%s_%s_%d.png'%(symbol, time ,s_name, m))
    # plt.close()



if __name__ == "__main__":

    symbol = 'btc'

    # 导入历史数据 1min
    df = pd.read_hdf(f'{symbol}usdt.h5')

    # 转换到所需要的时间粒度 15min = 15 * 1min 
    rule_type = '60T'
    period_df = df.resample(rule = rule_type, on='candle_begin_time', label='left', closed='left').agg(
        {'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        })
    # 去除一天都没有交易的周期
    period_df.dropna(subset=['open'], inplace=True)  
    # 去除成交量为0的交易周期
    period_df = period_df[period_df['volume'] > 0]
    # 重置标签
    period_df.reset_index(inplace=True)
    df = period_df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]
    df.reset_index(inplace=True, drop=True)

    # 回测
    # 回测参数
    for m in range(100, 300, 20):
        backtest(
            df,
            m,
            'dc',
            symbol,
            rule_type
        )