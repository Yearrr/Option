# 期权回测框架

这是一个专门针对场内期权策略的回测框架，特别适合测试卖出看涨价差（sell call spread）和其他期权组合策略。

## 核心功能

### 🎯 主要策略支持

1. **卖出看涨价差 (Sell Call Spread)**
   - 卖出虚值看涨期权 + 买入更高行权价的看涨期权
   - 获得时间价值衰减和隐含波动率下降的收益
   - 有限风险和有限收益

2. **比率价差 (Ratio Spreads)**
   - 支持1:2、1:3等不同比例的期权组合
   - 适合对标的价格区间有明确预期的情况

3. **铁鹰策略 (Iron Condor)**
   - 结合看涨和看跌价差的中性策略
   - 适合低波动率环境

4. **蝶式策略 (Butterfly)**
   - 三个行权价的期权组合
   - 适合预期标的价格在特定点位附近波动

5. **宽跨式策略 (Strangle)**
   - 同时卖出虚值看涨和看跌期权
   - 适合低波动率环境下获取时间价值

### 🛠️ 核心模块

#### 1. 期权定价引擎 (OptionPricer)
- **Black-Scholes模型**: 标准欧式期权定价
- **蒙特卡洛模拟**: 复杂期权（如障碍期权、雪球期权）定价
- **希腊字母计算**: Delta, Gamma, Theta, Vega

#### 2. 风险管理模块 (RiskManager)
- **止盈止损**: 可配置的盈利目标和损失限制
- **持仓规模控制**: 最大持仓数量和资金限制
- **Delta对冲阈值**: 自动触发对冲的Delta暴露水平

#### 3. 对冲引擎 (HedgeEngine)
- **Delta中性对冲**: 通过买卖标的资产对冲方向性风险
- **动态对冲频率**: 支持每日、每周等不同对冲频率
- **对冲比例调整**: 可配置的对冲比例

#### 4. 回测引擎 (BacktestEngine)
- **历史数据回放**: 支持分钟级到日级数据
- **实时P&L计算**: 每日更新持仓价值和盈亏
- **多策略并行**: 同时运行多个期权策略

### 📊 数据记录与分析

#### 交易记录 (TradeRecord)
```python
@dataclass
class TradeRecord:
    date: str              # 交易日期
    symbol: str            # 期权代码
    action: str            # 买入/卖出/对冲
    quantity: int          # 交易数量
    price: float           # 交易价格
    trade_type: str        # 开仓/平仓/止盈/止损
    pnl: float            # 盈亏
    commission: float     # 手续费
```

#### 持仓记录 (PositionRecord)
```python
@dataclass
class PositionRecord:
    date: str              # 日期
    symbol: str            # 期权代码
    quantity: int          # 持仓数量
    market_value: float    # 市场价值
    unrealized_pnl: float  # 未实现盈亏
    realized_pnl: float    # 已实现盈亏
    delta: float           # Delta暴露
    gamma: float           # Gamma
    theta: float           # Theta衰减
    vega: float            # Vega风险
```

#### 每日盈亏记录
- 总资产价值
- 现金余额
- 持仓市值
- 未实现盈亏
- 组合Delta
- 每日收益率

## 🚀 快速开始

### 安装依赖
```bash
pip install numpy pandas matplotlib scipy
```

### 简单示例 - 卖出看涨价差
```bash
python3 simple_demo.py
```

### 完整回测框架
```python
from option_spread_strategy import SpreadBacktestEngine, SellCallSpreadStrategy

# 创建回测引擎
engine = SpreadBacktestEngine(initial_capital=100000)

# 添加市场数据
engine.add_market_data(market_data)

# 运行卖出看涨价差策略
strategy_params = {
    'signal_frequency': 10,      # 每10天评估开仓
    'contracts_per_trade': 2,    # 每次2手
    'max_positions': 5           # 最大5个持仓
}
engine.run_sell_call_spread_backtest(strategy_params)

# 获取绩效分析
summary = engine.get_performance_summary()
engine.plot_results()
engine.export_records()
```

### 多策略组合测试
```python
from advanced_option_strategies import MultiStrategyBacktest

engine = MultiStrategyBacktest(initial_capital=200000)
engine.add_market_data(market_data)

# 配置多个策略
strategy_configs = {
    'sell_call_spread': {
        'frequency': 12,
        'max_positions': 4,
        'iv_threshold': 0.22
    },
    'iron_condor': {
        'frequency': 20,
        'max_positions': 3,
        'iv_threshold': 0.25
    }
}

engine.run_multi_strategy_backtest(strategy_configs)
```

## 📈 绩效分析指标

### 基础指标
- **总收益率**: (最终价值 - 初始资金) / 初始资金
- **胜率**: 盈利交易数 / 总交易数
- **盈亏比**: 平均盈利 / 平均亏损
- **最大回撤**: 资金曲线的最大下跌幅度

### 期权专用指标
- **平均持仓天数**: 期权策略的平均持有时间
- **IV变化影响**: 隐含波动率变化对盈亏的影响
- **时间价值衰减收益**: Theta带来的收益
- **波动率收益**: Vega变化带来的收益

### 风险指标
- **VaR (95%, 99%)**: 风险价值
- **最大连续亏损天数**: 风险控制指标
- **夏普比率**: 风险调整收益
- **卡尔玛比率**: 收益/最大回撤

## 🎨 可视化分析

框架提供多种图表分析：

1. **资金曲线**: 总资产随时间变化
2. **收益率分布**: 每日收益率直方图
3. **回撤分析**: 资金回撤时间序列
4. **持仓分析**: 持仓数量和价值变化
5. **希腊字母暴露**: Delta, Gamma等风险暴露
6. **IV vs 盈亏关系**: 隐含波动率变化与交易盈亏的相关性

## ⚙️ 配置参数

### 策略参数
```python
@dataclass
class BacktestConfig:
    initial_capital: float = 1000000     # 初始资金
    commission_rate: float = 0.001       # 手续费率
    stop_loss_pct: float = 0.2          # 止损比例
    take_profit_pct: float = 0.5        # 止盈比例
    delta_hedge_threshold: float = 0.1   # Delta对冲阈值
```

### 卖出看涨价差专用参数
```python
class SellCallSpreadStrategy:
    def __init__(self, 
                 short_strike_delta: float = 0.3,    # 卖出期权Delta
                 long_strike_delta: float = 0.15,    # 买入期权Delta
                 days_to_expiry: int = 30,           # 到期天数
                 profit_target: float = 0.5,        # 止盈目标
                 stop_loss: float = 2.0):            # 止损倍数
```

## 📁 文件结构

```
option-backtest-framework/
├── option_backtest_framework.py      # 基础回测框架
├── option_spread_strategy.py         # 价差策略专用模块
├── advanced_option_strategies.py     # 高级策略组合
├── config.py                        # 配置文件
├── simple_demo.py                   # 简化演示
├── demo_sell_call_spread.py         # 卖出看涨价差演示
├── requirements.txt                 # 依赖包
└── README.md                       # 说明文档
```

## 🎯 使用场景

### 1. 策略开发与验证
- 测试新的期权组合策略
- 验证策略的历史表现
- 优化策略参数

### 2. 风险管理
- 评估策略的最大回撤
- 测试不同市场环境下的表现
- 分析希腊字母风险暴露

### 3. 投资组合优化
- 比较不同策略的风险收益特征
- 确定最优的资金分配比例
- 构建多策略投资组合

### 4. 教学与研究
- 理解期权策略的盈亏特征
- 学习期权定价和风险管理
- 研究市场波动率的影响

## ⚠️ 重要提示

1. **历史回测不代表未来表现**: 回测结果仅供参考，实际交易可能面临不同的市场环境
2. **流动性假设**: 框架假设所有期权都有充足流动性，实际交易中需考虑买卖价差
3. **税费成本**: 框架中的手续费是简化处理，实际交易还需考虑税费等成本
4. **市场冲击**: 大额交易可能对市场价格产生影响，框架未考虑此因素

## 🔧 扩展开发

框架采用模块化设计，便于扩展：

### 添加新策略
```python
class NewStrategy:
    def create_strategy(self, spot_price: float, date: str, iv: float) -> SpreadStrategy:
        # 实现新策略逻辑
        pass
```

### 自定义风险管理
```python
class CustomRiskManager(RiskManager):
    def check_custom_rule(self, position) -> bool:
        # 实现自定义风险规则
        pass
```

### 添加新的定价模型
```python
class CustomPricer(OptionPricer):
    def exotic_option_price(self, params) -> float:
        # 实现新的期权定价方法
        pass
```

## 📞 技术支持

如有问题或建议，请参考代码注释或查看示例用法。框架设计注重实用性和扩展性，适合期权交易者和量化研究人员使用。 
