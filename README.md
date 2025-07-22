# 期权回测框架 (Options Backtest Framework)

🚀 **专业的期权量化交易回测系统** - 为资深期权交易员量身打造

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 项目简介

这是一个完整的期权策略回测框架，专门为期权量化交易而设计。框架集成了数据管理、期权定价、信号生成、风险管理、交易执行等核心功能，支持多种期权策略的回测和分析。

### 🎯 主要特性

- ✅ **完整的数据管理系统** - 支持历史数据获取、缓存和管理
- ✅ **专业的期权定价** - Black-Scholes定价和Greeks计算
- ✅ **多样化信号生成** - 波动率、动量、偏度等多种技术分析信号
- ✅ **智能风险管理** - 止盈止损、保证金管理、自动对冲
- ✅ **强大的交易引擎** - 订单执行、持仓管理、交易记录
- ✅ **丰富的策略支持** - Straddle、Strangle、Spreads、Iron Condor等
- ✅ **详细的结果分析** - 绩效指标、图表可视化、风险报告

### 📊 支持的期权策略

| 策略类型 | 策略名称 | 说明 |
|---------|---------|------|
| 波动率策略 | Long/Short Straddle | 做多/做空跨式组合 |
| 波动率策略 | Long/Short Strangle | 做多/做空宽跨式组合 |
| 方向性策略 | Bull/Bear Call Spread | 牛市/熊市看涨价差 |
| 方向性策略 | Bull/Bear Put Spread | 牛市/熊市看跌价差 |
| 复合策略 | Iron Condor | 铁鹰策略 |
| 高级策略 | Ratio Spreads | 比例价差策略 |
| 自定义策略 | Custom Strategies | 用户自定义策略 |

## 🛠️ 安装说明

### 系统要求

- Python 3.8+
- 内存: 4GB+ (推荐8GB+)
- 硬盘: 1GB+ (用于数据缓存)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-repo/options-backtest-framework.git
cd options-backtest-framework
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **验证安装**
```bash
python -c "from options_backtest_framework import OptionsBacktestFramework; print('安装成功！')"
```

## 🚀 快速开始

### 基础示例

```python
from options_backtest_framework import (
    OptionsBacktestFramework,
    VolatilitySignalGenerator,
    MomentumSignalGenerator
)

# 1. 配置参数
config = {
    'underlying_symbol': 'AAPL',
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    'initial_cash': 100000,
    'risk_management': {
        'max_loss_pct': 0.5,
        'take_profit_pct': 1.0,
        'max_positions': 10,
    }
}

# 2. 创建框架
framework = OptionsBacktestFramework(config)

# 3. 设置信号生成器
signal_generators = [
    VolatilitySignalGenerator(lookback_period=21),
    MomentumSignalGenerator(short_window=10, long_window=30)
]

# 4. 运行回测
results = framework.run_backtest(signal_generators)

# 5. 查看结果
framework.save_results()
framework.plot_results()
```

### 运行示例

查看 `examples/` 目录中的完整示例：

```bash
cd examples
python basic_backtest_example.py
```

## 📁 项目结构

```
options_backtest_framework/
├── __init__.py                 # 包初始化文件
├── data_manager.py             # 数据管理模块
├── option_pricing.py           # 期权定价模块
├── signal_generator.py         # 信号生成模块
├── risk_manager.py             # 风险管理模块
├── trading_engine.py           # 交易引擎模块
└── backtest_framework.py       # 主框架模块

examples/
├── basic_backtest_example.py   # 基础回测示例
└── advanced_strategies.py      # 高级策略示例

docs/
├── api_reference.md            # API参考文档
├── strategy_guide.md           # 策略指南
└── user_manual.md              # 用户手册

tests/
├── test_data_manager.py        # 数据管理测试
├── test_option_pricing.py      # 期权定价测试
└── test_backtest.py            # 回测功能测试
```

## 📖 详细功能

### 1. 数据管理 (DataManager)

支持多种数据源和缓存机制：

```python
from options_backtest_framework import DataManager

# 创建数据管理器
dm = DataManager(data_source="yfinance", cache_dir="data_cache")

# 获取标的数据
data = dm.get_underlying_data("AAPL", "2023-01-01", "2023-12-31")

# 获取期权链
options = dm.get_options_chain("AAPL", "2024-01-19")

# 获取无风险利率
risk_free_rate = dm.get_risk_free_rate()
```

### 2. 期权定价 (OptionPricer)

实现了完整的期权定价模型：

```python
from options_backtest_framework import OptionPricer

pricer = OptionPricer()

# Black-Scholes定价
price = pricer.black_scholes(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")

# 计算Greeks
greeks = pricer.calculate_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type="call")

# 隐含波动率
iv = pricer.implied_volatility(market_price=5.5, S=100, K=100, T=0.25, r=0.05, option_type="call")
```

### 3. 信号生成 (SignalGenerator)

提供多种技术分析信号：

```python
from options_backtest_framework import VolatilitySignalGenerator, MomentumSignalGenerator

# 波动率信号
vol_gen = VolatilitySignalGenerator(lookback_period=21, volatility_threshold=0.25)
vol_signals = vol_gen.generate_signals(data)

# 动量信号
momentum_gen = MomentumSignalGenerator(short_window=10, long_window=30)
momentum_signals = momentum_gen.generate_signals(data)

# 组合信号
from options_backtest_framework import CompositeSignalGenerator
composite_gen = CompositeSignalGenerator([vol_gen, momentum_gen])
combined_signals = composite_gen.generate_composite_signals(data)
```

### 4. 风险管理 (RiskManager)

智能的风险控制系统：

```python
from options_backtest_framework import RiskManager

risk_config = {
    'max_loss_pct': 0.5,
    'take_profit_pct': 1.0,
    'max_positions': 10,
    'delta_hedge_threshold': 0.1
}

risk_manager = RiskManager(risk_config)

# 检查止损
should_stop = risk_manager.should_stop_loss(position, current_pnl)

# 检查止盈
should_profit = risk_manager.should_take_profit(position, current_pnl)

# 计算对冲需求
hedge_req = risk_manager.calculate_hedge_requirements(portfolio_greeks)
```

### 5. 交易引擎 (TradingEngine)

完整的交易执行系统：

```python
from options_backtest_framework import TradingEngine

engine = TradingEngine(initial_cash=100000, commission_rate=0.001)

# 下单
order_id = engine.place_order("AAPL_CALL_150_2024_01", "buy", 10, strategy="long_call")

# 执行订单
order = engine.orders[0]
success = engine.execute_order(order, market_price=5.50)

# 查看持仓
positions = engine.get_position_summary()

# 获取交易历史
trades = engine.get_trade_history()
```

## ⚙️ 配置参数

### 基础配置

```python
config = {
    # 基础参数
    'underlying_symbol': 'AAPL',           # 标的代码
    'start_date': '2023-01-01',            # 回测开始日期
    'end_date': '2023-12-31',              # 回测结束日期
    'initial_cash': 100000,                # 初始资金
    'commission_rate': 0.001,              # 佣金费率
    
    # 数据源配置
    'data_source': 'yfinance',             # 数据源
    'cache_dir': 'data_cache',             # 数据缓存目录
    
    # 风险管理参数
    'risk_management': {
        'max_loss_pct': 0.5,               # 最大亏损比例
        'take_profit_pct': 1.0,            # 止盈比例
        'max_positions': 10,               # 最大持仓数
        'delta_hedge_threshold': 0.1,      # Delta对冲阈值
    },
    
    # 仓位管理参数
    'position_sizing': {
        'max_position_size': 0.05,         # 单个头寸最大比例
        'kelly_fraction': 0.25             # Kelly公式分数
    }
}
```

### 高级配置

```python
advanced_config = {
    # 保证金参数
    'margin_parameters': {
        'initial_margin': 0.2,             # 初始保证金
        'maintenance_margin': 0.15,        # 维持保证金
        'margin_call_threshold': 0.1       # 追保阈值
    },
    
    # 对冲参数
    'hedge_parameters': {
        'auto_hedge': True,                # 自动对冲
        'hedge_frequency': 'daily',        # 对冲频率
        'delta_hedge_threshold': 0.1,      # Delta对冲阈值
        'gamma_hedge_threshold': 0.05,     # Gamma对冲阈值
        'vega_hedge_threshold': 0.1        # Vega对冲阈值
    },
    
    # 滑点和市场冲击
    'market_impact': {
        'bid_ask_spread': 0.01,            # 买卖价差
        'market_impact_factor': 0.001,     # 市场冲击因子
        'liquidity_penalty': 0.0005       # 流动性惩罚
    }
}
```

## 📊 结果分析

### 绩效指标

框架自动计算以下绩效指标：

- **收益指标**: 总收益率、年化收益率、超额收益
- **风险指标**: 年化波动率、最大回撤、VaR、CVaR
- **风险调整收益**: 夏普比率、索提诺比率、卡尔玛比率
- **交易统计**: 胜率、平均持仓时间、交易频率
- **Greeks分析**: Delta、Gamma、Theta、Vega敞口

### 可视化图表

- 投资组合价值曲线
- 收益率分布直方图
- 回撤曲线
- Greeks敞口变化
- 策略表现对比
- 风险指标热力图

### 导出格式

- CSV文件 (交易记录、每日持仓)
- JSON文件 (绩效指标、配置参数)
- PNG图表 (各种分析图表)
- PDF报告 (完整回测报告)

## 🔧 自定义扩展

### 创建自定义信号生成器

```python
from options_backtest_framework import BaseSignalGenerator
import pandas as pd

class CustomSignalGenerator(BaseSignalGenerator):
    def __init__(self, custom_param=10):
        super().__init__("CustomSignal")
        self.custom_param = custom_param
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        
        # 实现自定义逻辑
        signals['custom_indicator'] = data['Close'].rolling(self.custom_param).mean()
        signals['signal'] = 0
        signals['strategy'] = ''
        
        # 生成信号
        condition = signals['custom_indicator'] > data['Close']
        signals.loc[condition, 'signal'] = 1
        signals.loc[condition, 'strategy'] = 'custom_strategy'
        
        return signals
```

### 添加自定义策略

```python
# 在 backtest_framework.py 中添加新的策略执行方法
def _execute_custom_strategy(self, current_price: float, 
                           option_chains: Dict[str, pd.DataFrame]) -> List[Order]:
    """执行自定义策略"""
    orders = []
    
    # 实现自定义策略逻辑
    # ...
    
    return orders
```

## 🧪 测试

运行单元测试：

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_option_pricing.py -v

# 生成覆盖率报告
python -m pytest tests/ --cov=options_backtest_framework --cov-report=html
```

## 📈 性能优化

### 数据缓存

框架使用智能缓存机制：

```python
# 数据会自动缓存到本地
dm = DataManager(cache_dir="data_cache")
data = dm.get_underlying_data("AAPL", "2023-01-01", "2023-12-31")

# 强制更新缓存
data = dm.get_underlying_data("AAPL", "2023-01-01", "2023-12-31", force_update=True)
```

### 并行计算

对于大规模回测，可以启用并行计算：

```python
config = {
    'parallel_processing': True,
    'num_processes': 4,  # 使用4个进程
    'chunk_size': 1000   # 每次处理1000条数据
}
```

### 内存管理

处理大数据集时的内存优化：

```python
config = {
    'memory_optimization': True,
    'batch_size': 5000,        # 批处理大小
    'data_compression': True,   # 数据压缩
    'lazy_loading': True       # 懒加载
}
```

## 🔍 故障排除

### 常见问题

1. **数据获取失败**
   ```python
   # 检查网络连接和数据源
   dm = DataManager(data_source="yfinance")
   data = dm.get_underlying_data("AAPL", "2023-01-01", "2023-12-31")
   ```

2. **内存不足**
   ```python
   # 减少回测周期或启用内存优化
   config['memory_optimization'] = True
   config['batch_size'] = 1000
   ```

3. **计算速度慢**
   ```python
   # 启用并行处理
   config['parallel_processing'] = True
   config['num_processes'] = 4
   ```

### 调试模式

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

framework = OptionsBacktestFramework(config)
framework.run_backtest(signal_generators, debug=True)
```

## 📚 文档

- [API参考文档](docs/api_reference.md)
- [策略开发指南](docs/strategy_guide.md)
- [用户手册](docs/user_manual.md)
- [常见问题](docs/faq.md)

## 🤝 贡献

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

### 贡献类型

- 🐛 Bug修复
- ✨ 新功能
- 📝 文档改进
- 🧪 测试用例
- 🎨 代码优化

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目：

- [NumPy](https://numpy.org/) - 数值计算
- [Pandas](https://pandas.pydata.org/) - 数据处理
- [SciPy](https://scipy.org/) - 科学计算
- [Matplotlib](https://matplotlib.org/) - 数据可视化
- [yfinance](https://github.com/ranaroussi/yfinance) - 金融数据

## 📞 联系方式

- 作者: 资深期权量化交易员
- 邮箱: options.trader@example.com
- 讨论群: [Telegram群组](https://t.me/options_quant)

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！

📢 欢迎分享给其他期权交易员和量化研究者！ 
