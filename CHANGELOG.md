# 更新日志 (Changelog)

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-07-22

### Added
- 🎉 初始版本发布
- 📊 完整的期权回测框架
- 💰 数据管理模块 (DataManager)
  - 支持yfinance和tushare数据源
  - 智能数据缓存机制
  - 期权链数据获取
  - 无风险利率获取
- 🧮 期权定价模块 (OptionPricer)
  - Black-Scholes期权定价公式
  - 完整的Greeks计算 (Delta, Gamma, Theta, Vega, Rho)
  - 隐含波动率计算
  - 美式期权二项树定价
  - 期权策略组合定价
- 📈 信号生成模块 (SignalGenerator)
  - 波动率信号生成器
  - 动量信号生成器
  - 偏度信号生成器
  - VIX信号生成器
  - 财报事件信号生成器
  - 组合信号生成器
- ⚠️ 风险管理模块 (RiskManager)
  - 智能止盈止损机制
  - 基于Greeks的动态对冲
  - 保证金计算和管理
  - 仓位限制和集中度控制
  - Kelly公式仓位管理
- 🔄 交易引擎模块 (TradingEngine)
  - 订单管理系统
  - 持仓跟踪和管理
  - 佣金计算
  - 期权到期处理
  - 绩效统计分析
- 🚀 主回测框架 (OptionsBacktestFramework)
  - 完整的回测流程
  - 期权链生成 (模拟和真实数据)
  - 多策略执行引擎
  - 结果分析和可视化
  - 报告生成

### 期权策略支持
- ✅ Long/Short Straddle (跨式组合)
- ✅ Long/Short Strangle (宽跨式组合)
- ✅ Bull/Bear Call Spread (看涨价差)
- ✅ Bull/Bear Put Spread (看跌价差)
- ✅ Iron Condor (铁鹰策略)
- ✅ Ratio Spreads (比例价差)
- ✅ 自定义策略扩展框架

### 绩效分析功能
- 📊 详细的绩效指标计算
  - 总收益率、年化收益率
  - 年化波动率、最大回撤
  - 夏普比率、索提诺比率
  - 胜率、平均持仓时间
- 📈 可视化图表
  - 投资组合价值曲线
  - 收益率分布
  - 回撤曲线
  - Greeks敞口变化
- 📁 多格式结果导出
  - CSV格式交易记录
  - JSON格式绩效指标
  - PNG格式图表
  - 详细的回测报告

### 技术特性
- 🔧 模块化设计，易于扩展
- 🚀 高性能计算优化
- 💾 智能数据缓存
- 🧪 完整的单元测试
- 📚 详细的API文档
- 🎨 用户友好的接口

### 示例和文档
- 📖 完整的README文档
- 💡 基础使用示例
- 🔧 高级配置指南
- 📋 API参考文档
- 🤝 贡献指南

## [Unreleased]

### Planned Features
- 🔮 机器学习信号生成
- 🌐 实时数据接口
- 📱 Web界面
- 🔄 并行计算优化
- 📊 更多可视化图表
- 🧮 更多期权定价模型
- 📈 更多技术指标
- 🔒 风险控制增强

---

## 版本说明

### 版本号格式
我们使用[语义化版本](https://semver.org/)格式：`MAJOR.MINOR.PATCH`

- **MAJOR**: 不兼容的API变更
- **MINOR**: 向后兼容的新功能
- **PATCH**: 向后兼容的错误修复

### 变更类型
- `Added`: 新增功能
- `Changed`: 已有功能的变更
- `Deprecated`: 即将移除的功能
- `Removed`: 已移除的功能
- `Fixed`: 错误修复
- `Security`: 安全相关修复

### 发布计划
- **稳定版本**: 每个季度发布一个主要版本
- **补丁版本**: 根据需要发布错误修复
- **预览版本**: 提供新功能的早期访问

## 贡献

如果您发现任何问题或有改进建议，请：
1. 查看[Issues](https://github.com/your-username/options-backtest-framework/issues)
2. 创建新的Issue或Pull Request
3. 参考[贡献指南](CONTRIBUTING.md)

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。