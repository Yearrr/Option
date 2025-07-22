#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期权回测框架 - 基础回测示例

这个示例展示了如何使用期权回测框架进行基本的策略回测。
包含了波动率策略、动量策略等多种信号的组合使用。
"""

import sys
import os
sys.path.append('..')  # 添加上级目录到路径

from options_backtest_framework import (
    OptionsBacktestFramework,
    VolatilitySignalGenerator,
    MomentumSignalGenerator,
    SkewSignalGenerator
)

def run_basic_backtest():
    """运行基础回测示例"""
    
    print("=" * 60)
    print("期权回测框架 - 基础示例")
    print("=" * 60)
    
    # 1. 配置回测参数
    config = {
        # 基础参数
        'underlying_symbol': 'AAPL',           # 标的代码
        'start_date': '2023-01-01',            # 回测开始日期
        'end_date': '2023-12-31',              # 回测结束日期
        'initial_cash': 100000,                # 初始资金
        'commission_rate': 0.001,              # 佣金费率 0.1%
        
        # 数据源配置
        'data_source': 'yfinance',             # 数据源
        'cache_dir': 'data_cache',             # 数据缓存目录
        
        # 风险管理参数
        'risk_management': {
            'max_loss_pct': 0.5,               # 最大亏损比例 50%
            'take_profit_pct': 1.0,            # 止盈比例 100%
            'max_positions': 10,               # 最大持仓数
            'margin_requirement': 0.2,         # 保证金要求 20%
            'delta_hedge_threshold': 0.1,      # Delta对冲阈值
            'gamma_hedge_threshold': 0.05,     # Gamma对冲阈值
            'vega_hedge_threshold': 0.1        # Vega对冲阈值
        },
        
        # 仓位管理参数
        'position_sizing': {
            'max_position_size': 0.05,         # 单个头寸最大5%
            'kelly_fraction': 0.25             # Kelly公式分数
        },
        
        # 保证金参数
        'margin_parameters': {
            'initial_margin': 0.2,             # 初始保证金
            'maintenance_margin': 0.15         # 维持保证金
        },
        
        # 对冲参数
        'hedge_parameters': {
            'auto_hedge': True,                # 自动对冲
            'hedge_frequency': 'daily',        # 对冲频率
            'hedge_threshold': 0.1             # 对冲阈值
        },
        
        # 其他参数
        'risk_free_rate': 0.05                # 无风险利率 5%
    }
    
    # 2. 创建回测框架
    print("初始化回测框架...")
    framework = OptionsBacktestFramework(config)
    
    # 3. 设置信号生成器
    print("设置信号生成器...")
    
    # 创建不同类型的信号生成器
    signal_generators = [
        VolatilitySignalGenerator(
            lookback_period=21,
            volatility_threshold=0.25
        ),
        MomentumSignalGenerator(
            short_window=10,
            long_window=30
        ),
        SkewSignalGenerator(
            lookback_period=21
        )
    ]
    
    # 设置信号权重
    signal_weights = {
        'VolatilitySignal': 0.4,  # 波动率信号权重40%
        'MomentumSignal': 0.4,    # 动量信号权重40%
        'SkewSignal': 0.2         # 偏度信号权重20%
    }
    
    # 4. 运行回测
    print("开始运行回测...")
    
    try:
        results = framework.run_backtest(
            signal_generators=signal_generators,
            weights=signal_weights,
            expiry_dates=None,  # 使用默认的月度到期日
        )
        
        # 5. 分析结果
        print("\n回测完成！正在分析结果...")
        
        # 打印详细的绩效指标
        performance = results['performance_metrics']
        print("\n" + "=" * 50)
        print("详细绩效分析")
        print("=" * 50)
        
        print(f"📊 总体表现:")
        print(f"   总收益率: {performance.get('total_return', 0):.2%}")
        print(f"   年化收益率: {performance.get('annualized_return', 0):.2%}")
        print(f"   最终价值: ${performance.get('final_value', 0):,.2f}")
        
        print(f"\n📈 风险指标:")
        print(f"   年化波动率: {performance.get('volatility', 0):.2%}")
        print(f"   夏普比率: {performance.get('sharpe_ratio', 0):.3f}")
        print(f"   最大回撤: {performance.get('max_drawdown', 0):.2%}")
        
        print(f"\n🎯 交易统计:")
        print(f"   总交易次数: {performance.get('total_trades', 0)}")
        print(f"   胜率: {performance.get('win_rate', 0):.2%}")
        
        # 按策略分析
        if 'strategy_performance' in results:
            strategy_perf = results['strategy_performance']
            print(f"\n📋 分策略表现:")
            for strategy, metrics in strategy_perf.items():
                print(f"   {strategy}:")
                print(f"     总盈亏: ${metrics.get('total_pnl', 0):.2f}")
                print(f"     交易次数: {metrics.get('trade_count', 0)}")
                print(f"     胜率: {metrics.get('win_rate', 0):.2%}")
                print(f"     平均盈亏: ${metrics.get('avg_pnl_per_trade', 0):.2f}")
        
        # 6. 保存结果
        print("\n保存回测结果...")
        framework.save_results("backtest_results")
        
        # 7. 生成图表
        print("生成分析图表...")
        framework.plot_results(save_plots=True, output_dir="backtest_results")
        
        print(f"\n✅ 回测完成！结果已保存到 'backtest_results' 目录")
        
        # 8. 生成回测报告
        generate_backtest_report(framework, results)
        
    except Exception as e:
        print(f"❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()


def generate_backtest_report(framework, results):
    """生成详细的回测报告"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("期权策略回测报告")
    report_lines.append("=" * 80)
    
    # 基本信息
    config = framework.config
    report_lines.append(f"\n📋 基本信息:")
    report_lines.append(f"标的代码: {config['underlying_symbol']}")
    report_lines.append(f"回测周期: {config['start_date']} 至 {config['end_date']}")
    report_lines.append(f"初始资金: ${config['initial_cash']:,}")
    report_lines.append(f"佣金费率: {config['commission_rate']:.3%}")
    
    # 绩效摘要
    performance = results['performance_metrics']
    report_lines.append(f"\n📊 绩效摘要:")
    report_lines.append(f"总收益率: {performance.get('total_return', 0):.2%}")
    report_lines.append(f"年化收益率: {performance.get('annualized_return', 0):.2%}")
    report_lines.append(f"年化波动率: {performance.get('volatility', 0):.2%}")
    report_lines.append(f"夏普比率: {performance.get('sharpe_ratio', 0):.3f}")
    report_lines.append(f"最大回撤: {performance.get('max_drawdown', 0):.2%}")
    report_lines.append(f"胜率: {performance.get('win_rate', 0):.2%}")
    report_lines.append(f"总交易次数: {performance.get('total_trades', 0)}")
    
    # 风险分析
    report_lines.append(f"\n⚠️ 风险分析:")
    if performance.get('max_drawdown', 0) < -0.2:
        report_lines.append("❌ 最大回撤超过20%，风险较高")
    elif performance.get('max_drawdown', 0) < -0.1:
        report_lines.append("⚠️ 最大回撤在10-20%之间，风险中等")
    else:
        report_lines.append("✅ 最大回撤小于10%，风险可控")
    
    if performance.get('sharpe_ratio', 0) > 1.0:
        report_lines.append("✅ 夏普比率大于1，风险调整后收益良好")
    elif performance.get('sharpe_ratio', 0) > 0.5:
        report_lines.append("⚠️ 夏普比率在0.5-1之间，表现一般")
    else:
        report_lines.append("❌ 夏普比率小于0.5，风险调整后收益较差")
    
    # 策略建议
    report_lines.append(f"\n💡 策略建议:")
    if performance.get('win_rate', 0) < 0.4:
        report_lines.append("- 胜率较低，建议优化信号生成逻辑")
    if performance.get('total_trades', 0) < 50:
        report_lines.append("- 交易次数较少，建议增加信号敏感度")
    if performance.get('total_trades', 0) > 500:
        report_lines.append("- 交易次数过多，建议增加信号过滤条件")
    
    report_lines.append("\n" + "=" * 80)
    
    # 保存报告
    with open("backtest_results/backtest_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print("📝 详细报告已保存到 'backtest_results/backtest_report.txt'")


def run_volatility_only_example():
    """运行纯波动率策略示例"""
    
    print("\n" + "=" * 60)
    print("纯波动率策略回测示例")
    print("=" * 60)
    
    # 简化配置
    config = {
        'underlying_symbol': 'SPY',  # 使用SPY作为标的
        'start_date': '2023-01-01',
        'end_date': '2023-06-30',
        'initial_cash': 50000,
        'risk_management': {
            'max_loss_pct': 0.3,
            'take_profit_pct': 0.8,
            'max_positions': 5,
        }
    }
    
    framework = OptionsBacktestFramework(config)
    
    # 只使用波动率信号
    signal_generators = [VolatilitySignalGenerator(lookback_period=14)]
    
    try:
        results = framework.run_backtest(signal_generators)
        
        # 保存到不同目录
        framework.save_results("volatility_strategy_results")
        framework.plot_results(save_plots=True, output_dir="volatility_strategy_results")
        
        print("✅ 波动率策略回测完成！")
        
    except Exception as e:
        print(f"❌ 波动率策略回测失败: {e}")


if __name__ == "__main__":
    # 运行基础回测示例
    run_basic_backtest()
    
    # 运行纯波动率策略示例
    run_volatility_only_example()
    
    print("\n🎉 所有示例运行完成！")
    print("请查看生成的结果文件和图表。")