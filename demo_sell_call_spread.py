#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卖出看涨价差策略演示脚本
主要目标：获得时间价值衰减和隐含波动率下降的收益
"""

from option_spread_strategy import run_sell_call_spread_demo

if __name__ == "__main__":
    print("=" * 80)
    print("卖出看涨价差（Sell Call Spread）策略回测")
    print("=" * 80)
    print()
    print("策略描述:")
    print("• 卖出一份虚值看涨期权（收取权利金）")
    print("• 买入一份更高行权价的虚值看涨期权（支付权利金）")
    print("• 净收益 = 卖出期权权利金 - 买入期权权利金")
    print()
    print("收益来源:")
    print("• 时间价值衰减（Theta收益）")
    print("• 隐含波动率下降（Vega收益）")
    print("• 标的价格保持在卖出期权行权价以下")
    print()
    print("风险特征:")
    print("• 最大收益：收到的净权利金")
    print("• 最大损失：两个行权价的差额 - 净权利金")
    print("• 盈亏平衡点：卖出期权行权价 + 净权利金")
    print()
    print("开始回测...")
    print("=" * 80)
    
    # 运行回测
    engine, summary = run_sell_call_spread_demo()
    
    print("\n" + "=" * 80)
    print("回测总结:")
    print("=" * 80)
    print("✅ 该策略适合以下市场环境:")
    print("  • 横盘整理或小幅上涨的市场")
    print("  • 隐含波动率较高的时期（有利于期权卖方）")
    print("  • 预期波动率会下降的时期")
    print()
    print("⚠️  风险提醒:")
    print("  • 如果标的价格大幅上涨突破卖出期权行权价，将面临损失")
    print("  • 隐含波动率上升会不利于期权卖方")
    print("  • 需要关注希腊字母（特别是Delta和Vega）的变化")
    print()
    print("📊 优化建议:")
    print("  • 在VIX较高时开仓（IV rank > 50%）")
    print("  • 选择30-45天到期的期权以获得最佳时间价值衰减")
    print("  • 设置50%最大利润的止盈目标")
    print("  • 考虑在期权到期前5-10天主动平仓以避免Gamma风险")
    print("=" * 80)