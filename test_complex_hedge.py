#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试复杂的反向交易情况，包括完全平仓后反向开仓
"""

import sys
sys.path.append('.')
from delta_hedge_verification import DetailedPortfolioTracker

def test_complex_reverse_trading():
    """测试复杂的反向交易逻辑"""
    
    print("=== 复杂反向交易逻辑测试 ===\n")
    
    portfolio = DetailedPortfolioTracker(initial_cash=100000)
    
    print("场景：测试完全平仓后反向开仓的情况")
    print()
    
    # 1. 建立初始空头仓位
    portfolio.execute_trade('TEST_FUTURE', 'sell', 10, 100.0, 'future', '建立初始空头仓位')
    
    # 2. 加仓
    portfolio.execute_trade('TEST_FUTURE', 'sell', 5, 105.0, 'future', '追加空头仓位')
    
    # 3. 部分平仓
    portfolio.execute_trade('TEST_FUTURE', 'buy', 8, 95.0, 'future', '部分平仓获利')
    
    # 4. 完全平仓并反向开仓 - 这里测试关键逻辑
    portfolio.execute_trade('TEST_FUTURE', 'buy', 12, 110.0, 'future', '完全平仓并反向开多头')
    
    # 5. 继续加仓
    portfolio.execute_trade('TEST_FUTURE', 'buy', 3, 115.0, 'future', '继续加多头仓位')
    
    # 6. 部分平仓
    portfolio.execute_trade('TEST_FUTURE', 'sell', 2, 120.0, 'future', '部分平多头仓位')
    
    print("\n=== 最终分析 ===")
    breakdown = portfolio.get_detailed_breakdown()
    
    print(f"已实现盈亏: ${breakdown['total_realized_pnl']:+,.2f}")
    print(f"未实现盈亏: ${breakdown['total_unrealized_pnl']:+,.2f}")
    print(f"总收益: ${breakdown['total_return']:+,.2f}")
    
    print(f"\n持仓明细:")
    for pos in breakdown['positions']:
        print(f"  {pos['symbol']}: {pos['quantity']}手 @ ${pos['entry_price']:.2f}")
        print(f"    当前价: ${pos['current_price']:.2f}")
        print(f"    未实现盈亏: ${pos['unrealized_pnl']:+,.2f}")
    
    print(f"\n交易记录验证:")
    for i, trade in enumerate(portfolio.trades, 1):
        print(f"  {i}. {trade['action'].upper()} {trade['quantity']} @ ${trade['price']:.2f}")
        print(f"     {trade['description']}")
    
    print(f"\n验证总收益计算:")
    print(f"已实现盈亏 + 未实现盈亏 = ${breakdown['total_realized_pnl']:+.2f} + ${breakdown['total_unrealized_pnl']:+.2f} = ${breakdown['total_return']:+.2f}")
    
    return portfolio

if __name__ == "__main__":
    test_complex_reverse_trading()