# 贡献指南 (Contributing Guide)

感谢您对期权回测框架的兴趣！我们欢迎各种形式的贡献。

## 🤝 如何贡献

### 报告Bug
如果您发现了bug，请：
1. 在GitHub Issues中搜索是否已有相关问题
2. 如果没有，请创建新的issue，包含：
   - 清晰的bug描述
   - 重现步骤
   - 期望行为
   - 实际行为
   - 环境信息（Python版本、操作系统等）

### 建议新功能
1. 在Issues中描述您的想法
2. 说明为什么这个功能有用
3. 提供实现的初步想法

### 提交代码
1. Fork这个仓库
2. 创建feature分支：`git checkout -b feature/amazing-feature`
3. 提交您的更改：`git commit -m 'Add amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 创建Pull Request

## 📝 代码规范

### Python代码风格
- 遵循PEP 8标准
- 使用type hints
- 添加docstrings (Google风格)
- 变量和函数名使用英文

### 示例代码格式
```python
def calculate_option_price(
    underlying_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float
) -> float:
    """
    Calculate option price using Black-Scholes formula.
    
    Args:
        underlying_price: Current price of underlying asset
        strike_price: Option strike price
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate
        volatility: Implied volatility
        
    Returns:
        Option price
        
    Raises:
        ValueError: If any parameter is negative
    """
    # Implementation here
    pass
```

## 🧪 测试

### 运行测试
```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行所有测试
pytest

# 运行带覆盖率的测试
pytest --cov=options_backtest_framework --cov-report=html

# 运行特定测试
pytest tests/test_option_pricing.py -v
```

### 编写测试
- 为新功能添加单元测试
- 测试文件命名：`test_*.py`
- 测试函数命名：`test_*`
- 使用pytest fixtures
- 保持测试独立性

## 📚 文档

### 更新文档
- 为新功能添加docstrings
- 更新README.md
- 添加使用示例
- 更新API文档

### 文档构建
```bash
# 安装文档依赖
pip install -e ".[docs]"

# 构建文档
cd docs
make html
```

## 🏗️ 开发环境设置

### 1. 克隆仓库
```bash
git clone https://github.com/your-username/options-backtest-framework.git
cd options-backtest-framework
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -e ".[dev]"
```

### 4. 安装pre-commit hooks
```bash
pip install pre-commit
pre-commit install
```

## 🔄 Pull Request流程

### PR标题格式
- `feat: 添加新功能描述`
- `fix: 修复bug描述`
- `docs: 更新文档`
- `test: 添加测试`
- `refactor: 重构代码`

### PR检查清单
- [ ] 代码通过所有测试
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 遵循代码规范
- [ ] 没有引入breaking changes（或已说明）

### PR描述模板
```markdown
## 描述
简要描述这个PR的内容

## 变更类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 文档更新
- [ ] 重构
- [ ] 其他：

## 测试
描述如何测试这些变更

## 相关Issue
关联的issue编号：#123

## 截图（如适用）
添加截图帮助解释变更
```

## 📋 发布流程

### 版本号规则
遵循语义化版本 (Semantic Versioning)：
- `MAJOR.MINOR.PATCH`
- MAJOR：不兼容的API变更
- MINOR：向后兼容的功能新增
- PATCH：向后兼容的bug修复

### 发布检查清单
- [ ] 所有测试通过
- [ ] 文档更新
- [ ] 版本号更新
- [ ] CHANGELOG更新
- [ ] 创建Git tag

## 🎯 贡献重点领域

我们特别欢迎以下方面的贡献：

### 高优先级
- 新的期权策略实现
- 性能优化
- 错误处理改进
- 测试覆盖率提升

### 中优先级
- 数据源扩展
- 新的技术指标
- 可视化改进
- 文档完善

### 长期目标
- 实时交易接口
- 机器学习集成
- Web界面
- 移动端支持

## 💬 社区

### 获取帮助
- GitHub Issues：技术问题和bug报告
- Discussions：一般讨论和问答
- Email：options.trader@example.com

### 行为准则
我们致力于创建一个友好、包容的社区环境：
- 尊重不同观点
- 提供建设性反馈
- 帮助新贡献者
- 保持专业态度

## 🙏 致谢

感谢所有贡献者的努力！您的贡献使这个项目变得更好。

## 📄 许可证

通过贡献代码，您同意您的贡献将在MIT许可证下发布。