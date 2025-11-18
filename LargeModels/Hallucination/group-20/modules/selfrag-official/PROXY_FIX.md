# 代理问题修复说明 / Proxy Issue Fix

## 问题 / Issue

运行评估时出现错误：

```
ImportError: Using SOCKS proxy, but the 'socksio' package is not installed.
```

或者：

```
AttributeError: 'InstructorLLM' object has no attribute 'agenerate_prompt'
```

## 原因 / Root Cause

1. **代理冲突**: 你的环境中设置了 SOCKS5 代理 (`socks5h://127.0.0.1:7890`)，这与 langchain/httpx 的兼容性有问题。

2. **LLM 类型问题**: 之前使用的 `llm_factory` 创建的 LLM 对象不完全兼容 RAGAS 的评估需求。

## 已应用的修复 / Fixes Applied

### 修复 1: 使用 langchain ChatOpenAI

**Before:**
```python
from ragas.llms import llm_factory
llm = llm_factory("deepseek-reasoner", client=client)
```

**After:**
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0,
    max_tokens=512,
)
```

**好处**:
- ✅ 完全兼容 RAGAS 评估
- ✅ 支持所有必需的方法 (`agenerate`, `generate` 等)
- ✅ 更稳定的API调用

### 修复 2: 临时禁用代理

在创建 LLM 时，临时禁用代理设置：

```python
# Save original proxy settings
original_proxies = {}
proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
              'http_proxy', 'https_proxy', 'all_proxy']
for var in proxy_vars:
    if var in os.environ:
        original_proxies[var] = os.environ[var]
        del os.environ[var]

try:
    # Create LLM without proxy
    llm = ChatOpenAI(...)
finally:
    # Restore proxy settings
    for var, value in original_proxies.items():
        os.environ[var] = value
```

这样可以：
- ✅ 避免 SOCKS proxy 错误
- ✅ 直连 DeepSeek API
- ✅ 不影响其他程序的代理设置

### 修复 3: 使用 deepseek-chat 模型

**Before:** `deepseek-reasoner` (推理模型)
**After:** `deepseek-chat` (对话模型)

**原因**:
- `deepseek-chat` 更适合评估任务
- 响应更快，成本更低
- API 更稳定

## 已修复的文件 / Fixed Files

- ✅ `evaluate_with_ragas.py`
- ✅ `evaluate_with_ragas_simple.py`

## 现在可以运行 / Now You Can Run

```bash
export DEEPSEEK_API_KEY='your-key'
python3 evaluate_with_ragas.py
```

## 验证修复 / Verify Fix

运行测试：

```bash
python3 << 'EOF'
import os
os.environ['DEEPSEEK_API_KEY'] = 'sk-b44e8978b5b046cfa0f64d96d53cb062'

from evaluate_with_ragas import create_deepseek_llm

api_key = os.environ['DEEPSEEK_API_KEY']
llm = create_deepseek_llm(api_key)
print(f"✓ LLM created: {type(llm)}")
print("✅ Ready to evaluate!")
EOF
```

应该看到：
```
✓ LLM created: <class 'langchain_openai.chat_models.base.ChatOpenAI'>
✅ Ready to evaluate!
```

## 关于代理 / About Proxy

你的环境中有以下代理设置：

```bash
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
all_proxy=socks5h://127.0.0.1:7890
```

评估脚本会：
1. 在创建 LLM 时**临时禁用**代理
2. 直连 DeepSeek API (`https://api.deepseek.com`)
3. 评估完成后**恢复**代理设置

这不会影响：
- ✅ 其他程序的网络连接
- ✅ 浏览器的代理设置
- ✅ 系统的代理配置

## 如果还有问题 / If Still Have Issues

### 选项 1: 临时禁用代理运行

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
python3 evaluate_with_ragas.py
```

### 选项 2: 安装 socksio 支持

如果你需要通过代理访问 DeepSeek API：

```bash
pip install 'httpx[socks]'
```

然后修改脚本，不要禁用代理。

### 选项 3: 检查网络连接

确保可以直接访问 DeepSeek API：

```bash
curl -v https://api.deepseek.com
```

## 技术细节 / Technical Details

### 为什么使用 ChatOpenAI?

1. **完全兼容**: langchain 的 `ChatOpenAI` 是 RAGAS 原生支持的 LLM 类型
2. **标准接口**: 提供所有 RAGAS 需要的方法
3. **灵活配置**: 可以使用自定义 `base_url` 指向任何 OpenAI 兼容的 API

### DeepSeek API 兼容性

DeepSeek 的 API 是 OpenAI 兼容的，所以可以使用 `ChatOpenAI` 客户端：

```python
ChatOpenAI(
    base_url="https://api.deepseek.com",  # DeepSeek endpoint
    api_key="sk-xxx",                      # DeepSeek API key
    model="deepseek-chat"                  # DeepSeek model
)
```

### RAGAS 与 Langchain

RAGAS 内部使用 langchain 的 LLM 抽象:
- 接受 langchain LLM 对象
- 自动调用 `generate()` 和 `agenerate()` 方法
- 支持同步和异步评估

---

**修复时间 Fix Applied**: 2025-10-29
**状态 Status**: ✅ 已解决 Resolved
**测试状态 Test Status**: ✅ 通过 Passed
