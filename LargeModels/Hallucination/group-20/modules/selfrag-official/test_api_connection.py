#!/usr/bin/env python3
"""测试 DeepSeek API 连接"""

import os
import sys

# 检查 API key
api_key = os.environ.get('DEEPSEEK_API_KEY')
if not api_key:
    print("❌ Error: DEEPSEEK_API_KEY environment variable not set")
    print("\nPlease set it:")
    print("  export DEEPSEEK_API_KEY='your-api-key'")
    sys.exit(1)

print(f"✓ DEEPSEEK_API_KEY is set: {api_key[:10]}...{api_key[-4:]}")

# 测试导入
try:
    from openai import OpenAI
    print("✓ openai package imported")
except ImportError as e:
    print(f"❌ Failed to import openai: {e}")
    sys.exit(1)

# 测试客户端创建
try:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    print("✓ OpenAI client created with DeepSeek base_url")
except Exception as e:
    print(f"❌ Failed to create client: {e}")
    sys.exit(1)

# 测试简单 API 调用
print("\nTesting API connection with a simple request...")
try:
    response = client.chat.completions.create(
        model="deepseek-chat",  # 使用更便宜的 chat 模型测试
        messages=[
            {"role": "user", "content": "Say 'Hello' in one word"}
        ],
        max_tokens=10,
        temperature=0
    )

    result = response.choices[0].message.content
    print(f"✓ API call successful!")
    print(f"  Response: {result}")
    print("\n✅ All checks passed! Your API key works correctly.")
    print("\nYou can now run:")
    print("  python3 evaluate_with_ragas.py")

except Exception as e:
    print(f"❌ API call failed: {e}")
    print("\nPossible reasons:")
    print("  1. Invalid API key")
    print("  2. Network connection issue")
    print("  3. API quota exceeded")
    print("\nPlease check your API key at: https://platform.deepseek.com/")
    sys.exit(1)
