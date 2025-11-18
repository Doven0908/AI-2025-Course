#!/bin/bash
# 快速测试脚本

echo "=========================================="
echo "Quick Test Script"
echo "=========================================="
echo ""

# Check if DEEPSEEK_API_KEY is set
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "❌ DEEPSEEK_API_KEY is not set"
    echo ""
    echo "Please run:"
    echo "  export DEEPSEEK_API_KEY='your-api-key'"
    echo ""
    echo "Then run this script again:"
    echo "  bash quick_test.sh"
    exit 1
fi

echo "✓ DEEPSEEK_API_KEY is set"
echo ""

# Test 1: API connection
echo "Test 1: Testing API connection..."
python3 test_api_connection.py
if [ $? -ne 0 ]; then
    echo "❌ API connection test failed"
    exit 1
fi
echo ""

# Test 2: Run evaluation on 1 sample
echo "=========================================="
echo "Test 2: Running mini evaluation (1 sample)..."
echo "=========================================="
echo ""

# Create a temporary test script
cat > /tmp/test_eval.py << 'EOF'
import os
import sys
sys.path.insert(0, '/data/self-rag')

# Import the evaluation function
from evaluate_with_ragas import create_deepseek_llm

try:
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    print(f"Creating LLM with API key: {api_key[:10]}...{api_key[-4:]}")

    llm = create_deepseek_llm(api_key)
    print("✓ LLM created successfully!")
    print(f"  Type: {type(llm)}")
    print(f"  Model: deepseek-reasoner")

    print("\n✅ All tests passed!")
    print("\nYou can now run the full evaluation:")
    print("  python3 evaluate_with_ragas.py")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

python3 /tmp/test_eval.py
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ All tests passed!"
    echo "=========================================="
    echo ""
    echo "You can now run:"
    echo "  python3 evaluate_with_ragas.py         # 20 samples per experiment"
    echo "  python3 evaluate_with_ragas_simple.py  # 10 samples per experiment"
else
    echo ""
    echo "❌ Tests failed"
    exit 1
fi

# Clean up
rm -f /tmp/test_eval.py
