#!/usr/bin/env python3
"""
ICL提示策略对比系统 - 演示启动脚本
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path
import signal
import threading


def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        "streamlit", "pandas", "plotly", "openai", "dspy"
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False

    return True


def setup_environment():
    """设置环境变量"""
    # 检查DeepSeek API密钥（现在已内置在配置中，此检查可选）
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("ℹ️  未设置 DEEPSEEK_API_KEY 环境变量（可选，已在配置中内置）")
        print("如需覆盖默认密钥，可设置环境变量：")
        print("export DEEPSEEK_API_KEY='your-api-key-here'")
        print()

    # 检查OpenAI API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("ℹ️  未设置 OPENAI_API_KEY 环境变量（可选）")
        print("如果需要使用OpenAI模型，请设置环境变量：")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print()

    # 创建必要的目录
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)


def run_tests():
    """运行基本测试"""
    print("🧪 运行基本测试...")

    try:
        # 测试配置加载
        from config import TASKS, PROMPT_STRATEGIES
        print(f"✅ 加载了 {len(TASKS)} 个任务和 {len(PROMPT_STRATEGIES)} 个策略")

        # 测试模型推理（模拟模式）
        from model_inference import ModelInference
        inference = ModelInference()
        print("✅ 模型推理模块初始化成功")

        # 测试评估模块
        from evaluation import Evaluator
        evaluator = Evaluator(inference)
        print("✅ 评估模块初始化成功")

        # 测试DSPy集成 - 使用正确的类名
        try:
            from dspy_integration import DSPyPipelineOptimizer
            optimizer = DSPyPipelineOptimizer({
                "model": "gemma3:1b",
                "api_base": "http://localhost:11434"
            })
            print("✅ DSPy集成模块初始化成功")
        except ImportError as e:
            # 如果DSPy相关模块导入失败，检查具体错误
            print(f"⚠️ DSPy模块导入警告: {e}")
            print("🔶 尝试使用简化模式...")
            # 检查是否有其他DSPy类可用
            try:
                from dspy_integration import DSPyBasicPredictor, DSPyTaskEvaluator
                print("✅ DSPy基础类加载成功")
            except ImportError as e2:
                print(f"❌ DSPy类导入失败: {e2}")
                raise

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dspy_classes():
    """检查DSPy类是否存在"""
    print("🔍 检查DSPy类...")
    try:
        from dspy_integration import (
            DSPyPipelineOptimizer,
            DSPyBasicPredictor,
            DSPyTaskEvaluator,
            DSPyEvaluationMetric,
            DSPySentimentMetric,
            DSPyMathMetric,
            DSPyOllamaClient,
            DSPyCostTracker
        )
        print("✅ 所有DSPy类导入成功")
        return True
    except ImportError as e:
        print(f"❌ DSPy类导入失败: {e}")
        # 列出dspy_integration模块中可用的类
        try:
            import dspy_integration
            available_classes = [cls for cls in dir(dspy_integration) if not cls.startswith('_')]
            print(f"📋 可用的类: {available_classes}")
        except:
            pass
        return False


def start_web_interface():
    """启动Web界面"""
    print("🚀 启动ICL提示策略对比系统...")
    print("📊 访问地址: http://localhost:8501")
    print("⏹️  按 Ctrl+C 停止服务")
    print()

    try:
        # 使用 run_demo.py 同目录下的 app.py 绝对路径，避免找不到文件
        app_path = Path(__file__).parent / "app.py"
        if not app_path.exists():
            raise FileNotFoundError(f"未找到应用文件: {app_path}")

        # 使用Popen启动Streamlit，保持进程运行
        # 添加 --server.headless true 禁用Streamlit自动打开浏览器，只通过我们的函数打开一次
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.serverAddress", "localhost",
            "--server.headless", "true"
        ])

        # 等待进程结束
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n👋 收到停止信号，正在关闭服务...")
            process.terminate()
            process.wait()
            print("✅ 服务已停止")
        except Exception as e:
            print(f"❌ 进程错误: {e}")
            process.terminate()

    except Exception as e:
        print(f"❌ 启动失败: {e}")


def open_browser_after_delay():
    """延迟打开浏览器"""
    time.sleep(5)  # 等待5秒让Streamlit完全启动
    try:
        webbrowser.open("http://localhost:8501")
        print("✅ 浏览器已自动打开")
    except Exception as e:
        print(f"❌ 自动打开浏览器失败: {e}")


def main():
    """主函数"""
    print("=" * 50)
    print("🧠 ICL提示策略对比系统")
    print("=" * 50)

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 设置环境
    setup_environment()

    # 检查DSPy类
    if not check_dspy_classes():
        print("❌ DSPy类检查失败，请检查dspy_integration.py文件")
        sys.exit(1)

    # 运行测试
    if not run_tests():
        print("❌ 系统测试失败，请检查依赖和配置")
        sys.exit(1)

    print("✅ 所有检查通过，准备启动系统...")
    print()

    # 默认自动打开浏览器（无交互）
    print("将在Streamlit启动后自动打开浏览器...")
    browser_thread = threading.Thread(target=open_browser_after_delay, daemon=True)
    browser_thread.start()

    # 启动Web界面
    start_web_interface()


if __name__ == "__main__":
    main()