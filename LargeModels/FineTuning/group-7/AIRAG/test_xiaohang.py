"""
测试小航API集成
验证小航LLM是否正常工作
"""
import os
from dotenv import load_dotenv
from loguru import logger
from langchain.schema import Document
from src.generator import AnswerGenerator

# 加载环境变量
load_dotenv()

def test_xiaohang_basic():
    """基础测试：验证小航API连接"""
    print("=" * 60)
    print("测试 1: 基础连接测试")
    print("=" * 60)
    
    try:
        # 初始化小航生成器
        generator = AnswerGenerator(
            model_type="api",
            api_key="f93082e1-2cbf-4f81-af8f-9c98d528b6b1",
            base_url="https://xhang.buaa.edu.cn/xhang/v1",
            model_name="xhang"
        )
        
        # 测试文档
        test_docs = [
            Document(
                page_content="人工智能(AI)是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。AI系统可以学习、推理、解决问题、理解自然语言等。",
                metadata={"source": "ai_basics.txt", "page": 1}
            ),
            Document(
                page_content="机器学习是人工智能的一个重要分支，它使计算机系统能够通过经验自动改进。常见的机器学习方法包括监督学习、无监督学习和强化学习。",
                metadata={"source": "ml_intro.txt", "page": 1}
            )
        ]
        
        # 测试问题
        question = "什么是人工智能？"
        
        print(f"\n问题: {question}")
        print(f"检索到 {len(test_docs)} 个相关文档")
        print("\n生成答案中...\n")
        
        # 生成答案
        result = generator.generate_answer(question, test_docs)
        
        print("✅ 测试成功！")
        print("\n" + "=" * 60)
        print("回答:")
        print("=" * 60)
        print(result['answer'])
        print("\n" + "=" * 60)
        print(f"来源数量: {len(result['sources'])}")
        
        for i, source in enumerate(result['sources'], 1):
            print(f"\n[来源 {i}]")
            print(f"  内容: {source['content'][:100]}...")
            print(f"  文件: {source['metadata'].get('source', '未知')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        logger.error(f"小航API测试失败: {e}")
        return False


def test_xiaohang_multiple_questions():
    """测试多个问题"""
    print("\n\n" + "=" * 60)
    print("测试 2: 多问题测试")
    print("=" * 60)
    
    try:
        generator = AnswerGenerator(
            model_type="api",
            api_key="f93082e1-2cbf-4f81-af8f-9c98d528b6b1",
            base_url="https://xhang.buaa.edu.cn/xhang/v1",
            model_name="xhang"
        )
        
        # 测试文档
        test_docs = [
            Document(
                page_content="深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的表示。深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。",
                metadata={"source": "deep_learning.txt"}
            ),
            Document(
                page_content="神经网络由许多相互连接的节点（神经元）组成，模仿人脑的结构。每个连接都有一个权重，通过训练来调整这些权重以改善模型性能。",
                metadata={"source": "neural_networks.txt"}
            )
        ]
        
        questions = [
            "什么是深度学习？",
            "神经网络是如何工作的？",
            "深度学习和机器学习的关系是什么？"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n问题 {i}: {question}")
            result = generator.generate_answer(question, test_docs)
            print(f"答案: {result['answer'][:150]}...")
            print("✅ 成功")
        
        print("\n✅ 多问题测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_from_env():
    """从环境变量测试"""
    print("\n\n" + "=" * 60)
    print("测试 3: 使用环境变量配置")
    print("=" * 60)
    
    try:
        import config
        
        print(f"\n当前配置:")
        print(f"  MODEL_TYPE: {config.MODEL_TYPE}")
        print(f"  LLM_MODEL_NAME: {config.LLM_MODEL_NAME}")
        print(f"  OPENAI_BASE_URL: {config.OPENAI_BASE_URL}")
        print(f"  API_KEY: {'*' * 20 + config.OPENAI_API_KEY[-10:] if config.OPENAI_API_KEY else '未设置'}")
        
        if config.MODEL_TYPE != "api" or config.LLM_MODEL_NAME != "xhang":
            print("\n⚠️  警告: 当前配置未使用小航API")
            print("   请修改 .env 文件或 config.py 以使用小航API")
            return False
        
        # 使用默认配置创建生成器
        generator = AnswerGenerator()
        
        test_docs = [
            Document(
                page_content="自然语言处理(NLP)是人工智能的一个分支，专注于使计算机能够理解、解释和生成人类语言。",
                metadata={"source": "nlp.txt"}
            )
        ]
        
        question = "什么是自然语言处理？"
        print(f"\n问题: {question}")
        
        result = generator.generate_answer(question, test_docs)
        
        print("✅ 环境变量配置测试成功！")
        print(f"\n答案: {result['answer'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("\n🚀 小航API集成测试")
    print("=" * 60)
    print("本测试将验证小航LLM是否正常工作")
    print("=" * 60)
    
    results = []
    
    # 测试1: 基础测试
    results.append(("基础连接测试", test_xiaohang_basic()))
    
    # 测试2: 多问题测试
    results.append(("多问题测试", test_xiaohang_multiple_questions()))
    
    # 测试3: 环境变量测试
    results.append(("环境变量配置测试", test_from_env()))
    
    # 汇总结果
    print("\n\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！小航API集成成功！")
        print("\n下一步:")
        print("  1. 运行 run_enhanced_web.bat 启动Web界面")
        print("  2. 或运行 python run.py --mode cli 启动CLI")
    else:
        print("\n⚠️  部分测试失败，请检查:")
        print("  1. 网络连接是否正常")
        print("  2. API Key 是否正确")
        print("  3. API Base URL 是否可访问")
        print("  4. 查看 data/logs/ 目录下的日志文件")


if __name__ == "__main__":
    main()


