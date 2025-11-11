"""CLI Demo：命令行问答界面"""
import argparse
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
import config
from src.document_loader import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.retriever import RetrieverManager
from src.generator import AnswerGenerator, RAGPipeline

# 配置日志
logger.add(
    config.LOGS_DIR / "cli_demo.log",
    rotation="500 MB",
    level=config.LOG_LEVEL
)

console = Console()


class RAGCLIDemo:
    """RAG CLI演示应用"""
    
    def __init__(self):
        self.rag_pipeline = None
        self.retriever_manager = None
    
    def setup(
        self,
        documents_path: str,
        force_reindex: bool = False
    ):
        """设置RAG系统"""
        console.print("\n[bold blue]🚀 初始化RAG系统...[/bold blue]\n")
        
        try:
            # 检查是否已有索引
            index_path = Path(config.VECTOR_DB_PATH) / "cli_demo"
            
            if index_path.exists() and not force_reindex:
                console.print("[yellow]发现已有索引，正在加载...[/yellow]")
                
                # 加载已有索引
                vs_manager = VectorStoreManager()
                vs_manager.load("cli_demo")
                
                # 需要重新加载文档用于BM25
                processor = DocumentProcessor()
                chunks = processor.process_documents(documents_path)
            
            else:
                console.print("[yellow]正在加载和索引文档...[/yellow]")
                
                # 加载和处理文档
                processor = DocumentProcessor()
                chunks = processor.process_documents(documents_path)
                
                if not chunks:
                    console.print("[red]❌ 未找到任何文档[/red]")
                    return False
                
                console.print(f"[green]✅ 加载了 {len(chunks)} 个文档块[/green]")
                
                # 创建向量索引
                console.print("[yellow]正在创建向量索引...[/yellow]")
                vs_manager = VectorStoreManager()
                vs_manager.create_vectorstore(chunks)
                vs_manager.save("cli_demo")
                
                console.print("[green]✅ 向量索引创建完成[/green]")
            
            # 初始化检索器
            console.print("[yellow]正在初始化检索器...[/yellow]")
            self.retriever_manager = RetrieverManager(vs_manager)
            self.retriever_manager.setup_bm25(chunks)
            
            # 初始化生成器
            console.print("[yellow]正在初始化生成器...[/yellow]")
            generator = AnswerGenerator()
            
            # 创建RAG流水线
            self.rag_pipeline = RAGPipeline(self.retriever_manager, generator)
            
            console.print("\n[bold green]✅ RAG系统初始化完成！[/bold green]\n")
            return True
        
        except Exception as e:
            console.print(f"[bold red]❌ 初始化失败: {e}[/bold red]")
            logger.error(f"系统初始化失败: {e}")
            return False
    
    def query(
        self,
        question: str,
        retriever_type: str = "faiss",
        top_k: int = None
    ):
        """执行查询"""
        if not self.rag_pipeline:
            console.print("[red]❌ 请先初始化系统[/red]")
            return
        
        top_k = top_k or config.TOP_K
        
        try:
            console.print(f"\n[bold cyan]🔍 查询中...[/bold cyan]")
            
            # 执行查询
            result = self.rag_pipeline.query(
                question=question,
                retriever_type=retriever_type,
                k=top_k
            )
            
            # 显示答案
            console.print("\n")
            console.print(Panel(
                f"[bold]问题:[/bold] {result['question']}\n\n"
                f"[bold]答案:[/bold]\n{result['answer']}",
                title="💬 回答",
                border_style="green"
            ))
            
            # 显示来源
            if result.get('sources'):
                console.print("\n[bold blue]📚 参考来源:[/bold blue]\n")
                
                for source in result['sources']:
                    idx = source['index']
                    content = source['content'][:200]
                    metadata = source.get('metadata', {})
                    score = source.get('score')
                    
                    table = Table(show_header=False, box=None)
                    table.add_column("Field", style="cyan")
                    table.add_column("Value")
                    
                    table.add_row("来源", f"[来源 {idx}]")
                    table.add_row("内容", content + "...")
                    table.add_row("文件", metadata.get('source', '未知'))
                    
                    if score is not None:
                        table.add_row("相关度", f"{score:.4f}")
                    
                    console.print(table)
                    console.print()
        
        except Exception as e:
            console.print(f"[bold red]❌ 查询失败: {e}[/bold red]")
            logger.error(f"查询失败: {e}")
    
    def compare(self, question: str, top_k: int = None):
        """对比检索器"""
        if not self.retriever_manager:
            console.print("[red]❌ 请先初始化系统[/red]")
            return
        
        top_k = top_k or config.TOP_K
        
        try:
            console.print(f"\n[bold cyan]🔍 正在对比检索器...[/bold cyan]\n")
            
            comparison = self.retriever_manager.compare_retrievers(question, k=top_k)
            
            for retriever_name in ['faiss', 'bm25', 'hybrid']:
                console.print(f"\n[bold yellow]{retriever_name.upper()} 检索器:[/bold yellow]")
                
                results = comparison[retriever_name]
                for i, doc_info in enumerate(results[:3], 1):
                    content = doc_info['content'][:150]
                    score = doc_info.get('score', 'N/A')
                    
                    console.print(f"  {i}. {content}...")
                    if score != 'N/A':
                        console.print(f"     [dim]相关度: {score:.4f}[/dim]")
                
                console.print()
        
        except Exception as e:
            console.print(f"[bold red]❌ 对比失败: {e}[/bold red]")
            logger.error(f"对比失败: {e}")
    
    def interactive_mode(self):
        """交互模式"""
        console.print(Panel(
            "[bold]RAG问答系统 - 交互模式[/bold]\n\n"
            "命令:\n"
            "  输入问题 - 直接提问\n"
            "  /compare <问题> - 对比检索器\n"
            "  /retriever <faiss|bm25|hybrid> - 切换检索器\n"
            "  /topk <数字> - 设置Top-K\n"
            "  /help - 显示帮助\n"
            "  /quit 或 /exit - 退出",
            border_style="blue"
        ))
        
        retriever_type = "faiss"
        top_k = config.TOP_K
        
        while True:
            try:
                user_input = console.input("\n[bold cyan]>>> [/bold cyan]").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.startswith('/'):
                    parts = user_input.split(maxsplit=1)
                    command = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    
                    if command in ['/quit', '/exit']:
                        console.print("[yellow]再见！[/yellow]")
                        break
                    
                    elif command == '/help':
                        console.print("""
[bold]可用命令:[/bold]
  输入问题 - 直接提问
  /compare <问题> - 对比检索器
  /retriever <faiss|bm25|hybrid> - 切换检索器
  /topk <数字> - 设置Top-K
  /help - 显示帮助
  /quit 或 /exit - 退出
                        """)
                    
                    elif command == '/compare':
                        if args:
                            self.compare(args, top_k)
                        else:
                            console.print("[red]请提供问题[/red]")
                    
                    elif command == '/retriever':
                        if args.lower() in ['faiss', 'bm25', 'hybrid']:
                            retriever_type = args.lower()
                            console.print(f"[green]✅ 切换到 {retriever_type} 检索器[/green]")
                        else:
                            console.print("[red]无效的检索器类型[/red]")
                    
                    elif command == '/topk':
                        try:
                            top_k = int(args)
                            console.print(f"[green]✅ Top-K 设置为 {top_k}[/green]")
                        except:
                            console.print("[red]无效的数字[/red]")
                    
                    else:
                        console.print("[red]未知命令，输入 /help 查看帮助[/red]")
                
                else:
                    # 执行查询
                    self.query(user_input, retriever_type, top_k)
            
            except KeyboardInterrupt:
                console.print("\n[yellow]再见！[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]错误: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="RAG问答系统 CLI Demo")
    parser.add_argument(
        '--documents',
        type=str,
        default=config.DOCUMENTS_PATH,
        help='文档目录路径'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='单次查询的问题'
    )
    parser.add_argument(
        '--retriever',
        type=str,
        default='faiss',
        choices=['faiss', 'bm25', 'hybrid'],
        help='检索器类型'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=config.TOP_K,
        help='Top-K值'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='对比检索器'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='交互模式'
    )
    parser.add_argument(
        '--force-reindex',
        action='store_true',
        help='强制重新索引'
    )
    
    args = parser.parse_args()
    
    # 创建CLI Demo
    cli_demo = RAGCLIDemo()
    
    # 初始化系统
    if not cli_demo.setup(args.documents, args.force_reindex):
        return
    
    # 交互模式
    if args.interactive or (not args.question and not args.compare):
        cli_demo.interactive_mode()
    
    # 单次查询
    elif args.question:
        if args.compare:
            cli_demo.compare(args.question, args.topk)
        else:
            cli_demo.query(args.question, args.retriever, args.topk)


if __name__ == "__main__":
    main()

