from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from .config import load_settings
from .pipeline import RAGPipeline

app = typer.Typer(help="RAG QA Pipeline CLI")
console = Console()


@app.command()
def ingest() -> None:
    """从文档构建向量索引。"""

    settings = load_settings()
    pipeline = RAGPipeline(settings)
    pipeline.ingest()
    console.print("[green]索引构建完成[/green]")


@app.command()
def ask(
    query: str = typer.Argument(..., help="用户问题"),
    top_k: Optional[int] = typer.Option(None, help="检索深度"),
) -> None:
    """问答入口。"""

    settings = load_settings()
    pipeline = RAGPipeline(settings)
    answer = pipeline.answer(query, top_k=top_k)

    console.rule("[bold cyan]回答[/bold cyan]")
    console.print(answer.answer)

    table = Table(title="引用片段", show_header=True, header_style="bold magenta")
    table.add_column("标签")
    table.add_column("来源")
    table.add_column("摘要")

    for citation in answer.citations:
        table.add_row(
            citation["label"],
            citation.get("source", ""),
            citation.get("excerpt", ""),
        )
    console.print(table)
    console.print(f"[yellow]总延迟: {answer.latency_ms:.2f} ms[/yellow]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="监听地址"),
    port: int = typer.Option(8000, "--port", help="端口"),
) -> None:
    """启动 FastAPI 服务。"""

    uvicorn.run("app.server:app", host=host, port=port, reload=False)


@app.command()
def evaluate(
    dataset_path: Path = typer.Argument(..., exists=True, readable=True),
    limit: Optional[int] = typer.Option(None, help="样本数量上限"),
    output_path: Optional[Path] = typer.Option(
        None, help="评估结果输出文件（JSON 格式）"
    ),
) -> None:
    """离线评测与 RAGAS 评估。"""

    from .evaluation import evaluate_dataset

    settings = load_settings()
    pipeline = RAGPipeline(settings)
    report = asyncio.run(
        evaluate_dataset(pipeline=pipeline, dataset_path=dataset_path, limit=limit)
    )

    console.print_json(data=report)

    if output_path:
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        console.print(f"[green]评估结果已保存至 {output_path}[/green]")


def main():
    """Typer CLI 入口。"""
    app()


if __name__ == "__main__":
    main()
