import glob
import os
from pydantic import BaseModel, Field
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import argparse
import sys

from llm_hallucination_evaluate.evaluator.eval_result import EvalResult
from llm_hallucination_evaluate.utils.format import safe_name

class AnalyzeItem(BaseModel):
    metrics: dict[str, float | None] = Field(..., description="评估指标字典")
    dataset: str = Field(..., description="数据集名称")
    evaluator: str = Field(..., description="评估器名称")
    solve_time: float = Field(..., description="求解时间（秒）")
    method_with_model: str = Field(..., description="求解方法与模型的组合")

    @classmethod
    def from_eval_result(cls, eval_result: EvalResult) -> "AnalyzeItem":
        """
        从EvalResult对象创建AnalyzeItem实例
        Args:
            eval_result: 评估结果对象

        Returns:
            AnalyzeItem: 创建的分析项对象
        """
        method_with_model = f"{eval_result.solve_result.metadata['solve_method']}-with-{eval_result.solve_result.metadata['model_name']}"
        
        return cls(
            metrics=eval_result.results,
            dataset=eval_result.solve_result.metadata["dataset"],
            evaluator=eval_result.evaluator_name,
            solve_time=eval_result.solve_result.metadata["solve_time"],
            method_with_model=method_with_model
        )

def get_user_choice(options: List[str], prompt: str = "请选择: ") -> str:
    """
    让用户从选项列表中选择，输入序号即可
    
    Args:
        options: 选项列表
        prompt: 提示信息
        
        
    Returns:
        用户选择的选项
    """
    print("\n" + "="*50)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print("="*50)
    
    while True:
        try:
            choice = input(prompt).strip()
            if not choice:
                continue
                
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                return options[choice_num - 1]
            else:
                print(f"请输入 1-{len(options)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n用户中断")
            sys.exit(0)

class EvalResultAnalyzer(BaseModel):
    items: list[AnalyzeItem] = Field(default_factory=list)
    _df: Optional[pd.DataFrame] = None
    plot_as_image: bool = Field(default=False, description="是否将图表保存为图片")
    statistical_charts_dir: str = Field(default="./statistical_charts", description="统计图表保存目录")

    @classmethod
    def load_results(cls, results_dir: str, plot_as_image: bool = False, statistical_charts_dir: str = "./statistical_charts") -> "EvalResultAnalyzer":
        """
        从指定目录加载所有评估结果文件
        
        Args:
            results_dir: 评估结果文件所在目录
            plot_as_image: 是否将图表保存为图片
            statistical_charts_dir: 统计图表保存目录
            
        Returns:
            EvalResultAnalyzer: 包含所有评估结果的分析器对象
        """
        result_files = glob.glob(f"{results_dir}/*/*/*.yaml")
        items = []
        for file_path in tqdm(result_files, desc="加载评估结果文件..."):
            try:
                result = EvalResult.read_from_yaml(file_path)
                items.append(AnalyzeItem.from_eval_result(result))
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")
        return cls(items=items, plot_as_image=plot_as_image, statistical_charts_dir=statistical_charts_dir)

    def _save_or_show_plot(self, plt, chart_type: str, filename: str, subdir: str = ""):
        """
        根据plot_as_image设置保存或显示图表
        
        Args:
            plt: matplotlib的pyplot对象
            chart_type: 图表类型
            filename: 文件名
            subdir: 子目录
        """
        if self.plot_as_image:
            # 创建目录
            save_dir = os.path.join(self.statistical_charts_dir, chart_type, subdir)
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存图片
            filepath = os.path.join(save_dir, f"{safe_name(filename)}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {filepath}")
            plt.close()
        else:
            plt.show()

    @property
    def dataframe(self) -> pd.DataFrame:
        """将分析项转换为DataFrame"""
        if self._df is not None:
            return self._df
            
        data = []
        for item in self.items:
            row = {
                'dataset': item.dataset,
                'evaluator': item.evaluator,
                'solve_time': item.solve_time,
                'method_with_model': item.method_with_model
            }
            # 添加所有指标
            for metric, value in item.metrics.items():
                row[metric] = value
            data.append(row)
        
        self._df = pd.DataFrame(data)
        return self._df

    def get_available_dimensions(self) -> Dict[str, List]:
        """获取可用的分析维度"""
        df = self.dataframe
        return {
            'datasets': df['dataset'].unique().tolist(),
            'evaluators': df['evaluator'].unique().tolist(),
            'method_with_models': df['method_with_model'].unique().tolist(),
            'metrics': [col for col in df.columns if col not in 
                       ['dataset', 'evaluator', 'solve_time', 'method_with_model']]
        }

    def filter_data(self, 
                   datasets: Optional[List[str]] = None,
                   evaluators: Optional[List[str]] = None,
                   method_with_models: Optional[List[str]] = None) -> pd.DataFrame:
        """根据条件过滤数据"""
        df = self.dataframe.copy()
        
        if datasets:
            df = df[df['dataset'].isin(datasets)]
        if evaluators:
            df = df[df['evaluator'].isin(evaluators)]
        if method_with_models:
            df = df[df['method_with_model'].isin(method_with_models)]
            
        return df

    def summary_statistics(self, 
                        group_by: List[str] = None,
                        metrics: List[str] = None,
                        save_path: str = None) -> pd.DataFrame:
        """生成统计摘要"""
        if group_by is None:
            group_by = ['method_with_model']  # 默认按方法+模型组合分组
        if metrics is None:
            metrics = self.get_available_dimensions()['metrics']
        
        # 设置默认保存路径
        if save_path is None:
            summary_dir = f"{self.statistical_charts_dir}/summary/"
            os.makedirs(summary_dir, exist_ok=True)
            save_path = f"{summary_dir}summary_statistics_grouped_by_{'_'.join(group_by)}.csv"
            
        df = self.dataframe
        numeric_metrics = [m for m in metrics if m in df.columns and pd.api.types.is_numeric_dtype(df[m])]
        
        if not numeric_metrics:
            return pd.DataFrame()
            
        summary = df.groupby(group_by)[numeric_metrics].agg(['mean', 'std', 'count'])
        
        # 保存到文件
        try:
            summary.to_csv(save_path)
            print(f"统计摘要已保存到: {save_path}")
        except Exception as e:
            print(f"保存统计摘要时出错: {e}")
        
        return summary

    def compare_methods_with_models(self, metric: str, dataset: str = None) -> pd.DataFrame:
        """比较不同方法+模型组合在指定指标上的表现"""
        df = self.dataframe
        if dataset:
            df = df[df['dataset'] == dataset]
            
        if metric not in df.columns:
            available_metrics = self.get_available_dimensions()['metrics']
            print(f"指标 '{metric}' 不存在。可用指标: {available_metrics}")
            return pd.DataFrame()
            
        comparison = df.groupby('method_with_model')[metric].agg(['mean', 'std', 'count']).round(4)
        comparison = comparison.sort_values('mean', ascending=False)
        return comparison

    def performance_heatmap(self, 
                           x_dim: str = 'method_with_model',
                           y_dim: str = 'dataset',
                           metric: str = None,
                           figsize: tuple = (12, 8)):
        """生成性能热力图"""
        df = self.dataframe
        dimensions = self.get_available_dimensions()
        
        if metric is None:
            metric = dimensions['metrics'][0] if dimensions['metrics'] else None
            
        if not metric or metric not in df.columns:
            print("请指定有效的指标")
            return
            
        # 创建透视表
        pivot_data = df.pivot_table(
            values=metric,
            index=y_dim,
            columns=x_dim,
            aggfunc='mean'
        )
        
        plt.figure(figsize=figsize)
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', center=0.5)
        plt.title(f'{metric} 性能热力图\n({x_dim} vs {y_dim})')
        plt.tight_layout()
        
        # 保存或显示图表
        filename = f"heatmap_{metric}_{x_dim}_vs_{y_dim}"
        subdir = f"{metric}"
        self._save_or_show_plot(plt, "heatmap", filename, subdir)

    def metric_comparison_plot(self, 
                              method_with_models: List[str] = None,
                              datasets: List[str] = None,
                              metrics: List[str] = None):
        """多指标对比图"""
        df = self.filter_data(method_with_models=method_with_models, datasets=datasets)
        dimensions = self.get_available_dimensions()
        
        if metrics is None:
            metrics = dimensions['metrics']
            
        # 筛选数值型指标
        numeric_metrics = [m for m in metrics if m in df.columns and pd.api.types.is_numeric_dtype(df[m])]
        
        if not numeric_metrics:
            print("没有找到可用的数值型指标")
            return
            
        fig, axes = plt.subplots(1, len(numeric_metrics), figsize=(5*len(numeric_metrics), 6))
        if len(numeric_metrics) == 1:
            axes = [axes]
            
        # 生成方法名称标签（用于文件名）
        method_label = "all_methods" if method_with_models is None else "_".join(method_with_models[:3])  # 限制文件名长度
        dataset_label = "all_datasets" if datasets is None else "_".join(datasets[:3])
        
        for i, metric in enumerate(numeric_metrics):
            metric_data = []
            method_names = df['method_with_model'].unique() if method_with_models is None else method_with_models
            
            for method in method_names:
                method_data = df[df['method_with_model'] == method][metric].dropna()
                if len(method_data) > 0:
                    metric_data.append(method_data)
                else:
                    metric_data.append([])
            
            valid_data = [data for data in metric_data if len(data) > 0]
            valid_labels = [method for j, method in enumerate(method_names) if len(metric_data[j]) > 0]
            
            if valid_data:
                axes[i].boxplot(valid_data, labels=valid_labels)
                axes[i].set_title(f'{metric} 分布')
                axes[i].set_xticklabels(valid_labels, rotation=45)
            
        plt.tight_layout()
        
        # 保存或显示图表
        filename = f"metric_comparison_{method_label}_{dataset_label}"
        subdir = "multi_metric"
        self._save_or_show_plot(plt, "comparison", filename, subdir)

    def solve_time_analysis(self):
        """求解时间分析"""
        df = self.dataframe
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 按方法+模型组合分析求解时间
        time_by_method = df.groupby('method_with_model')['solve_time'].mean().sort_values(ascending=False)
        ax1.bar(range(len(time_by_method)), time_by_method.values)
        ax1.set_xticks(range(len(time_by_method)))
        ax1.set_xticklabels(time_by_method.index, rotation=45, ha='right')
        ax1.set_title('各方法+模型组合平均求解时间')
        ax1.set_ylabel('时间 (秒)')
        
        # 按数据集分析求解时间
        time_by_dataset = df.groupby('dataset')['solve_time'].mean().sort_values(ascending=False)
        ax2.bar(range(len(time_by_dataset)), time_by_dataset.values)
        ax2.set_xticks(range(len(time_by_dataset)))
        ax2.set_xticklabels(time_by_dataset.index, rotation=45, ha='right')
        ax2.set_title('各数据集平均求解时间')
        ax2.set_ylabel('时间 (秒)')
        
        plt.tight_layout()
        
        # 保存或显示图表
        filename = "solve_time_analysis"
        self._save_or_show_plot(plt, "time_analysis", filename)

    def interactive_analysis(self):
        """启动交互式分析界面"""
        print("=" * 60)
        print("        LLM 评估结果分析系统")
        print("=" * 60)
        
        while True:
            print("\n请选择分析功能:")
            options = [
                "显示数据摘要",
                "比较方法+模型组合性能", 
                "生成性能热力图",
                "多指标对比图",
                "求解时间分析",
                "显示可用维度",
                "退出系统"
            ]
            
            choice = get_user_choice(options)
            
            try:
                if choice == "显示数据摘要":
                    # 选择分组维度
                    dim_options = ['method_with_model', 'dataset', 'evaluator']
                    group_by_choice = get_user_choice(dim_options, "请选择分组维度: ")
                    summary = self.summary_statistics(group_by=[group_by_choice])
                    print("\n统计摘要:")
                    print(summary)
                    
                elif choice == "比较方法+模型组合性能":
                    # 选择指标
                    metrics = self.get_available_dimensions()['metrics']
                    if not metrics:
                        print("没有可用的指标")
                        continue
                    metric_choice = get_user_choice(metrics, "请选择要比较的指标: ")
                    
                    # 选择数据集（可选）
                    datasets = ['所有数据集'] + self.get_available_dimensions()['datasets']
                    dataset_choice = get_user_choice(datasets, "请选择数据集: ")
                    dataset = None if dataset_choice == '所有数据集' else dataset_choice
                    
                    comparison = self.compare_methods_with_models(metric_choice, dataset)
                    print(f"\n{metric_choice} 方法+模型组合比较:")
                    print(comparison)
                    
                elif choice == "生成性能热力图":
                    # 选择X轴维度
                    x_dim_options = ['method_with_model', 'evaluator']
                    x_choice = get_user_choice(x_dim_options, "请选择X轴维度: ")
                    
                    # 选择Y轴维度  
                    y_dim_options = ['dataset', 'method_with_model']
                    y_choice = get_user_choice(y_dim_options, "请选择Y轴维度: ")
                    
                    # 选择指标
                    metrics = self.get_available_dimensions()['metrics']
                    if not metrics:
                        print("没有可用的指标")
                        continue
                    metric_choice = get_user_choice(metrics, "请选择指标: ")
                    
                    self.performance_heatmap(x_choice, y_choice, metric_choice)
                    
                elif choice == "多指标对比图":
                    # 选择方法+模型组合
                    method_options = ['所有组合'] + self.get_available_dimensions()['method_with_models']
                    method_choice = get_user_choice(method_options, "请选择方法+模型组合: ")
                    methods = None if method_choice == '所有组合' else [method_choice]
                    
                    self.metric_comparison_plot(method_with_models=methods)
                    
                elif choice == "求解时间分析":
                    self.solve_time_analysis()
                    
                elif choice == "显示可用维度":
                    dims = self.get_available_dimensions()
                    print("\n可用维度:")
                    for key, values in dims.items():
                        print(f"{key}: {values}")
                        
                elif choice == "退出系统":
                    print("感谢使用！")
                    break
                    
            except Exception as e:
                print(f"执行操作时出错: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description='LLM 评估结果分析工具')
    parser.add_argument('--results_dir', type=str, default='./outputs/eval_results',
                       help='评估结果目录路径')
    parser.add_argument('--interactive', action='store_true',
                       help='启动交互式模式')
    parser.add_argument('--summary', action='store_true',
                       help='显示数据摘要')
    parser.add_argument('--plot_as_image', action='store_true',
                       help='不显示图表，将其保存为图片')
    parser.add_argument('--statistical_charts_dir', type=str, default='./outputs/statistical_charts',
                       help='统计图表保存目录')
    
    args = parser.parse_args()
    
    # 加载数据
    print("正在加载评估结果...")
    analyzer = EvalResultAnalyzer.load_results(
        args.results_dir, 
        plot_as_image=args.plot_as_image,
        statistical_charts_dir=args.statistical_charts_dir
    )
    
    if len(analyzer.items) == 0:
        print("未找到评估结果文件，请检查目录路径")
        return
    
    print(f"成功加载 {len(analyzer.items)} 个评估结果")
    print(f"图表模式: {'保存为图片' if args.plot_as_image else '直接显示'}")
    if args.plot_as_image:
        print(f"图表保存目录: {args.statistical_charts_dir}")
    
    # 显示基本信息
    dims = analyzer.get_available_dimensions()
    print(f"\n数据集: {dims['datasets']}")
    print(f"评估器: {dims['evaluators']}")
    print(f"方法+模型组合: {dims['method_with_models']}")
    print(f"评估指标: {dims['metrics']}")
    
    if args.summary:
        summary = analyzer.summary_statistics()
        print("\n统计摘要:")
        print(summary)
    
    if args.interactive or (not args.summary):
        analyzer.interactive_analysis()

if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()