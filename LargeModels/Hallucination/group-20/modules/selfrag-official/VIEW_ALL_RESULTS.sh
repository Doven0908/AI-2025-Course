#!/bin/bash
# 查看所有生成的评估结果

echo "=================================="
echo "Self-RAG 评估结果 - 快速查看"
echo "=================================="
echo ""

echo "📊 可视化图表 (15个):"
echo "-------------------"
cd ragas_results
ls -lh *.png *.svg | awk '{print "  " $9, " - ", $5}'
echo ""

echo "📄 文档报告 (5个):"
echo "-------------------"
ls -lh *.md | awk '{print "  " $9, " - ", $5}'
echo ""

echo "📊 数据文件 (5个):"
echo "-------------------"
ls -lh *.json | awk '{print "  " $9, " - ", $5}'
echo ""

echo "📁 总计:"
echo "-------------------"
echo "  文件数: $(ls -1 | wc -l)"
echo "  总大小: $(du -sh . | awk '{print $1}')"
echo ""

echo "🎯 快速导航:"
echo "-------------------"
echo "  1. 查看索引:        cat INDEX.md"
echo "  2. 查看使用指南:    cat README.md"
echo "  3. 查看技术报告:    cat COMPREHENSIVE_TECHNICAL_REPORT.md"
echo "  4. 查看演示文稿:    cat PRESENTATION_SUMMARY.md"
echo "  5. 查看案例数据:    cat case_studies.json"
echo ""

echo "✅ 所有材料已准备就绪！"
echo ""
