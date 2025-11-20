(() => {
    const config = window.QA_CONFIG || {};
    const form = document.getElementById("qa-form");
    const queryInput = document.getElementById("qa-query");
    const topkInput = document.getElementById("qa-topk");
    const submitBtn = document.getElementById("qa-submit");
    const answerBox = document.getElementById("qa-answer");
    const citationsBox = document.getElementById("qa-citations");
    const metricsBody = document.getElementById("qa-metrics-body");
    const referenceInput = document.getElementById("qa-reference");
    const evaluationBox = document.getElementById("qa-evaluation");
    const evaluateBtn = document.getElementById("qa-run-eval");

    const metricLabels = {
        faithfulness: "Faithfulness",
        answer_relevancy: "Answer Relevancy",
        context_precision: "Context Precision",
        context_recall: "Context Recall",
    };

    let latestAnswer = null;
    let latestQuery = "";

    const escapeHtml = (str) =>
        str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");

    const applyInlineMarkdown = (text) => {
        let formatted = escapeHtml(text);
        formatted = formatted.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        formatted = formatted.replace(/\*(.+?)\*/g, "<em>$1</em>");
        formatted = formatted.replace(/`([^`]+)`/g, "<code>$1</code>");
        formatted = formatted.replace(
            /\[(.+?)\]\((https?:\/\/[^\s)]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener">$1</a>'
        );
        formatted = formatted.replace(/\n/g, "<br>");
        return formatted;
    };

    const markdownToHtml = (markdown) => {
        if (!markdown) return "";
        const lines = markdown.split(/\r?\n/);
        let html = "";
        let inList = false;
        let inCodeBlock = false;
        let codeBuffer = [];
        let paragraphBuffer = [];

        const flushParagraphBuffer = () => {
            if (paragraphBuffer.length === 0) return;
            const content = paragraphBuffer.join(" ");
            html += `<p>${applyInlineMarkdown(content)}</p>`;
            paragraphBuffer = [];
        };

        const flushList = () => {
            if (inList) {
                html += "</ul>";
                inList = false;
            }
        };

        const flushCodeBlock = () => {
            if (inCodeBlock) {
                const codeContent = codeBuffer.join("\n");
                html += `<pre><code>${escapeHtml(codeContent)}</code></pre>`;
                codeBuffer = [];
                inCodeBlock = false;
            }
        };

        lines.forEach((line) => {
            if (line.trim().startsWith("```")) {
                if (inCodeBlock) {
                    flushCodeBlock();
                } else {
                    flushParagraphBuffer();
                    flushList();
                    inCodeBlock = true;
                    codeBuffer = [];
                }
                return;
            }

            if (inCodeBlock) {
                codeBuffer.push(line);
                return;
            }

            if (/^\s*[-*]\s+/.test(line)) {
                flushParagraphBuffer();
                if (!inList) {
                    html += "<ul>";
                    inList = true;
                }
                const itemContent = line.replace(/^\s*[-*]\s+/, "");
                html += `<li>${applyInlineMarkdown(itemContent)}</li>`;
                return;
            }

            if (line.trim() === "") {
                flushParagraphBuffer();
                flushList();
                return;
            }

            flushList();
            paragraphBuffer.push(line.trim());
        });

        flushParagraphBuffer();
        flushList();
        flushCodeBlock();

        return html || applyInlineMarkdown(markdown);
    };

    if (!config.askUrl || !config.metricsUrl || !config.evaluateUrl) {
        console.error("缺少 QA 配置信息，无法初始化。");
        return;
    }

    const setLoading = (loading) => {
        submitBtn.disabled = loading;
        submitBtn.textContent = loading ? "处理中..." : "发送请求";
    };

    const renderAnswer = (data) => {
        answerBox.classList.remove("answer-placeholder");
        const html = markdownToHtml(data.answer || "");
        answerBox.innerHTML = html || "未获取到回答";

        citationsBox.innerHTML = "";
        if (Array.isArray(data.citations) && data.citations.length > 0) {
            data.citations.forEach((item) => {
                const div = document.createElement("div");
                div.className = "citation-item";
                const label = item.label ? `${item.label} ` : "";
                const source = item.source || "未知来源";
                const excerpt = item.excerpt || "暂无引用摘要";
                const excerptHtml = applyInlineMarkdown(excerpt);
                div.innerHTML = `<strong>${label}${source}</strong><div>${excerptHtml}</div>`;
                citationsBox.appendChild(div);
            });
        }
    };

    const renderMetricsTable = (records) => {
        metricsBody.innerHTML = "";
        if (!Array.isArray(records) || records.length === 0) {
            const row = document.createElement("tr");
            row.innerHTML = `<td colspan="6" class="table-empty">暂无数据</td>`;
            metricsBody.appendChild(row);
            return;
        }

        records.slice(0, 10).forEach((record) => {
            const row = document.createElement("tr");
            const timestamp = record.timestamp
                ? new Date(record.timestamp).toLocaleString("zh-CN")
                : "-";
            const query = record.query ? record.query.slice(0, 40) : "-";
            row.innerHTML = `
                <td>${timestamp}</td>
                <td>${query}</td>
                <td>${Math.round(record.latency_ms ?? 0)}</td>
                <td>${Math.round(record.retrieval_ms ?? 0)}</td>
                <td>${Math.round(record.generation_ms ?? 0)}</td>
                <td>${record.retrieved_k ?? "-"}</td>
            `;
            metricsBody.appendChild(row);
        });
    };

    const loadMetrics = async () => {
        try {
            const resp = await fetch(config.metricsUrl);
            if (!resp.ok) throw new Error(`拉取指标失败: ${resp.status}`);
            const data = await resp.json();
            renderMetricsTable(data.records);
        } catch (error) {
            console.error("加载指标失败:", error);
        }
    };

    const formatMetricName = (metric) => {
        const rawName = metric.label || metric.name;
        if (!rawName) return "指标";
        const key = rawName.toLowerCase();
        if (metricLabels[key]) return metricLabels[key];
        return rawName.charAt(0).toUpperCase() + rawName.slice(1);
    };

    const renderEvaluation = (evaluation) => {
        if (!evaluationBox) return;

        if (!evaluation || !Array.isArray(evaluation.metrics) || evaluation.metrics.length === 0) {
            if (evaluation && evaluation.error) {
                evaluationBox.classList.remove("evaluation-placeholder");
                evaluationBox.innerHTML = `⚠️ 评估失败：${evaluation.error}`;
                return;
            }
            evaluationBox.classList.add("evaluation-placeholder");
            evaluationBox.innerHTML = "未返回可用的评估结果。";
            return;
        }

        evaluationBox.classList.remove("evaluation-placeholder");
        const source = evaluation.ground_truth_source || "none";
        const references = Array.isArray(evaluation.references) ? evaluation.references : [];
        const diagnosis = Array.isArray(evaluation.diagnosis) ? evaluation.diagnosis : [];

        let headerNote;
        if (source === "generated") {
            headerNote = "系统已自动生成参考答案（评估模型），综合评估可证性与相关性指标。";
        } else if (source === "user") {
            headerNote = "已使用你提供的参考答案计算相关性/召回指标。";
        } else if (source === "error") {
            headerNote = evaluation.error
                ? `评估出现异常：${evaluation.error}`
                : "评估出现异常，请稍后再试。";
        } else {
            headerNote = "未提供参考答案或未启用自动生成，仅展示基于上下文的一致性指标。";
        }

        const metricRows = evaluation.metrics
            .map((metric) => {
                const score =
                    typeof metric.score === "number"
                        ? metric.score.toFixed(3)
                        : "-";
                let ci = "-";
                if (typeof metric.ci_low === "number" && typeof metric.ci_high === "number") {
                    ci = `[${metric.ci_low.toFixed(3)}, ${metric.ci_high.toFixed(3)}]`;
                }
                const label = formatMetricName(metric);
                return `<tr><td>${label}</td><td>${score}</td><td>${ci}</td></tr>`;
            })
            .join("");

        const referenceHtml =
            references.length > 0
                ? `
            <div class="evaluation-reference">
                <span>参考答案：</span>
                ${references.map((ref) => `<div>${markdownToHtml(ref)}</div>`).join("")}
            </div>
        `
                : "";

        const diagnosisHtml =
            diagnosis.length > 0
                ? `
            <div class="evaluation-diagnosis">
                <h4>误差归因</h4>
                ${diagnosis
                    .map(
                        (item) =>
                            `<div class="diagnosis-item"><strong>${item.type}：</strong>${markdownToHtml(
                                item.detail
                            )}</div>`
                    )
                    .join("")}
            </div>
        `
                : "";

        evaluationBox.innerHTML = `
            <p class="evaluation-note">${headerNote}</p>
            ${
                metricRows
                    ? `<table class="evaluation-table">
                <thead>
                    <tr><th>指标</th><th>得分</th><th>置信区间</th></tr>
                </thead>
                <tbody>${metricRows}</tbody>
            </table>`
                    : '<p class="evaluation-note">尚未返回指标分数。</p>'
            }
            ${referenceHtml}
            ${diagnosisHtml}
        `;
    };

    form.addEventListener("submit", async (evt) => {
        evt.preventDefault();
        const query = queryInput.value.trim();
        const topK = parseInt(topkInput.value, 10);

        if (!query) {
            alert("请输入问题");
            return;
        }

        setLoading(true);
        latestAnswer = null;
        latestQuery = query;
        if (evaluateBtn) evaluateBtn.disabled = true;

        answerBox.classList.add("answer-placeholder");
        answerBox.textContent = "正在思考...";
        citationsBox.innerHTML = "";
        if (evaluationBox) {
            evaluationBox.classList.add("evaluation-placeholder");
            evaluationBox.innerHTML = "等待评估。";
        }

        const payload = { query };
        if (Number.isFinite(topK) && topK > 0) {
            payload.top_k = topK;
        }

        try {
            const resp = await fetch(config.askUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!resp.ok) {
                const detail = await resp.json().catch(() => ({}));
                throw new Error(detail.detail || `请求失败 (${resp.status})`);
            }

            const data = await resp.json();
            latestAnswer = { ...data, query };
            renderAnswer(data);
            if (evaluationBox) {
                evaluationBox.classList.add("evaluation-placeholder");
                evaluationBox.innerHTML = "点击“运行 RAGAS 评估”以获取指标。";
            }
            if (evaluateBtn) {
                evaluateBtn.disabled = false;
            }
            await loadMetrics();
        } catch (error) {
            answerBox.classList.remove("answer-placeholder");
            answerBox.innerHTML = `❌ 请求失败：${error.message}`;
            console.error("请求失败:", error);
            if (evaluationBox) {
                evaluationBox.classList.remove("evaluation-placeholder");
                evaluationBox.innerHTML = `⚠️ 无法生成回答：${error.message}`;
            }
        } finally {
            setLoading(false);
        }
    });

    if (evaluateBtn) {
        evaluateBtn.addEventListener("click", async () => {
            if (!latestAnswer) {
                alert("请先生成回答，再进行评估。");
                return;
            }

            const contexts = Array.isArray(latestAnswer.contexts)
                ? latestAnswer.contexts.map((ctx) => ctx.text || "").filter(Boolean)
                : [];
            if (contexts.length === 0) {
                alert("当前回答缺少上下文，无法执行评估。");
                return;
            }

            const reference = referenceInput ? referenceInput.value.trim() : "";
            const payload = {
                question: latestAnswer.query || latestQuery || "",
                answer: latestAnswer.answer || "",
                contexts,
            };
            if (reference) {
                payload.ground_truths = [reference];
            }

            evaluateBtn.disabled = true;
            if (evaluationBox) {
                evaluationBox.classList.add("evaluation-placeholder");
                evaluationBox.innerHTML = "评估计算中...";
            }

            try {
                const resp = await fetch(config.evaluateUrl, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                if (!resp.ok) {
                    const detail = await resp.json().catch(() => ({}));
                    throw new Error(detail.detail || `评估失败 (${resp.status})`);
                }
                const data = await resp.json();
                renderEvaluation(data);
            } catch (error) {
                console.error("评估失败:", error);
                if (evaluationBox) {
                    evaluationBox.classList.remove("evaluation-placeholder");
                    evaluationBox.innerHTML = `⚠️ 评估失败：${error.message}`;
                }
            } finally {
                evaluateBtn.disabled = false;
            }
        });
    }

    loadMetrics();
    setInterval(loadMetrics, 30000);
})();
