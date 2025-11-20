(() => {
    const config = window.RAGAS_CONFIG || {};
    const datasetList = document.getElementById("ragas-dataset");
    const variantSelect = document.getElementById("ragas-variant");
    const runButton = document.getElementById("ragas-run");
    const summaryBox = document.getElementById("ragas-summary");
    const metricsTable = document.getElementById("ragas-metrics-table");
    const metricsBody = metricsTable.querySelector("tbody");
    const samplesBox = document.getElementById("ragas-samples");

    if (!config.datasetUrl || !config.resultsUrl || !config.runUrl) {
        console.error("缺少 RAGAS 配置，无法初始化。");
        return;
    }

    const buildUrl = (template, variant) =>
        template.replace("__placeholder__", encodeURIComponent(variant));

    const fmtScore = (value) => {
        if (value === null || value === undefined) return "-";
        return Number(value).toFixed(3);
    };

    const loadDataset = async () => {
        datasetList.innerHTML = "<li>加载中...</li>";
        try {
            const resp = await fetch(config.datasetUrl);
            if (!resp.ok) {
                throw new Error(`加载失败 (${resp.status})`);
            }
            const data = await resp.json();
            const samples = data.samples || [];
            if (samples.length === 0) {
                datasetList.innerHTML = "<li>尚未配置评估数据</li>";
                return;
            }
            datasetList.innerHTML = "";
            samples.forEach((sample, idx) => {
                const li = document.createElement("li");
                li.innerHTML = `<strong>${idx + 1}. ${sample.question}</strong><br><small>${(sample.ground_truths || []).join("；")}</small>`;
                datasetList.appendChild(li);
            });
        } catch (error) {
            datasetList.innerHTML = `<li>加载数据集失败：${error.message}</li>`;
            console.error("加载评估数据集失败:", error);
        }
    };

    const renderResults = (result) => {
        if (!result) {
            summaryBox.innerHTML = "<p>尚未生成评估结果</p>";
            metricsTable.classList.add("hidden");
            samplesBox.innerHTML = "<p>暂无样本数据</p>";
            return;
        }

        const runAt = result.run_at
            ? new Date(result.run_at).toLocaleString("zh-CN")
            : "未知时间";
        summaryBox.innerHTML = `<p>检索策略：<strong>${config.variants[result.variant]?.label || result.variant}</strong> ｜ 最近评估时间：${runAt}</p>`;

        const metrics = result.metrics || [];
        if (metrics.length === 0) {
            metricsTable.classList.add("hidden");
        } else {
            metricsBody.innerHTML = "";
            metrics.forEach((metric) => {
                const row = document.createElement("tr");
                const ciLow = metric.ci_low != null ? fmtScore(metric.ci_low) : null;
                const ciHigh = metric.ci_high != null ? fmtScore(metric.ci_high) : null;
                const ciText = ciLow && ciHigh ? `[${ciLow}, ${ciHigh}]` : "-";
                row.innerHTML = `
                    <td>${metric.name}</td>
                    <td>${fmtScore(metric.score)}</td>
                    <td>${ciText}</td>
                `;
                metricsBody.appendChild(row);
            });
            metricsTable.classList.remove("hidden");
        }

        const samples = result.samples || [];
        if (samples.length === 0) {
            samplesBox.innerHTML = "<p>暂无样本数据</p>";
        } else {
            samplesBox.innerHTML = "";
            samples.forEach((sample) => {
                const card = document.createElement("div");
                card.className = "sample-card";
                const contexts = Array.isArray(sample.contexts)
                    ? sample.contexts.map((ctx) => `<li>${ctx}</li>`).join("")
                    : "";
                card.innerHTML = `
                    <h4>问题</h4>
                    <p>${sample.question}</p>
                    <h4>回答</h4>
                    <p>${sample.answer}</p>
                    <h4>参考答案</h4>
                    <p>${(sample.ground_truths || []).join("；") || "未提供"}</p>
                    <h4>检索上下文</h4>
                    <ul>${contexts}</ul>
                `;
                samplesBox.appendChild(card);
            });
        }
    };

    const loadResults = async () => {
        const variant = variantSelect.value;
        try {
            const resp = await fetch(buildUrl(config.resultsUrl, variant));
            if (!resp.ok) {
                if (resp.status === 404) {
                    renderResults(null);
                    return;
                }
                throw new Error(`加载失败 (${resp.status})`);
            }
            const data = await resp.json();
            renderResults(data);
        } catch (error) {
            summaryBox.innerHTML = `<p>加载评估结果失败：${error.message}</p>`;
            metricsTable.classList.add("hidden");
            samplesBox.innerHTML = "<p>暂无样本数据</p>";
            console.error("加载 RAGAS 结果失败:", error);
        }
    };

    runButton.addEventListener("click", async () => {
        const variant = variantSelect.value;
        runButton.disabled = true;
        runButton.textContent = "运行中...";
        summaryBox.innerHTML = "<p>评估执行中，请稍候...</p>";
        try {
            const resp = await fetch(buildUrl(config.runUrl, variant), {
                method: "POST",
            });
            if (!resp.ok) {
                const detail = await resp.json().catch(() => ({}));
                throw new Error(detail.detail || `运行失败 (${resp.status})`);
            }
            const data = await resp.json();
            renderResults(data.result);
        } catch (error) {
            summaryBox.innerHTML = `<p>评估执行失败：${error.message}</p>`;
            console.error("运行 RAGAS 失败:", error);
        } finally {
            runButton.disabled = false;
            runButton.textContent = "运行评估";
        }
    });

    variantSelect.addEventListener("change", () => {
        loadResults();
    });

    loadDataset();
    loadResults();
})();
