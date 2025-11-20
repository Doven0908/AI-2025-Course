// 处理用户提交的问题，并调用后端问答接口
async function askQuestion(event) {
    event.preventDefault();
    const query = document.getElementById("query").value.trim();
    const topK = parseInt(document.getElementById("top_k").value, 10);

    if (!query) {
        alert("请输入问题");
        return;
    }

    const payload = { query: query };
    if (Number.isFinite(topK) && topK > 0) {
        payload.top_k = topK;
    }

    try {
        // ✅ 添加超时控制（60秒）
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000);

        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: controller.signal  // ✅ 关联中止信号
        });

        clearTimeout(timeoutId);  // ✅ 清除超时定时器

        if (!response.ok) {
            const data = await response.json();
            alert("❌ 请求失败: " + data.detail);
            return;
        }

        const data = await response.json();
        renderAnswer(data);
        await refreshMetrics();

    } catch (error) {
        // ✅ 捕获所有错误
        if (error.name === 'AbortError') {
            alert("⏱️ 请求超时，服务器处理时间过长，请稍后重试");
        } else if (error.message.includes('Failed to fetch')) {
            alert("❌ 网络错误：无法连接到服务器，请检查服务器是否运行");
        } else {
            alert("❌ 抱歉，处理请求时出现错误: " + error.message);
        }
        console.error("请求失败详情:", error);
    }
}