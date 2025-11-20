const state = {
  dataset: [],
  groupedBySubject: new Map(),
  currentSubject: null,
  currentQuestion: null,
  running: false,
  lastMetrics: null,
  priceTable: {
    "gpt-4o-mini": { input: 0.00015, output: 0.0006 },
    "gpt-4o": { input: 0.005, output: 0.015 },
    "gpt-4.1-mini": { input: 0.00015, output: 0.0006 },
    "gpt-4.1": { input: 0.005, output: 0.015 },
    "qwen-plus": { input: 0, output: 0 }
  },
  logHistory: []
};

const elements = {};

document.addEventListener("DOMContentLoaded", async () => {
  cacheElements();
  restorePersistedState();
  wireEventHandlers();
  await loadDataset();
  renderMetrics();
});

function cacheElements() {
  elements.apiKeyInput = document.getElementById("apiKeyInput");
  elements.rememberKey = document.getElementById("rememberKey");
  elements.toggleKeyVisibility = document.getElementById("toggleKeyVisibility");
  elements.baseUrlInput = document.getElementById("baseUrlInput");
  elements.subjectSelect = document.getElementById("subjectSelect");
  elements.questionSelect = document.getElementById("questionSelect");
  elements.randomQuestion = document.getElementById("randomQuestion");
  elements.strategySelect = document.getElementById("strategySelect");
  elements.modelInput = document.getElementById("modelInput");
  elements.temperatureInput = document.getElementById("temperatureInput");
  elements.maxTokensInput = document.getElementById("maxTokensInput");
  elements.runButton = document.getElementById("runButton");
  elements.referenceAnswer = document.getElementById("referenceAnswer");
  elements.notesInput = document.getElementById("notesInput");
  elements.metricsGrid = document.getElementById("metricsGrid");
  elements.logStream = document.getElementById("logStream");
  elements.logEntryTemplate = document.getElementById("logEntryTemplate");
}

function restorePersistedState() {
  const storedKey = window.localStorage.getItem("prompt-playground-key");
  if (storedKey) {
    elements.apiKeyInput.value = storedKey;
    elements.rememberKey.checked = true;
  }

  const storedBaseUrl = window.localStorage.getItem("prompt-playground-base-url");
  if (storedBaseUrl) {
    elements.baseUrlInput.value = storedBaseUrl;
  }

  const storedNotes = window.localStorage.getItem("prompt-playground-notes");
  if (storedNotes) {
    elements.notesInput.value = storedNotes;
  }
}

function wireEventHandlers() {
  elements.toggleKeyVisibility.addEventListener("click", () => {
    const currentType = elements.apiKeyInput.getAttribute("type");
    elements.apiKeyInput.setAttribute("type", currentType === "password" ? "text" : "password");
    elements.toggleKeyVisibility.textContent = currentType === "password" ? "Hide key" : "Show key";
  });

  elements.rememberKey.addEventListener("change", () => {
    if (elements.rememberKey.checked) {
      window.localStorage.setItem("prompt-playground-key", elements.apiKeyInput.value);
    } else {
      window.localStorage.removeItem("prompt-playground-key");
    }
  });

  elements.apiKeyInput.addEventListener("input", () => {
    if (elements.rememberKey.checked) {
      window.localStorage.setItem("prompt-playground-key", elements.apiKeyInput.value);
    }
  });

  elements.baseUrlInput.addEventListener("input", () => {
    window.localStorage.setItem("prompt-playground-base-url", elements.baseUrlInput.value.trim());
  });

  elements.notesInput.addEventListener("input", () => {
    window.localStorage.setItem("prompt-playground-notes", elements.notesInput.value);
  });

  elements.subjectSelect.addEventListener("change", () => {
    state.currentSubject = elements.subjectSelect.value;
    renderQuestionOptions();
  });

  elements.questionSelect.addEventListener("change", () => {
    const questionId = elements.questionSelect.value;
    state.currentQuestion = state.dataset.find(item => item.id === questionId) || null;
    renderReferenceAnswer();
  });

  elements.randomQuestion.addEventListener("click", () => {
    const subject = state.currentSubject;
    const pool = subject ? state.groupedBySubject.get(subject) || [] : state.dataset;
    if (!pool.length) return;
    const randomItem = pool[Math.floor(Math.random() * pool.length)];
    elements.questionSelect.value = randomItem.id;
    elements.questionSelect.dispatchEvent(new Event("change"));
  });

  elements.runButton.addEventListener("click", async () => {
    if (!state.currentQuestion) {
      alert("Select a question before running a strategy.");
      return;
    }

    const apiKey = elements.apiKeyInput.value.trim();
    if (!apiKey) {
      alert("Provide a valid API key.");
      return;
    }

    const strategy = elements.strategySelect.value;
    const model = elements.modelInput.value.trim() || "qwen-plus";
    const temperature = Number(elements.temperatureInput.value) || 0.3;
    const maxTokens = Number(elements.maxTokensInput.value) || 1000;
    const baseUrl = (elements.baseUrlInput.value || "https://dashscope.aliyuncs.com/compatible-mode/v1").trim();

    setRunning(true);
    clearMetrics();

    try {
      await runStrategy(strategy, {
        apiKey,
        model,
        temperature,
        maxTokens,
        baseUrl
      });
    } catch (error) {
      console.error(error);
      addLogEntry({
        title: `Error: ${strategy}`,
        bodyHtml: `<p>${escapeHtml(error.message || error)}</p>`,
        metrics: [],
        timestamp: new Date(),
        isError: true
      });
    } finally {
      setRunning(false);
    }
  });
}

async function loadDataset() {
  try {
    const response = await fetch("data/mmlu_samples.json");
    if (!response.ok) throw new Error(`Unable to load dataset (${response.status})`);
    const dataset = await response.json();
    state.dataset = dataset;
    state.groupedBySubject = groupBySubject(dataset);
    renderSubjectOptions();
    renderQuestionOptions();
  } catch (error) {
    console.error(error);
    addLogEntry({
      title: "Dataset load failed",
      bodyHtml: `<p>${escapeHtml(error.message || error)}</p>`,
      metrics: [],
      timestamp: new Date(),
      isError: true
    });
  }
}

function groupBySubject(dataset) {
  return dataset.reduce((map, item) => {
    if (!map.has(item.subject)) {
      map.set(item.subject, []);
    }
    map.get(item.subject).push(item);
    return map;
  }, new Map());
}

function renderSubjectOptions() {
  const subjects = Array.from(state.groupedBySubject.keys()).sort();
  elements.subjectSelect.innerHTML = subjects
    .map(subject => `<option value="${escapeHtml(subject)}">${escapeHtml(subject)}</option>`)
    .join("");
  state.currentSubject = subjects[0] || null;
  if (state.currentSubject) {
    elements.subjectSelect.value = state.currentSubject;
  }
}

function renderQuestionOptions() {
  const subject = state.currentSubject;
  const questions = subject ? state.groupedBySubject.get(subject) || [] : state.dataset;

  elements.questionSelect.innerHTML = questions
    .map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(truncate(item.question, 80))}</option>`)
    .join("");

  state.currentQuestion = questions[0] || null;
  if (state.currentQuestion) {
    elements.questionSelect.value = state.currentQuestion.id;
  }
  renderReferenceAnswer();
}

function renderReferenceAnswer() {
  const question = state.currentQuestion;
  if (!question) {
    elements.referenceAnswer.textContent = "";
    return;
  }

  const { question: stem, choices, answer, subject } = question;
  const formatted = [`Subject: ${subject}`, "", stem, ""];
  for (const [key, value] of Object.entries(choices)) {
    formatted.push(`${key}. ${value}`);
  }
  formatted.push("", `Correct answer: ${answer}`);
  elements.referenceAnswer.textContent = formatted.join("\n");
}

function setRunning(flag) {
  state.running = flag;
  elements.runButton.disabled = flag;
  elements.runButton.textContent = flag ? "Running..." : "Run strategy";
}

async function runStrategy(strategy, config) {
  switch (strategy) {
    case "baseFewShot":
      await executeBaseFewShot(config);
      break;
    case "instruct":
      await executeInstruct(config);
      break;
    case "cot":
      await executeChainOfThought(config);
      break;
    case "selfReflection":
      await executeSelfReflection(config);
      break;
    case "cotVote":
      await executeCotVote(config);
      break;
    default:
      throw new Error(`Strategy ${strategy} is not implemented.`);
  }
}

async function executeBaseFewShot(config) {
  const shotPool = state.dataset.filter(item => item.id !== state.currentQuestion.id);
  const shots = pickRandom(shotPool, 2);
  const prompt = buildFewShotPrompt(state.currentQuestion, shots);
  const system = buildSystemPrompt();

  const run = await callModel({
    messages: [
      { role: "system", content: system },
      { role: "user", content: prompt }
    ],
    label: "Base + 2 shots",
    config
  });
  finalizeSingleRun(run, "Base + 2 shots");
}

async function executeInstruct(config) {
  const prompt = buildInstructionPrompt(state.currentQuestion);
  const system = "You are an instruction-following expert for multiple-choice exams. Respond strictly in JSON.";

  const run = await callModel({
    messages: [
      { role: "system", content: `${system}\n${jsonOutputReminder()}` },
      { role: "user", content: prompt }
    ],
    label: "Instruct",
    config
  });
  finalizeSingleRun(run, "Instruct");
}

async function executeChainOfThought(config) {
  const prompt = buildCotPrompt(state.currentQuestion);
  const system = `You reason step by step and deliver concise JSON results. ${jsonOutputReminder()}`;

  const run = await callModel({
    messages: [
      { role: "system", content: system },
      { role: "user", content: prompt }
    ],
    label: "Chain of Thought",
    config
  });
  finalizeSingleRun(run, "Chain of Thought");
}

async function executeSelfReflection(config) {
  const basePrompt = buildCotPrompt(state.currentQuestion);
  const system = `You must respond with JSON containing reasoning, final_answer, answer_letter, and confidence.`;

  const firstPass = await callModel({
    messages: [
      { role: "system", content: `${system} ${jsonOutputReminder()}` },
      { role: "user", content: `${basePrompt}\nRemember to answer in JSON format.` }
    ],
    label: "Self-reflection | draft",
    config: { ...config, temperature: Math.max(config.temperature, 0.5) }
  });

  const parsedDraft = parseModelPayload(firstPass.content);
  const reflectionPrompt = buildReflectionPrompt(state.currentQuestion, parsedDraft);

  const secondPass = await callModel({
    messages: [
      { role: "system", content: `${system} ${jsonOutputReminder()}` },
      { role: "user", content: reflectionPrompt }
    ],
    label: "Self-reflection | critique",
    config
  });

  const aggregate = combineRuns([firstPass, secondPass], "Self-reflection");
  renderMetrics(aggregate.metrics);
  for (const entry of aggregate.entries) {
    addLogEntry(entry);
  }
}

async function executeCotVote(config) {
  const prompt = buildCotPrompt(state.currentQuestion);
  const system = `You are part of a reasoning committee. Provide JSON with reasoning, answer_letter, final_answer, and confidence.`;
  const runs = [];
  for (let i = 0; i < 5; i += 1) {
    const run = await callModel({
      messages: [
        { role: "system", content: `${system} ${jsonOutputReminder()}` },
        { role: "user", content: `${prompt}\nYou are voter #${i + 1}.` }
      ],
      label: `CoT voter #${i + 1}`,
      config: { ...config, temperature: Math.max(config.temperature, 0.7) }
    });
    runs.push(run);
  }

  const aggregate = combineRuns(runs, "CoT vote (N=5)");
  renderMetrics(aggregate.metrics);
  for (const entry of aggregate.entries) {
    addLogEntry(entry);
  }
}

function finalizeSingleRun(run, label) {
  const metrics = summarizeRuns([run]);
  renderMetrics(metrics);
  addLogEntry(formatRunEntry(run, label));
}

function combineRuns(runs, label) {
  const metrics = summarizeRuns(runs);
  const entries = runs.map((run, idx) => formatRunEntry(run, `${label} | ${idx + 1}`));
  const majorityEntry = buildMajorityEntry(runs, label, metrics);
  if (majorityEntry) {
    entries.push(majorityEntry);
  }
  return { metrics, entries };
}

function summarizeRuns(runs) {
  const totalPromptTokens = runs.reduce((sum, run) => sum + (run.usage.prompt_tokens || 0), 0);
  const totalCompletionTokens = runs.reduce((sum, run) => sum + (run.usage.completion_tokens || 0), 0);
  const totalTokens = totalPromptTokens + totalCompletionTokens;
  const totalDurationMs = runs.reduce((sum, run) => sum + run.durationMs, 0);
  const totalCost = runs.reduce((sum, run) => sum + (run.costUsd || 0), 0);
  const accuracyCount = runs.filter(run => run.isCorrect).length;

  const metrics = {
    runCount: runs.length,
    totalPromptTokens,
    totalCompletionTokens,
    totalTokens,
    totalDurationMs,
    totalCost,
    accuracy: runs.length ? accuracyCount / runs.length : 0,
    lastAnswer: runs[runs.length - 1]?.answerLetter || null,
    lastCorrect: runs[runs.length - 1]?.isCorrect || false
  };
  state.lastMetrics = metrics;
  return metrics;
}

function buildMajorityEntry(runs, label, metrics) {
  if (!runs.length) return null;
  const voteTally = runs.reduce((map, run) => {
    if (!run.answerLetter) return map;
    map.set(run.answerLetter, (map.get(run.answerLetter) || 0) + 1);
    return map;
  }, new Map());

  if (!voteTally.size) return null;
  let majorityLetter = null;
  let topVotes = -1;
  for (const [letter, votes] of voteTally.entries()) {
    if (votes > topVotes) {
      majorityLetter = letter;
      topVotes = votes;
    }
  }

  const isCorrect = majorityLetter === state.currentQuestion.answer;
  const summary = Array.from(voteTally.entries())
    .map(([letter, votes]) => `${letter}: ${votes}`)
    .join(", ");

  return {
    title: `${label} | majority`,
    bodyHtml: `<p>Vote breakdown: ${escapeHtml(summary)}</p><p>Majority answer: <strong>${escapeHtml(majorityLetter || "n/a")}</strong></p>`,
    metrics: [
      `Combined tokens: ${metrics.totalTokens}`,
      `Cumulative time: ${formatMs(metrics.totalDurationMs)}`,
      isCorrect ? `<span class="correct">Majority correct</span>` : "Majority incorrect"
    ],
    timestamp: new Date(),
    isCorrect
  };
}

async function callModel({ messages, label, config }) {
  const { apiKey, model, temperature, maxTokens, baseUrl } = config;
  const started = performance.now();
  const normalizedBase = (baseUrl || "https://dashscope.aliyuncs.com/compatible-mode/v1").replace(/\/+$/, "");
  const endpoint = normalizedBase.endsWith("/chat/completions") ? normalizedBase : `${normalizedBase}/chat/completions`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      temperature,
      max_tokens: maxTokens,
      messages
    })
  });

  const elapsed = performance.now() - started;
  const timestamp = new Date();

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`${label} failed: ${errorText}`);
  }

  const data = await response.json();
  const choice = data.choices?.[0];
  const content = choice?.message?.content?.trim() || "";
  const usage = data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };

  const parsed = parseModelPayload(content);
  const answerLetter = parsed.answer_letter || parsed.final_answer || inferLetterFromText(content);
  const isCorrect = answerLetter ? answerLetter.toUpperCase() === state.currentQuestion.answer : false;
  const costUsd = estimateCost(model, usage.prompt_tokens, usage.completion_tokens);

  return {
    label,
    content,
    parsed,
    usage,
    timestamp,
    durationMs: elapsed,
    answerLetter: answerLetter ? answerLetter.toUpperCase() : null,
    isCorrect,
    costUsd
  };
}

function formatRunEntry(run, label) {
  const reasoning = run.parsed.reasoning || run.parsed.thought_process || run.content;
  const confidence = run.parsed.confidence != null ? Number(run.parsed.confidence).toFixed(2) : "n/a";
  const answer = run.answerLetter || "n/a";
  const reasoningHtml = formatReasoningHtml(reasoning);
  const metrics = [
    `Tokens: ${run.usage.total_tokens || run.usage.prompt_tokens + run.usage.completion_tokens}`,
    `Latency: ${formatMs(run.durationMs)}`
  ];
  if (run.costUsd != null) {
    metrics.push(`Cost: $${run.costUsd.toFixed(4)}`);
  }
  metrics.push(run.isCorrect ? `<span class="correct">Correct</span>` : "Incorrect");

  return {
    title: label,
    bodyHtml: `
      <p><strong>Answer:</strong> ${escapeHtml(answer)}</p>
      <p><strong>Confidence:</strong> ${escapeHtml(confidence)}</p>
      <div class="reasoning-block">
        <p class="reasoning-title">Reasoning</p>
        <div class="reasoning-body">${reasoningHtml}</div>
      </div>
    `,
    metrics,
    timestamp: run.timestamp,
    isCorrect: run.isCorrect
  };
}

function renderMetrics(metrics = state.lastMetrics) {
  elements.metricsGrid.innerHTML = "";
  if (!metrics) return;

  const chips = [
    { label: "Runs", value: metrics.runCount },
    { label: "Total tokens", value: metrics.totalTokens },
    { label: "Prompt tokens", value: metrics.totalPromptTokens },
    { label: "Completion tokens", value: metrics.totalCompletionTokens },
    { label: "Latency", value: formatMs(metrics.totalDurationMs) },
    { label: "Accuracy", value: metrics.runCount ? `${Math.round(metrics.accuracy * 100)}%` : "n/a" }
  ];

  if (metrics.totalCost != null && metrics.totalCost > 0) {
    chips.push({ label: "Est. cost", value: `$${metrics.totalCost.toFixed(4)}` });
  }

  for (const chip of chips) {
    const span = document.createElement("div");
    span.className = "metric-chip";
    span.innerHTML = `<span>${escapeHtml(chip.label)}</span><span>${escapeHtml(String(chip.value))}</span>`;
    elements.metricsGrid.appendChild(span);
  }
}

function clearMetrics() {
  state.lastMetrics = null;
  elements.metricsGrid.innerHTML = "";
}

function addLogEntry({ title, bodyHtml, metrics, timestamp, isCorrect, isError }) {
  const template = elements.logEntryTemplate.content.cloneNode(true);
  const article = template.querySelector(".log-entry");
  if (isError) {
    article.classList.add("error");
  }
  article.querySelector(".log-title").textContent = title;
  article.querySelector(".log-timestamp").textContent = timestamp.toLocaleTimeString();
  article.querySelector(".log-body").innerHTML = bodyHtml;
  article.querySelector(".log-metrics").innerHTML = metrics.map(item => `<span>${item}</span>`).join("");

  if (isCorrect) {
    article.classList.add("correct");
  }

  elements.logStream.prepend(article);

  state.logHistory.unshift({
    node: article,
    data: { title, bodyHtml, metrics, timestamp, isCorrect, isError }
  });
}


function buildFewShotPrompt(question, shots) {
  const baseline = `You are answering a multiple-choice question from the MMLU benchmark. Learn from the examples, then answer the new question. ${jsonOutputReminder()}`;
  const exampleBlocks = shots.map((item, idx) => {
    return `Example ${idx + 1}:\n${formatQuestionBlock(item)}\nAnswer: ${item.answer}`;
  });
  const newQuestion = `New question:\n${formatQuestionBlock(question)}\nReturn JSON.`;
  return [baseline, ...exampleBlocks, newQuestion].join("\n\n");
}

function buildInstructionPrompt(question) {
  return `Task: provide the best answer for the following MMLU question. Offer brief rationale and confidence (0-1). ${jsonOutputReminder()}\n\n${formatQuestionBlock(question)}`;
}

function buildCotPrompt(question) {
  return `Consider the following MMLU question. Reason step by step before committing to the final answer. ${jsonOutputReminder()}\n\n${formatQuestionBlock(question)}\n\nRespond in JSON.`;
}

function buildReflectionPrompt(question, draft) {
  const draftAnswer = draft?.final_answer || draft?.answer_letter || "unknown";
  const draftReasoning = draft?.reasoning || draft?.thought_process || "No reasoning provided.";
  return `You previously answered ${draftAnswer}. Review this reasoning and improve it if needed. ${jsonOutputReminder()}\n\nOriginal reasoning:\n${draftReasoning}\n\nRe-evaluate the question:\n${formatQuestionBlock(question)}\n\nRespond with refined reasoning, updated confidence, and final answer in JSON.`;
}

function buildSystemPrompt() {
  return `You are an elite exam taker focused on factual accuracy. ${jsonOutputReminder()}`;
}

function jsonOutputReminder() {
  return "Always respond with strict JSON using keys: reasoning, final_answer, answer_letter, confidence.";
}

function formatQuestionBlock(item) {
  const lines = [item.question];
  for (const [letter, choice] of Object.entries(item.choices)) {
    lines.push(`${letter}. ${choice}`);
  }
  return lines.join("\n");
}

function parseModelPayload(content) {
  if (!content) return {};
  const maybeJson = extractJson(content);
  if (!maybeJson) {
    return { reasoning: content };
  }
  try {
    const parsed = JSON.parse(maybeJson);
    if (typeof parsed === "object" && parsed !== null) {
      return parsed;
    }
    return { reasoning: content };
  } catch (error) {
    return { reasoning: content };
  }
}

function extractJson(text) {
  const first = text.indexOf("{");
  const last = text.lastIndexOf("}");
  if (first === -1 || last === -1 || last <= first) return null;
  return text.slice(first, last + 1);
}

function inferLetterFromText(text) {
  if (!text) return null;
  const match = text.match(/\b([A-D])\b/);
  return match ? match[1] : null;
}

function pickRandom(pool, count) {
  const copy = [...pool];
  const picks = [];
  while (copy.length && picks.length < count) {
    const index = Math.floor(Math.random() * copy.length);
    picks.push(copy.splice(index, 1)[0]);
  }
  return picks;
}

function truncate(text, length) {
  if (text.length <= length) return text;
  return `${text.slice(0, length - 1)}…`;
}

function formatReasoningHtml(raw) {
  if (!raw) {
    return `<p>${escapeHtml("No reasoning provided.")}</p>`;
  }

  const normalized = raw.replace(/\r\n/g, "\n").trim();
  if (!normalized) {
    return `<p>${escapeHtml("No reasoning provided.")}</p>`;
  }

  const blocks = normalized.split(/\n{2,}/).map(block => block.trim()).filter(Boolean);
  const segments = blocks.length ? blocks : [normalized];

  const html = segments
    .map(segment => {
      const lines = segment.split(/\n/).map(line => line.trim()).filter(Boolean);
      if (!lines.length) {
        return "";
      }

      const ordered = lines.every(line => /^\d+[\.)]/.test(line));
      const bulleted = lines.every(line => /^[-*•]/.test(line));

      if (ordered) {
        const items = lines.map(line => line.replace(/^\d+[\.)]\s*/, ""));
        return `<ol>${items.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ol>`;
      }

      if (bulleted) {
        const items = lines.map(line => line.replace(/^[-*•]\s*/, ""));
        return `<ul>${items.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
      }

      return `<p>${escapeHtml(lines.join(" "))}</p>`;
    })
    .filter(Boolean)
    .join("");

  return html || `<p>${escapeHtml(normalized)}</p>`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatMs(value) {
  if (value == null) return "n/a";
  if (value < 1000) return `${value.toFixed(0)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

function estimateCost(model, promptTokens, completionTokens) {
  const pricing = state.priceTable[model];
  if (!pricing || (pricing.input === 0 && pricing.output === 0)) return null;
  const promptCost = (promptTokens / 1000) * pricing.input;
  const completionCost = (completionTokens / 1000) * pricing.output;
  return promptCost + completionCost;
}
