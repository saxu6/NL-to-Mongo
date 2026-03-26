const API_URL = "/api/v1/chat";
const HEALTH_URL = "/health";
const MAX_TEXTAREA_HEIGHT = 120;

const ui = {
    chatArea: document.getElementById("chatArea"),
    queryInput: document.getElementById("queryInput"),
    sendBtn: document.getElementById("sendBtn"),
    statusDot: document.getElementById("statusDot"),
    statusText: document.getElementById("statusText"),
    welcome: document.getElementById("welcome"),
};

let isProcessing = false;

init();

function init() {
    ui.queryInput.addEventListener("input", handleInput);
    ui.queryInput.addEventListener("keydown", handleKeyDown);
    checkHealth();
}

function handleInput() {
    autoResize(ui.queryInput);
    ui.sendBtn.disabled = !ui.queryInput.value.trim() || isProcessing;
}

function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

async function checkHealth() {
    try {
        const response = await fetch(HEALTH_URL);
        setStatus(response.ok);
    } catch {
        setStatus(false);
    }
}

function setStatus(connected) {
    ui.statusDot.className = `status-dot ${connected ? "connected" : "error"}`;
    ui.statusText.textContent = connected ? "Connected" : "Offline";
}

async function sendMessage() {
    const text = ui.queryInput.value.trim();
    if (!text || isProcessing) return;

    isProcessing = true;
    ui.sendBtn.disabled = true;
    hideWelcome();
    addMessage(text, "user");
    resetInput();

    const typingEl = addTypingIndicator();

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text }),
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        addBotResponse(data);
    } catch {
        addMessage("Connection error. Is the server running?", "bot");
    } finally {
        typingEl.remove();
        isProcessing = false;
        ui.sendBtn.disabled = !ui.queryInput.value.trim();
    }
}

function useSuggestion(button) {
    ui.queryInput.value = button.textContent || "";
    handleInput();
    ui.queryInput.focus();
}

function hideWelcome() {
    if (ui.welcome) ui.welcome.style.display = "none";
}

function resetInput() {
    ui.queryInput.value = "";
    autoResize(ui.queryInput);
}

function addMessage(text, role) {
    const msg = createEl("div", `message ${role}`);
    const avatar = createEl("div", "message-avatar", role === "bot" ? "N" : "U");
    const content = createEl("div", "message-content");
    const bubble = createEl("div", "message-bubble", text);

    content.appendChild(bubble);
    msg.appendChild(avatar);
    msg.appendChild(content);
    ui.chatArea.appendChild(msg);
    scrollToBottom();
}

function addBotResponse(data) {
    const msg = createEl("div", "message bot");
    const avatar = createEl("div", "message-avatar", "N");
    const content = createEl("div", "message-content");
    const bubble = createEl("div", "message-bubble", data.reply || "Done.");

    content.appendChild(bubble);
    appendMeta(content, data);

    if (data.query) appendQueryPreview(content, data.query);
    if (Array.isArray(data.results) && data.results.length > 0) {
        content.appendChild(buildResultsTable(data.results));
    }

    msg.appendChild(avatar);
    msg.appendChild(content);
    ui.chatArea.appendChild(msg);
    scrollToBottom();
}

function appendMeta(container, data) {
    if (!data.query && data.confidence == null && (!data.warnings || data.warnings.length === 0)) {
        return;
    }

    const row = createEl("div", "meta-row");

    if (data.query?.collection) {
        row.appendChild(makeChip(data.query.collection, "collection"));
    }

    if (data.confidence != null) {
        const conf = Number(data.confidence);
        const confClass = conf >= 0.7 ? "confidence-high" : conf >= 0.4 ? "confidence-mid" : "confidence-low";
        row.appendChild(makeChip(`${Math.round(conf * 100)}% conf`, confClass));
    }

    if (Array.isArray(data.warnings)) {
        data.warnings.forEach((warning) => row.appendChild(makeChip(warning, "warning")));
    }

    container.appendChild(row);
}

function appendQueryPreview(container, query) {
    const toggleBtn = createEl("button", "query-toggle", "{ } Show Query");
    const block = createEl("pre", "query-block");
    block.textContent = JSON.stringify(buildQueryDisplay(query), null, 2);

    toggleBtn.addEventListener("click", () => {
        const visible = block.classList.toggle("visible");
        toggleBtn.textContent = visible ? "{ } Hide Query" : "{ } Show Query";
    });

    container.appendChild(toggleBtn);
    container.appendChild(block);
}

function buildQueryDisplay(query) {
    const display = {};
    if (query.collection) display.collection = query.collection;
    if (query.operation) display.operation = query.operation;

    const optionalMaps = ["filter", "projection", "sort", "update"];
    optionalMaps.forEach((key) => {
        if (query[key] && Object.keys(query[key]).length > 0) display[key] = query[key];
    });

    if (Array.isArray(query.pipeline) && query.pipeline.length > 0) display.pipeline = query.pipeline;
    if (query.limit != null) display.limit = query.limit;
    return display;
}

function buildResultsTable(results) {
    const wrapper = createEl("div", "results-wrapper");
    const scroll = createEl("div", "results-scroll");
    const table = createEl("table", "results-table");

    const columns = getColumns(results);
    const thead = createEl("thead");
    const headRow = createEl("tr");
    columns.forEach((col) => headRow.appendChild(createEl("th", "", col)));
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = createEl("tbody");
    results.forEach((doc) => {
        const row = createEl("tr");
        columns.forEach((col) => {
            const value = doc[col];
            const td = createEl("td", "", formatCellValue(value));
            td.title = typeof value === "object" ? JSON.stringify(value, null, 2) : String(value ?? "");
            row.appendChild(td);
        });
        tbody.appendChild(row);
    });

    table.appendChild(tbody);
    scroll.appendChild(table);
    wrapper.appendChild(scroll);
    return wrapper;
}

function getColumns(results) {
    const keys = new Set();
    results.forEach((doc) => Object.keys(doc).forEach((k) => keys.add(k)));
    return Array.from(keys).sort((a, b) => {
        if (a === "_id") return 1;
        if (b === "_id") return -1;
        return a.localeCompare(b);
    });
}

function formatCellValue(value) {
    if (value == null) return "-";
    if (typeof value === "object") {
        const text = JSON.stringify(value);
        return text.length > 60 ? `${text.slice(0, 57)}...` : text;
    }
    const text = String(value);
    return text.length > 60 ? `${text.slice(0, 57)}...` : text;
}

function makeChip(text, className) {
    return createEl("span", `chip ${className}`, text);
}

function addTypingIndicator() {
    const msg = createEl("div", "message bot");
    const avatar = createEl("div", "message-avatar", "N");
    const content = createEl("div", "message-content");
    const bubble = createEl("div", "message-bubble");
    const typing = createEl("div", "typing");

    typing.appendChild(createEl("span", "typing-dot"));
    typing.appendChild(createEl("span", "typing-dot"));
    typing.appendChild(createEl("span", "typing-dot"));
    bubble.appendChild(typing);
    content.appendChild(bubble);
    msg.appendChild(avatar);
    msg.appendChild(content);
    ui.chatArea.appendChild(msg);
    scrollToBottom();
    return msg;
}

function autoResize(el) {
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`;
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        ui.chatArea.scrollTop = ui.chatArea.scrollHeight;
    });
}

function createEl(tag, className = "", text = "") {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text) el.textContent = text;
    return el;
}
