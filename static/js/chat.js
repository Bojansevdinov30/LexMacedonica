// Main chat (Правен асистент).
// Flow: user submits → POST /api/chat → render answer + case cards + probability bar.

const chatWindow = document.getElementById("chat-window");
const form = document.getElementById("chat-form");
const textarea = document.getElementById("chat-text");
const sendBtn = document.getElementById("chat-send");

// This chat's memory lives HERE, in the browser — each request carries the
// history so the backend can understand follow-up questions. Refreshing the
// page starts a fresh chat.
const history = [];

function addMessage(text, who) {
    const div = document.createElement("div");
    div.className = `msg msg-${who}`;
    div.textContent = text;
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return div;
}

// Renders the structured answer from the backend: answer text,
// probability (0-100 or null) and the list of similar cases.
function renderAnswer(data) {
    const div = document.createElement("div");
    div.className = "msg msg-bot";

    const p = document.createElement("p");
    p.textContent = data.answer;
    div.appendChild(p);

    if (data.probability !== null && data.probability !== undefined) {
        const label = document.createElement("p");
        label.style.marginTop = "0.6rem";
        label.innerHTML = `<strong>Веројатност: ${data.probability}%</strong>`;
        div.appendChild(label);

        const track = document.createElement("div");
        track.className = "prob-bar-track";
        const fill = document.createElement("div");
        fill.className = "prob-bar-fill";
        fill.style.width = "0%";
        track.appendChild(fill);
        div.appendChild(track);
        // small delay so the CSS width transition animates
        setTimeout(() => { fill.style.width = data.probability + "%"; }, 50);
    }

    for (const c of data.cases || []) {
        const card = document.createElement("div");
        card.className = "case-card";
        card.innerHTML = `
            <div class="case-title">${c.case_number}</div>
            <div class="case-meta">${c.court} · ${c.date} · исход: ${c.outcome}</div>
            <div>${c.summary}</div>`;
        div.appendChild(card);
    }

    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = textarea.value.trim();
    if (!question) return;

    addMessage(question, "user");
    textarea.value = "";
    sendBtn.disabled = true;
    const thinking = addMessage("Пребарувам слични случаи", "bot");
    thinking.classList.add("msg-thinking");

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, history }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        thinking.remove();
        renderAnswer(data);
        history.push({ who: "user", text: question });
        history.push({ who: "bot", text: data.answer });
    } catch (err) {
        thinking.remove();
        addMessage("Се појави грешка при обработката. Обидете се повторно.", "bot");
        console.error(err);
    } finally {
        sendBtn.disabled = false;
        textarea.focus();
    }
});

// Enter = send, Shift+Enter = new line (like every chat app)
textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        form.requestSubmit();
    }
});
