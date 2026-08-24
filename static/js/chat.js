/* Main chat (Правен асистент).
Flow: user submits → POST /api/chat → render answer + case cards + probability bar. */

const chatWindow = document.getElementById("chat-window");
const form = document.getElementById("chat-form");
const textarea = document.getElementById("chat-text");
const sendBtn = document.getElementById("chat-send");

/* Refreshing the page starts a fresh chat.*/
const history = [];

// One similar-case citation card, built with textContent only (XSS-safe).
function makeCaseCard(c) {
    const card = document.createElement("div");
    card.className = "case-card";

    const title = document.createElement("div");
    title.className = "case-title";
    title.textContent = c.case_number;
    card.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "case-meta";
    meta.textContent = `${c.court} · ${c.date} · исход: ${c.outcome}`;
    card.appendChild(meta);

    const summary = document.createElement("div");
    summary.textContent = c.summary || "";
    card.appendChild(summary);

    // «Повеќе» = full decision text, fetched on first click.
    if (c.case_id) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "case-more-btn";
        btn.textContent = "Повеќе";

        const full = document.createElement("div");
        full.className = "case-fulltext";
        full.style.display = "none";

        let loaded = false;
        btn.addEventListener("click", async () => {
            if (full.style.display !== "none") {
                full.style.display = "none";
                btn.textContent = "Повеќе";
                return;
            }
            if (!loaded) {
                btn.textContent = "Вчитувам…";
                try {
                    const res = await fetch(`/api/case/${encodeURIComponent(c.case_id)}/text`);
                    if (!res.ok) {
                        throw new Error(`HTTP ${res.status}`);
                    }
                    const data = await res.json();
                    full.textContent = data.text;
                    loaded = true;
                } catch (err) {
                    full.textContent = "Не можев да го вчитам целиот текст на одлуката.";
                    console.error(err);
                }
            }
            full.style.display = "";
            btn.textContent = "Помалку";
        });

        card.appendChild(btn);
        card.appendChild(full);
    }
    return card;
}

// Renders the structured answer from the backend
function renderAnswer(data) {
    const div = document.createElement("div");
    div.className = "msg msg-bot";

    const p = document.createElement("p");
    p.textContent = data.answer;
    div.appendChild(p);

    if (data.probability !== null && data.probability !== undefined) {
        const label = document.createElement("p");
        label.style.marginTop = "0.6rem";
        const strong = document.createElement("strong");
        strong.textContent = `Веројатност: ${data.probability}%`;
        label.appendChild(strong);
        div.appendChild(label);

        const track = document.createElement("div");
        track.className = "prob-bar-track";
        const fill = document.createElement("div");
        fill.className = "prob-bar-fill";
        fill.style.width = "0%";
        track.appendChild(fill);
        div.appendChild(track);
        // small delay so the CSS width transition animates
        setTimeout(() => {
            fill.style.width = data.probability + "%";
        }, 50);
    }

    for (const c of data.cases || []) {
        div.appendChild(makeCaseCard(c));
    }

    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = textarea.value.trim();
    if (!question) return;

    addMessage(chatWindow, question, "user");
    textarea.value = "";
    sendBtn.disabled = true;
    const thinking = addMessage(chatWindow, "Пребарувам слични случаи", "bot");
    thinking.classList.add("msg-thinking");

    try {
        const data = await postJSON("/api/chat", {question, history});
        thinking.remove();
        renderAnswer(data);
        if(data.answer !== "Невалидно поставено прашање. Обидете се повторно.") {
            history.push({who: "user", text: question});
            history.push({who: "bot", text: data.answer});
        }
    } catch (err) {
        thinking.remove();
        showError(chatWindow, err);
    } finally {
        sendBtn.disabled = false;
        textarea.focus();
    }
});

attachEnterSubmit(textarea, form);
