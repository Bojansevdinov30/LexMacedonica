// Lawyer assistant: law-first answers with visible reasoning (collapsible)
// and a mode toggle (laws+cases / cases only).

const win = document.getElementById("lawyer-window");
const form = document.getElementById("lawyer-form");
const textarea = document.getElementById("lawyer-text");
const sendBtn = document.getElementById("lawyer-send");

function addMsg(text, who) {
    const div = document.createElement("div");
    div.className = `msg msg-${who}`;
    div.textContent = text;
    win.appendChild(div);
    win.scrollTop = win.scrollHeight;
    return div;
}

function renderAnswer(data) {
    const div = document.createElement("div");
    div.className = "msg msg-bot";

    if (data.reasoning) {
        const details = document.createElement("details");
        details.className = "reasoning";
        details.innerHTML = "<summary>🧠 Резонирање на моделот</summary>";
        const body = document.createElement("div");
        body.className = "reasoning-body";
        body.textContent = data.reasoning;
        details.appendChild(body);
        div.appendChild(details);
    }

    const p = document.createElement("p");
    p.textContent = data.answer;
    div.appendChild(p);

    if (data.sources && data.sources.length) {
        const src = document.createElement("div");
        src.className = "sources";
        src.innerHTML = "<strong>Извори:</strong> " + data.sources
            .map(s => `<span class="source-chip">${s.type === "закон" ? "📜" : "⚖️"} ${s.ref}</span>`)
            .join(" ");
        div.appendChild(src);
    }

    win.appendChild(div);
    win.scrollTop = win.scrollHeight;
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = textarea.value.trim();
    if (!question) return;
    const mode = document.querySelector('input[name="mode"]:checked').value;

    addMsg(question, "user");
    textarea.value = "";
    sendBtn.disabled = true;
    const thinking = addMsg(mode === "laws"
        ? "Пребарувам закони и пракса…" : "Пребарувам судска пракса…", "bot");
    thinking.classList.add("msg-thinking");

    try {
        const res = await fetch("/api/lawyer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, mode }),
        });
        const data = await res.json();
        thinking.remove();
        renderAnswer(data);
    } catch (err) {
        thinking.remove();
        addMsg("Се појави грешка. Обидете се повторно.", "bot");
        console.error(err);
    } finally {
        sendBtn.disabled = false;
        textarea.focus();
    }
});

textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        form.requestSubmit();
    }
});
