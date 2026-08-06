/* Lawyer assistant: law-first answers with visible reasoning (collapsible).*/

const win = document.getElementById("lawyer-window");
const form = document.getElementById("lawyer-form");
const textarea = document.getElementById("lawyer-text");
const sendBtn = document.getElementById("lawyer-send");

function renderAnswer(data) {
    const div = document.createElement("div");
    div.className = "msg msg-bot";

    if (data.reasoning) {
        const details = document.createElement("details");
        details.className = "reasoning";
        const summary = document.createElement("summary");
        summary.textContent = "🧠 Резонирање на моделот";
        details.appendChild(summary);
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
        const label = document.createElement("strong");
        label.textContent = "Извори:";
        src.appendChild(label);
        for (const s of data.sources) {
            src.appendChild(document.createTextNode(" "));
            const chip = document.createElement("span");
            chip.className = "source-chip";
            chip.textContent = `${s.type === "закон" ? "📜" : "⚖️"} ${s.ref}`;
            src.appendChild(chip);
        }
        div.appendChild(src);
    }

    win.appendChild(div);
    win.scrollTop = win.scrollHeight;
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = textarea.value.trim();
    if (!question) return;

    addMessage(win, question, "user");
    textarea.value = "";
    sendBtn.disabled = true;
    const thinking = addMessage(win, "Пребарувам закони и пракса…", "bot");
    thinking.classList.add("msg-thinking");

    try {
        const data = await postJSON("/api/lawyer", { question });
        thinking.remove();
        renderAnswer(data);
    } catch (err) {
        thinking.remove();
        showError(win, err);
    } finally {
        sendBtn.disabled = false;
        textarea.focus();
    }
});

attachEnterSubmit(textarea, form);
