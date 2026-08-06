/* Anonymization tool: paste text -> anonymized text + replacement table + reasoning.*/

const form = document.getElementById("anon-form");
const input = document.getElementById("anon-input");
const btn = document.getElementById("anon-btn");
const resultBox = document.getElementById("anon-result");
const output = document.getElementById("anon-output");
const replacementsEl = document.getElementById("anon-replacements");
const reasoningEl = document.getElementById("anon-reasoning");
const reasoningBox = document.getElementById("anon-reasoning-box");
const section = form.parentElement;   // грешките се прикажуваат под формата

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    btn.disabled = true;
    btn.textContent = "Анонимизирам…";

    try {
        const data = await postJSON("/api/anonymize", { text });

        output.textContent = data.anonymized || "(празно)";
        reasoningEl.textContent = data.reasoning || "";
        reasoningBox.style.display = data.reasoning ? "" : "none";

        replacementsEl.innerHTML = "";
        for (const r of data.replacements || []) {
            const tr = document.createElement("tr");
            for (const val of [r.original, r.replacement, r.method]) {
                const td = document.createElement("td");
                td.textContent = val;
                tr.appendChild(td);
            }
            replacementsEl.appendChild(tr);
        }
        if (!(data.replacements || []).length) {
            const tr = document.createElement("tr");
            const td = document.createElement("td");
            td.colSpan = 3;
            td.textContent = "Не се пронајдени лични податоци.";
            tr.appendChild(td);
            replacementsEl.appendChild(tr);
        }
        resultBox.style.display = "";
    } catch (err) {
        showError(section, err);
    } finally {
        btn.disabled = false;
        btn.textContent = "Анонимизирај";
    }
});

document.getElementById("anon-copy").addEventListener("click", () => {
    navigator.clipboard.writeText(output.textContent);
});
