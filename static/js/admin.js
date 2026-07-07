// Anonymization tool: paste text -> anonymized text + replacement table + reasoning.

const form = document.getElementById("anon-form");
const input = document.getElementById("anon-input");
const btn = document.getElementById("anon-btn");
const resultBox = document.getElementById("anon-result");
const output = document.getElementById("anon-output");
const replacementsEl = document.getElementById("anon-replacements");
const reasoningEl = document.getElementById("anon-reasoning");
const reasoningBox = document.getElementById("anon-reasoning-box");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    btn.disabled = true;
    btn.textContent = "Анонимизирам…";

    try {
        const res = await fetch("/api/anonymize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
        const data = await res.json();

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
            replacementsEl.innerHTML =
                "<tr><td colspan='3'>Не се пронајдени лични податоци.</td></tr>";
        }
        resultBox.style.display = "";
    } catch (err) {
        alert("Грешка при анонимизацијата.");
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.textContent = "Анонимизирај";
    }
});

document.getElementById("anon-copy").addEventListener("click", () => {
    navigator.clipboard.writeText(output.textContent);
});
