// Courtroom simulation: requests one turn at a time from the backend so the
// trial unfolds live, message by message.

const form = document.getElementById("sim-form");
const scenarioEl = document.getElementById("sim-scenario");
const startBtn = document.getElementById("sim-start");
const courtroom = document.getElementById("courtroom");

function addTurn(turn) {
    const div = document.createElement("div");
    div.className = `sim-turn sim-${turn.role}`;
    div.innerHTML = `
        <div class="sim-role">${turn.icon} ${turn.name}</div>
        <div class="sim-text"></div>`;
    div.querySelector(".sim-text").textContent = turn.text;
    courtroom.appendChild(div);
    div.scrollIntoView({ behavior: "smooth", block: "end" });
}

function addStatus(text) {
    const div = document.createElement("div");
    div.className = "sim-status";
    div.textContent = text;
    courtroom.appendChild(div);
    return div;
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const scenario = scenarioEl.value.trim();
    if (!scenario) return;

    courtroom.innerHTML = "";
    startBtn.disabled = true;
    const history = [];

    try {
        while (true) {
            const status = addStatus("… се подготвува следниот учесник …");
            const res = await fetch("/api/simulate/turn", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ scenario, history }),
            });
            const turn = await res.json();
            status.remove();

            if (!turn.text) break;
            addTurn(turn);
            history.push({ role: turn.role, text: turn.text });
            if (turn.done) break;
        }
        addStatus("— Судењето заврши —");
    } catch (err) {
        addStatus("Грешка при симулацијата. Обидете се повторно.");
        console.error(err);
    } finally {
        startBtn.disabled = false;
    }
});
