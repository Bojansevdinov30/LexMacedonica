/* Courtroom simulation — секој потег се СТРИМУВА (NDJSON):
серверот испраќа meta → token* → final*/

const form = document.getElementById("sim-form");
const scenarioEl = document.getElementById("sim-scenario");
const startBtn = document.getElementById("sim-start");
const courtroom = document.getElementById("courtroom");

function addTurn(turn) {
    const div = document.createElement("div");
    div.className = `sim-turn sim-${turn.role}`;

    const role = document.createElement("div");
    role.className = "sim-role";
    role.textContent = `${turn.icon} ${turn.name}`;
    div.appendChild(role);

    const text = document.createElement("div");
    text.className = "sim-text";
    text.textContent = turn.text || "";
    div.appendChild(text);

    courtroom.appendChild(div);
    div.scrollIntoView({behavior: "smooth", block: "end"});
    return div;
}

function addStatus(text) {
    const div = document.createElement("div");
    div.className = "sim-status";
    div.textContent = text;
    courtroom.appendChild(div);
    return div;
}

/* Еден потег: чита NDJSON стрим линија по линија и ја гради пораката
во живо. Враќа final-настан (или null ако стримот бил празен).
statusEl се трга штом учесникот почне да зборува (meta-настан).*/
async function streamTurn(scenario, history, statusEl) {
    const res = await fetch("/api/simulate/turn", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({scenario, history}),
    });
    if (!res.ok) {   // пр. 429 од rate-limitot — стигнува ПРЕД стримот
        const err = new Error(`HTTP ${res.status}`);
        try {
            const data = await res.json();
            if (data.detail) err.userMessage = data.detail;
        } catch (_) { /* нема JSON тело */
        }
        throw err;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let turnDiv = null, textDiv = null, final = null;

    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buf += decoder.decode(value, {stream: true});

        // транспортот се сече САМО на цели линии — \n внатре во JSON
        // стринговите се escape-ирани од серверот и не смета
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, nl).trim();
            buf = buf.slice(nl + 1);
            if (!line) continue;
            const ev = JSON.parse(line);

            if (ev.type === "meta") {
                statusEl.remove();
                turnDiv = addTurn({role: ev.role, name: ev.name, icon: ev.icon, text: ""});
                textDiv = turnDiv.querySelector(".sim-text");
            } else if (ev.type === "token" && textDiv) {
                textDiv.textContent += ev.text;
                turnDiv.scrollIntoView({block: "end"});
            } else if (ev.type === "final") {
                final = ev;
                if (textDiv) textDiv.textContent = ev.text;  // нормализиран целосен текст
            } else if (ev.type === "error") {
                const err = new Error(ev.detail);
                err.userMessage = ev.detail;
                throw err;
            }
        }
    }
    return final;
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
            const turn = await streamTurn(scenario, history, status);
            status.remove();   // но-оп ако веќе е тргнат од meta-настанот

            if (!turn || !turn.text) break;
            history.push({role: turn.role, text: turn.text});
            if (turn.done) break;
        }
        addStatus("— Судењето заврши —");
    } catch (err) {
        addStatus(err.userMessage || "Грешка при симулацијата. Обидете се повторно.");
        console.error(err);
    } finally {
        startBtn.disabled = false;
    }
});
