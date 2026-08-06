/* Заеднички помошни функции за сите табови (вчитано од base.html пред
   скриптата на секоја страница) */

// функција што го поврзува frontend-от со backend-от
async function postJSON(url, body) {
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const err = new Error(`HTTP ${res.status}`);
        try {
            const data = await res.json();
            if (data.detail) err.userMessage = data.detail;
        } catch (_) { /* телото не е JSON — остави генеричка порака */ }
        throw err;
    }
    return res.json();
}


// функција за додавање на една порака со textContent, а не innerHTML
function addMessage(container, text, who) {
    const div = document.createElement("div");
    div.className = `msg msg-${who}`;
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

// Функција за порака за грешка
function showError(container, err) {
    addMessage(container,
        err.userMessage || "Се појави грешка при обработката. Обидете се повторно.",
        "bot");
    console.error(err);
}

// Enter = испрати, Shift+Enter = нов ред (како секоја чет-апликација)
function attachEnterSubmit(textarea, form) {
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            form.requestSubmit();
        }
    });
}
