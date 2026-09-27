const form = document.querySelector("#question-form");
const input = document.querySelector("#question");
const conversation = document.querySelector("#conversation");
const activity = document.querySelector("#activity");
let freshSession = false;
let busy = false;
let history = [];

function saveConversation() {
  try {
    sessionStorage.setItem(
      "policy-session",
      JSON.stringify({ history, freshSession }),
    );
  } catch {
    // Storage is optional when the browser disables it.
  }
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function showContext(result) {
  const scope = document.querySelector("#scope");
  scope.replaceChildren();
  for (const [country, year] of result.scopes || []) {
    const line = element("div", undefined, "scope-line");
    line.append(element("span", country), element("span", year));
    scope.append(line);
  }
  if (!scope.childNodes.length) {
    scope.append(
      element(
        "p",
        [result.countries?.join(", "), result.requested_years?.join(", ")]
          .filter(Boolean)
          .join(" / ") || "Awaiting country and policy year.",
        "muted",
      ),
    );
  }
  const sources = document.querySelector("#sources");
  sources.replaceChildren();
  const evidence = result.retrieved_evidence || [];
  for (const source of evidence) {
    const details = element("details", undefined, "source");
    details.id = `source-${source.citation_id}`;
    details.append(
      element("summary", `${source.document_name}, p. ${source.page_number}`),
    );
    if (source.section_title)
      details.append(element("p", source.section_title, "muted"));
    details.append(element("blockquote", source.text));
    const link = element("a", "Open PDF page");
    link.href = `/documents/${encodeURIComponent(source.document_id)}#page=${source.page_number}`;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    details.append(link);
    sources.append(details);
  }
  if (!evidence.length)
    sources.append(element("p", "No sources for this reply.", "muted"));
  const diagnostic = document.querySelector("#diagnostic-content");
  diagnostic.replaceChildren();
  const data = result.diagnostics || {};
  const list = element("dl");
  for (const [key, value] of Object.entries(data)) {
    if (typeof value === "number")
      list.append(
        element("dt", key.replaceAll("_", " ")),
        element("dd", value),
      );
  }
  for (const [stage, milliseconds] of Object.entries(data.timing_ms || {})) {
    list.append(
      element("dt", `${stage} time`),
      element("dd", `${(milliseconds / 1000).toFixed(2)} s`),
    );
  }
  if (data.validation)
    list.append(
      element("dt", "Validation"),
      element("dd", data.validation.replaceAll("_", " ")),
    );
  diagnostic.append(list);
}

function renderTurn(question, result) {
  const turn = element("article", undefined, "turn");
  turn.append(element("h2", question));
  const statuses = {
    answered: "Sources checked",
    needs_clarification: "Needs clarification",
    insufficient_evidence: "Insufficient evidence",
  };
  turn.append(
    element(
      "span",
      statuses[result.status] || result.status,
      `status ${result.status}`,
    ),
  );
  const answer = element("div", undefined, "answer");
  const cited = new Map(
    (result.citations || []).map((source) => [source.citation_id, source]),
  );
  for (const paragraph of result.answer.split("\n\n")) {
    const line = element("p");
    for (const part of paragraph.split(/(\[S\d+\])/g)) {
      const id = part.slice(1, -1);
      if (cited.has(id)) {
        const source = cited.get(id);
        const button = element(
          "button",
          `[${source.document_name}, p. ${source.page_number}]`,
          "citation-link",
        );
        button.type = "button";
        button.addEventListener("click", () => {
          showContext(result);
          const details = document.getElementById(`source-${id}`);
          details.open = true;
          details.scrollIntoView({ block: "nearest" });
          details.querySelector("summary").focus();
        });
        line.append(button);
      } else line.append(document.createTextNode(part));
    }
    answer.append(line);
  }
  turn.append(answer);
  conversation.append(turn);
  showContext(result);
  document.querySelector("#welcome").hidden = true;
}

for (const button of document.querySelectorAll(".example")) {
  button.addEventListener("click", () => {
    input.value = button.dataset.question;
    input.focus();
  });
}

document.querySelector("#new-session").addEventListener("click", () => {
  if (busy) return;
  freshSession = true;
  history = [];
  saveConversation();
  conversation.replaceChildren();
  document.querySelector("#welcome").hidden = false;
  showContext({});
  input.value = "";
  activity.textContent = "Conversation cleared.";
  input.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy || !input.value.trim()) return;
  busy = true;
  document.querySelector("#submit").disabled = true;
  document.querySelector("#new-session").disabled = true;
  form.setAttribute("aria-busy", "true");
  activity.textContent =
    "Searching and checking sources. This may take a minute.";
  const question = input.value.trim();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, new_session: freshSession }),
    });
    const result = await response.json();
    if (!response.ok)
      throw new Error(
        typeof result.detail === "string"
          ? result.detail
          : "Please enter a question of at most 2,000 characters.",
      );
    freshSession = false;
    history.push({ question, result });
    if (history.length > 20) history.shift();
    renderTurn(question, result);
    saveConversation();
    input.value = "";
    activity.textContent =
      result.status === "needs_clarification"
        ? "Your question is saved. Enter the missing detail."
        : "Ready for your next question.";
    input.focus();
  } catch (error) {
    activity.textContent = error.message;
  } finally {
    busy = false;
    document.querySelector("#submit").disabled = false;
    document.querySelector("#new-session").disabled = false;
    form.removeAttribute("aria-busy");
  }
});

try {
  const saved = JSON.parse(sessionStorage.getItem("policy-session") || "{}");
  history = saved.history || [];
  freshSession = saved.freshSession || false;
  history.forEach(({ question, result }) => renderTurn(question, result));
} catch {
  history = [];
}
