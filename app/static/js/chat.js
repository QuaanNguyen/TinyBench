(function () {
  "use strict";

  const landing = document.getElementById("landing");
  const chat = document.getElementById("chat");
  const chatMessages = document.getElementById("chat-messages");

  const landingInput = document.getElementById("landing-input");
  const landingModel = document.getElementById("landing-model");

  const chatInput = document.getElementById("chat-input");
  const chatSend = document.getElementById("chat-send");
  const chatModel = document.getElementById("chat-model");

  let models = [];
  let generating = false;
  let activeSource = null;

  const FLUSH_INTERVAL_MS = 50;

  async function loadModels() {
    try {
      const res = await fetch("/api/models");
      models = await res.json();
    } catch {
      models = [];
    }
    populateSelect(landingModel);
    populateSelect(chatModel);
  }

  function populateSelect(select) {
    select.innerHTML = "";
    if (models.length === 0) {
      const opt = document.createElement("option");
      opt.textContent = "No models found";
      opt.disabled = true;
      select.appendChild(opt);
      return;
    }
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.name;
      select.appendChild(opt);
    }
  }

  function switchToChat() {
    landing.hidden = true;
    chat.hidden = false;
    document.querySelector("main").classList.add("chat-active");
    chatModel.value = landingModel.value;
    chatInput.focus();
  }

  function createBubble(text, role) {
    const wrapper = document.createElement("div");
    wrapper.className = "bubble-row " + role;

    const bubble = document.createElement("div");
    bubble.className = "bubble " + role;
    bubble.textContent = text;

    wrapper.appendChild(bubble);
    return wrapper;
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function setGenerating(val) {
    generating = val;
    chatSend.disabled = val;
  }

  async function sendMessage(inputEl, modelEl) {
    const text = inputEl.value.trim();
    if (!text || models.length === 0 || generating) return;

    if (!landing.hidden) {
      switchToChat();
    }

    chatMessages.appendChild(createBubble(text, "user"));
    scrollToBottom();
    inputEl.value = "";
    autoResize(chatInput);

    const bubbleRow = createBubble("", "assistant");
    chatMessages.appendChild(bubbleRow);
    const bubbleEl = bubbleRow.querySelector(".bubble");
    scrollToBottom();

    setGenerating(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelEl.value, prompt: text }),
      });
      if (!res.ok) {
        bubbleEl.textContent = "Error starting chat.";
        setGenerating(false);
        return;
      }
      const { job_id } = await res.json();
      streamTokens(job_id, bubbleEl);
    } catch {
      bubbleEl.textContent = "Network error.";
      setGenerating(false);
    }
  }

  function streamTokens(jobId, bubbleEl) {
    let buffer = "";
    let flushTimer = null;

    function flush() {
      if (buffer) {
        bubbleEl.textContent += buffer;
        buffer = "";
        scrollToBottom();
      }
    }

    function startFlushTimer() {
      if (!flushTimer) {
        flushTimer = setInterval(flush, FLUSH_INTERVAL_MS);
      }
    }

    function cleanup() {
      if (flushTimer) {
        clearInterval(flushTimer);
        flushTimer = null;
      }
      flush();
      if (activeSource) {
        activeSource.close();
        activeSource = null;
      }
      setGenerating(false);
    }

    const source = new EventSource("/sse/chat/" + jobId);
    activeSource = source;

    source.addEventListener("token", (e) => {
      buffer += e.data;
      startFlushTimer();
    });

    source.addEventListener("done", () => {
      cleanup();
    });

    source.addEventListener("error", (e) => {
      if (e.data) {
        buffer += "\n[Error: " + e.data + "]";
      }
      cleanup();
    });

    source.onerror = () => {
      cleanup();
    };
  }

  function handleKeydown(e, inputEl, modelEl) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputEl, modelEl);
    }
  }

  function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 150) + "px";
  }

  landingInput.addEventListener("keydown", (e) =>
    handleKeydown(e, landingInput, landingModel)
  );

  chatSend.addEventListener("click", () =>
    sendMessage(chatInput, chatModel)
  );
  chatInput.addEventListener("keydown", (e) =>
    handleKeydown(e, chatInput, chatModel)
  );
  chatInput.addEventListener("input", () => autoResize(chatInput));

  loadModels();
})();
