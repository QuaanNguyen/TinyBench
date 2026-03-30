(function () {
  "use strict";

  const landing = document.getElementById("fight-landing");
  const arena = document.getElementById("fight-arena");
  const fightInput = document.getElementById("fight-input");
  const fightSend = document.getElementById("fight-send");
  const promptText = document.getElementById("fight-prompt-text");

  const outputA = document.getElementById("output-a");
  const outputB = document.getElementById("output-b");
  const voteA = document.getElementById("vote-a");
  const voteB = document.getElementById("vote-b");
  const labelA = document.getElementById("label-a");
  const labelB = document.getElementById("label-b");
  const paneA = document.getElementById("pane-a");
  const paneB = document.getElementById("pane-b");

  const FLUSH_INTERVAL_MS = 50;

  let activeFightId = null;
  let doneCount = 0;
  let voted = false;

  function switchToArena(prompt) {
    landing.hidden = true;
    arena.hidden = false;
    document.querySelector("main").classList.add("fight-active");
    promptText.textContent = prompt;
  }

  function resetArena() {
    outputA.textContent = "";
    outputB.textContent = "";
    voteA.disabled = true;
    voteB.disabled = true;
    voteA.textContent = "Pick A";
    voteB.textContent = "Pick B";
    labelA.textContent = "Model A";
    labelB.textContent = "Model B";
    paneA.classList.remove("winner", "loser");
    paneB.classList.remove("winner", "loser");
    doneCount = 0;
    voted = false;
  }

  function enableVoting() {
    if (voted) return;
    voteA.disabled = false;
    voteB.disabled = false;
  }

  function renderMarkdown(el) {
    var raw = el.textContent;
    if (typeof marked !== "undefined" && raw) {
      el.innerHTML = marked.parse(raw);
      el.classList.add("md-rendered");
    }
  }

  function streamToPane(jobId, outputEl, onDone) {
    let buffer = "";
    let flushTimer = null;

    function flush() {
      if (buffer) {
        outputEl.textContent += buffer;
        buffer = "";
        outputEl.scrollTop = outputEl.scrollHeight;
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
      renderMarkdown(outputEl);
      outputEl.scrollTop = outputEl.scrollHeight;
      if (source) {
        source.close();
      }
      onDone();
    }

    const source = new EventSource("/sse/chat/" + jobId);

    source.addEventListener("token", function (e) {
      buffer += e.data;
      startFlushTimer();
    });

    source.addEventListener("done", function () {
      cleanup();
    });

    source.addEventListener("error", function (e) {
      if (e.data) {
        buffer += "\n[Error: " + e.data + "]";
      }
      cleanup();
    });

    source.onerror = function () {
      cleanup();
    };
  }

  async function startFight() {
    var text = fightInput.value.trim();
    if (!text) return;

    resetArena();
    switchToArena(text);
    fightInput.value = "";

    try {
      var res = await fetch("/api/fight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text }),
      });
      if (!res.ok) {
        var err = await res.json().catch(function () {
          return { detail: "Request failed" };
        });
        outputA.textContent = "[Error: " + (err.detail || "Request failed") + "]";
        return;
      }
      var data = await res.json();
      activeFightId = data.fight_id;

      var jobA = data.jobs[0].job_id;
      var jobB = data.jobs[1].job_id;

      function onStreamDone() {
        doneCount++;
        if (doneCount >= 2) {
          enableVoting();
        }
      }

      streamToPane(jobA, outputA, onStreamDone);
      streamToPane(jobB, outputB, onStreamDone);
    } catch (_) {
      outputA.textContent = "[Network error]";
    }
  }

  async function castVote(winner) {
    if (voted || !activeFightId) return;
    voted = true;
    voteA.disabled = true;
    voteB.disabled = true;

    try {
      var res = await fetch("/api/fight/" + activeFightId + "/vote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ winner: winner }),
      });
      if (res.ok) {
        var result = await res.json();
        labelA.textContent = result.model_a;
        labelB.textContent = result.model_b;

        if (winner === "A") {
          paneA.classList.add("winner");
          paneB.classList.add("loser");
          voteA.textContent = "Winner";
        } else {
          paneB.classList.add("winner");
          paneA.classList.add("loser");
          voteB.textContent = "Winner";
        }
      }
    } catch (_) {
      // vote failed silently
    }
  }

  fightInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      startFight();
    }
  });

  fightSend.addEventListener("click", function () {
    startFight();
  });

  voteA.addEventListener("click", function () {
    castVote("A");
  });

  voteB.addEventListener("click", function () {
    castVote("B");
  });
})();
