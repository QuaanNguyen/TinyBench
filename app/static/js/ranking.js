(function () {
  "use strict";

  var stats = document.getElementById("ranking-stats");
  var table = document.getElementById("ranking-table");
  var tbody = document.getElementById("ranking-tbody");
  var empty = document.getElementById("ranking-empty");

  function formatContext(n) {
    if (!n) return "--";
    if (n >= 1000000) return (n / 1000000).toFixed(0) + "M";
    if (n >= 1000) return (n / 1000).toFixed(0) + "K";
    return String(n);
  }

  function formatTps(tps) {
    if (!tps) return "--";
    return tps.toFixed(1) + " t/s";
  }

  function formatDate(iso) {
    if (!iso) return "--";
    var d = new Date(iso + "Z");
    return d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  }

  async function load() {
    try {
      var res = await fetch("/api/ranking");
      if (!res.ok) return;
      var data = await res.json();

      var parts = [];
      if (data.updated_at) parts.push(formatDate(data.updated_at));
      parts.push(data.total_votes + " votes");
      parts.push(data.total_models + " models");
      stats.textContent = parts.join("  ·  ");

      if (data.models.length === 0) {
        empty.hidden = false;
        table.hidden = true;
        return;
      }

      empty.hidden = true;
      table.hidden = false;
      tbody.innerHTML = "";

      data.models.forEach(function (m) {
        var tr = document.createElement("tr");

        var tdRank = document.createElement("td");
        tdRank.className = "ranking-rank";
        tdRank.textContent = m.rank;
        tr.appendChild(tdRank);

        var tdModel = document.createElement("td");
        tdModel.className = "ranking-model";
        var nameSpan = document.createElement("span");
        nameSpan.className = "ranking-model-name";
        nameSpan.textContent = m.model_id;
        tdModel.appendChild(nameSpan);
        tr.appendChild(tdModel);

        var tdScore = document.createElement("td");
        tdScore.className = "ranking-numeric";
        tdScore.textContent = m.score;
        if (m.ci) {
          var ciSpan = document.createElement("span");
          ciSpan.className = "ranking-ci";
          ciSpan.textContent = " \u00b1" + m.ci;
          tdScore.appendChild(ciSpan);
        }
        tr.appendChild(tdScore);

        var tdPower = document.createElement("td");
        tdPower.className = "ranking-numeric ranking-muted";
        tdPower.textContent = "--";
        tr.appendChild(tdPower);

        var tdVotes = document.createElement("td");
        tdVotes.className = "ranking-numeric";
        tdVotes.textContent = m.votes.toLocaleString();
        tr.appendChild(tdVotes);

        var tdCtx = document.createElement("td");
        tdCtx.className = "ranking-numeric";
        tdCtx.textContent = formatContext(m.context_length);
        tr.appendChild(tdCtx);

        var tdTps = document.createElement("td");
        tdTps.className = "ranking-numeric";
        tdTps.textContent = formatTps(m.throughput_tps);
        tr.appendChild(tdTps);

        tbody.appendChild(tr);
      });
    } catch (_) {
      /* network error — leave page empty */
    }
  }

  document.querySelector("main").classList.add("ranking-active");
  load();
})();
