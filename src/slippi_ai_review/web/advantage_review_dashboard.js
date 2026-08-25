(function () {
  "use strict";
  const list = document.querySelector("#review-list");
  const count = document.querySelector("#review-count");
  const worker = document.querySelector("#worker-state");
  const storage = document.querySelector("#storage-state");
  const nightlyList = document.querySelector("#nightly-list");
  const nightlyCount = document.querySelector("#nightly-count");
  const search = document.querySelector("#review-search");
  const statusFilter = document.querySelector("#status-filter");
  const ageFilter = document.querySelector("#age-filter");
  let reviews = [];
  let nightlyReports = [];
  const fmtBytes = (n) => n < 1024 ** 2 ? `${Math.round(n / 1024)} KB` : n < 1024 ** 3 ? `${(n / 1024 ** 2).toFixed(1)} MB` : `${(n / 1024 ** 3).toFixed(2)} GB`;
  const fmtTime = (seconds) => seconds < 90 ? `${Math.round(seconds)} sec` : seconds < 5400 ? `${Math.round(seconds / 60)} min` : `${(seconds / 3600).toFixed(1)} hr`;
  function playerLabel(p) { const color = p.character?.colorName ? ` (${p.character.colorName})` : ""; const ids = [...new Set([p.displayName, p.nametag, p.connectCode].filter(Boolean))]; return `P${p.port} ${p.character?.name || "Unknown"}${color}${ids.length ? ` · ${ids.join(" · ")}` : ""}`; }
  function matches(review) {
    if (statusFilter.value !== "all" && review.status !== statusFilter.value) return false;
    if (ageFilter.value !== "all" && Date.now() - new Date(review.createdAt).getTime() > Number(ageFilter.value) * 86400000) return false;
    const text = `${review.originalFilename} ${(review.players || []).map(playerLabel).join(" ")} ${review.targetPlayer ? playerLabel(review.targetPlayer) : ""}`.toLowerCase();
    return text.includes(search.value.trim().toLowerCase());
  }
  async function action(review, name) {
    const destructive = name === "delete";
    if (destructive && !confirm(`Delete ${review.originalFilename} and all generated files?`)) return;
    const response = await fetch(`/api/reviews/${review.reviewId}/actions/${name}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const body = await response.json().catch(() => null);
    if (!response.ok) alert(body?.error?.message || `${name} failed`);
    await refresh();
  }
  function render() {
    const visible = reviews.filter(matches); list.replaceChildren(); count.textContent = `${visible.length} shown · ${reviews.length} total`;
    if (!visible.length) { list.innerHTML = '<div class="empty">No matching reviews.</div>'; return; }
    for (const review of visible) {
      const row = document.createElement("article"); row.className = "review-row";
      const target = review.targetPlayer ? playerLabel(review.targetPlayer) : "Choose analyzed player";
      const queue = review.queue ? `Queue #${review.queue.position} · ~${fmtTime(review.queue.estimatedSeconds)}` : (review.progress?.message || review.message || "");
      row.innerHTML = `<div class="matchup"><strong></strong><span></span><small></small></div><span class="state"></span><span class="created"></span><div class="actions"><a>Open</a><button data-action="copy">Copy</button><button data-action="log">Log</button><button data-action="cancel">Cancel</button><button data-action="retry">Retry</button><button data-action="archive">Archive</button><button data-action="delete">Delete</button></div>`;
      row.querySelector("strong").textContent = (review.players || []).map(playerLabel).join(" vs ");
      row.querySelector(".matchup span").textContent = `${target} · ${review.settings?.qualityPreset || "standard"}`;
      row.querySelector("small").textContent = `${queue}${queue ? " · " : ""}${fmtBytes(review.storageBytes || 0)}`;
      const badge = row.querySelector(".state"); badge.textContent = review.status.replace("_", " "); badge.dataset.state = review.status;
      row.querySelector(".created").textContent = new Date(review.createdAt).toLocaleString();
      const url = new URL(review.urls.review, location.href).href; row.querySelector("a").href = url;
      const allowed = { cancel: ["queued", "processing"], retry: ["failed", "cancelled", "complete", "archived"], archive: ["complete", "failed", "cancelled"], delete: ["awaiting_target", "queued", "complete", "failed", "cancelled", "archived"] };
      row.querySelectorAll("button[data-action]").forEach((button) => {
        const name = button.dataset.action;
        if (allowed[name] && !allowed[name].includes(review.status)) button.hidden = true;
        button.addEventListener("click", async () => {
          if (name === "copy") { await navigator.clipboard.writeText(url); button.textContent = "Copied"; setTimeout(() => button.textContent = "Copy", 1000); }
          else if (name === "log") window.open(`/api/reviews/${review.reviewId}/log`, "_blank");
          else await action(review, name);
        });
      });
      list.append(row);
    }
  }
  function renderNightly() {
    nightlyList.replaceChildren();
    nightlyCount.textContent = `${nightlyReports.length} total`;
    if (!nightlyReports.length) { nightlyList.innerHTML = '<div class="empty">No nightly reports yet.</div>'; return; }
    for (const report of nightlyReports) {
      const row = document.createElement("article"); row.className = "nightly-row";
      const gameCount = report.stats?.analyzedGames ?? report.gamesAnalyzed ?? report.reviewCount ?? report.gameCount;
      const detail = [gameCount == null ? null : `${gameCount} games`, report.summary].filter(Boolean).join(" | ");
      const createdAt = report.completedAt || report.createdAt || report.updatedAt;
      row.innerHTML = '<div><strong></strong><span></span></div><span class="created"></span><div class="actions"><a>Open report</a></div>';
      row.querySelector("strong").textContent = report.title || "Nightly Slippi report";
      row.querySelector("div > span").textContent = detail || "Cross-game analysis";
      row.querySelector(".created").textContent = createdAt ? new Date(createdAt).toLocaleString() : "";
      const link = row.querySelector("a"); link.href = report.urls.report; link.setAttribute("aria-label", `Open ${row.querySelector("strong").textContent}`);
      if (!report.reportReady) { link.textContent = "Processing"; link.removeAttribute("href"); }
      nightlyList.append(row);
    }
  }
  async function refreshNightly() {
    try {
      const response = await fetch("/api/nightly", { cache: "no-store" });
      if (!response.ok) throw new Error("nightly request failed");
      const body = await response.json(); nightlyReports = body.reports || []; renderNightly();
    } catch (_) { nightlyList.innerHTML = '<div class="empty">Could not load nightly reports.</div>'; nightlyCount.textContent = ""; }
  }
  async function refresh() {
    try {
      const response = await fetch("/api/reviews", { cache: "no-store" }); const body = await response.json(); reviews = body.reviews || [];
      worker.textContent = `Worker ${body.worker?.status || "offline"}`; worker.dataset.online = String(Boolean(body.worker?.online)); worker.title = body.worker?.message || "";
      storage.textContent = `${fmtBytes(body.storage?.totalBytes || 0)} stored`; render();
    } catch (_) { list.textContent = "Could not load reviews."; }
  }
  document.querySelector("#cleanup-storage").addEventListener("click", async () => {
    if (!confirm("Remove pipeline intermediates for archived, failed, and cancelled reviews older than 30 days? Final reports and replay files are kept.")) return;
    const response = await fetch("/api/storage/cleanup", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ olderThanDays: 30 }) });
    const body = await response.json().catch(() => null); alert(response.ok ? `Reclaimed ${fmtBytes(body.cleanup.reclaimedBytes)}.` : (body?.error?.message || "Cleanup failed")); refresh();
  });
  search.addEventListener("input", render); statusFilter.addEventListener("change", render); ageFilter.addEventListener("change", render);
  SlpUploadPanel.mount(document.querySelector("#slp-upload-mount"), { onAccepted: refresh, onValidated: refresh });
  refresh(); refreshNightly(); setInterval(refresh, 5000); setInterval(refreshNightly, 30000);
})();
