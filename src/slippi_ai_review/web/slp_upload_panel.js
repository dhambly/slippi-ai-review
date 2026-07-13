/* Slippi upload, validation, analyzed-player selection, and quality controls. */
(function (global) {
  "use strict";

  function mount(target, options) {
    if (!(target instanceof Element)) throw new TypeError("SlpUploadPanel target must be an Element");
    const config = options || {};
    const endpoint = config.endpoint || "/api/slp-upload";
    const root = document.createElement("form");
    root.className = "slp-upload-panel";
    root.innerHTML = `
      <label class="slp-upload-panel__drop">
        <input class="slp-upload-panel__input" type="file" accept=".slp,application/octet-stream" required>
        <strong>Drop a Slippi replay here</strong><span>or choose a .slp file</span>
      </label>
      <fieldset class="slp-upload-panel__quality"><legend>Analysis depth</legend>
        <label><input type="radio" name="quality" value="quick"><span>Quick</span></label>
        <label><input type="radio" name="quality" value="standard" checked><span>Standard</span></label>
        <label><input type="radio" name="quality" value="deep"><span>Deep</span></label>
      </fieldset>
      <button class="slp-upload-panel__submit" type="submit">Validate replay</button>
      <progress class="slp-upload-panel__progress" max="100" value="0" hidden></progress>
      <div class="slp-upload-panel__target" hidden></div>
      <output class="slp-upload-panel__status" aria-live="polite"></output>`;
    target.replaceChildren(root);

    const input = root.querySelector("input[type=file]");
    const drop = root.querySelector(".slp-upload-panel__drop");
    const button = root.querySelector(".slp-upload-panel__submit");
    const progress = root.querySelector("progress");
    const targetPicker = root.querySelector(".slp-upload-panel__target");
    const status = root.querySelector("output");

    function describePlayer(player) {
      const color = player.character.colorName ? ` (${player.character.colorName})` : "";
      const identities = [...new Set([player.displayName, player.nametag, player.connectCode].filter(Boolean))];
      return `P${player.port} ${player.character.name}${color}${identities.length ? ` · ${identities.join(" · ")}` : ""}`;
    }
    function quality() { return root.querySelector('input[name="quality"]:checked').value; }
    function showReviewLink(review, prefix) {
      status.replaceChildren(document.createTextNode(prefix));
      if (!review?.urls?.review) return;
      const link = document.createElement("a");
      link.className = "slp-upload-panel__review-link";
      link.href = review.urls.review;
      link.textContent = "Open review status";
      status.append(document.createElement("br"), link);
    }
    async function chooseTarget(payload, player) {
      targetPicker.querySelectorAll("button").forEach((item) => { item.disabled = true; });
      status.dataset.state = "loading";
      status.textContent = `Queuing ${describePlayer(player)}...`;
      try {
        const response = await fetch(`${payload.review.urls.status}/target`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ playerIndex: player.playerIndex, qualityPreset: quality() }),
        });
        const selection = await response.json().catch(() => null);
        if (!response.ok || !selection?.ok) throw new Error(selection?.error?.message || "Player selection failed");
        targetPicker.hidden = true; targetPicker.replaceChildren(); input.value = ""; button.hidden = false;
        status.dataset.state = "accepted";
        showReviewLink(selection.review, `Queued: ${describePlayer(player)} · ${quality()} quality`);
        config.onAccepted?.(selection);
      } catch (error) {
        targetPicker.querySelectorAll("button").forEach((item) => { item.disabled = false; });
        status.dataset.state = "error"; status.textContent = error.message || "Player selection failed";
        config.onError?.(error);
      }
    }
    function upload(file) {
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", endpoint);
        xhr.setRequestHeader("Content-Type", "application/octet-stream");
        xhr.setRequestHeader("X-Slp-Filename", encodeURIComponent(file.name));
        xhr.upload.addEventListener("progress", (event) => {
          progress.hidden = false;
          progress.value = event.lengthComputable ? Math.round(event.loaded / event.total * 100) : 0;
          status.textContent = event.lengthComputable ? `Uploading ${progress.value}%...` : "Uploading replay...";
        });
        xhr.addEventListener("load", () => {
          let payload = null; try { payload = JSON.parse(xhr.responseText); } catch (_) {}
          if (xhr.status < 200 || xhr.status >= 300 || !payload?.ok) reject(new Error(payload?.error?.message || `Upload failed (${xhr.status})`));
          else resolve(payload);
        });
        xhr.addEventListener("error", () => reject(new Error("Upload failed")));
        xhr.send(file);
      });
    }
    async function submit(event) {
      event.preventDefault();
      const file = input.files?.[0]; if (!file) return;
      button.disabled = true; status.dataset.state = "loading"; status.textContent = "Uploading replay...";
      try {
        const payload = await upload(file);
        progress.hidden = true;
        targetPicker.replaceChildren();
        for (const player of payload.players) {
          const option = document.createElement("button"); option.type = "button";
          option.textContent = `Analyze ${describePlayer(player)}`;
          option.addEventListener("click", () => chooseTarget(payload, player)); targetPicker.append(option);
        }
        targetPicker.hidden = false; button.hidden = true; status.dataset.state = "validated";
        const duplicate = payload.duplicates?.[0];
        const duplicateText = duplicate ? ` Existing ${duplicate.status} review found: ${duplicate.reviewId.slice(0, 8)}. You can still run this preset/player combination.` : "";
        showReviewLink(payload.review, `Validated ${payload.players.map(describePlayer).join(" vs ")}. Choose the analyzed player.${duplicateText}`);
        config.onValidated?.(payload);
      } catch (error) {
        progress.hidden = true; status.dataset.state = "error"; status.textContent = error.message || "Upload failed"; config.onError?.(error);
      } finally { button.disabled = false; }
    }
    ["dragenter", "dragover"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); drop.dataset.dragging = "true"; }));
    ["dragleave", "drop"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); delete drop.dataset.dragging; }));
    drop.addEventListener("drop", (event) => { const file = event.dataTransfer?.files?.[0]; if (!file) return; const transfer = new DataTransfer(); transfer.items.add(file); input.files = transfer.files; drop.querySelector("span").textContent = file.name; });
    input.addEventListener("change", () => { if (input.files?.[0]) drop.querySelector("span").textContent = input.files[0].name; });
    root.addEventListener("submit", submit);
    return { input, destroy() { root.removeEventListener("submit", submit); root.remove(); } };
  }
  global.SlpUploadPanel = Object.freeze({ mount });
})(window);
