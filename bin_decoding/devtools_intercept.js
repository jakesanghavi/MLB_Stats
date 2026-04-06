(function () {
  console.log("🔍 Safe FlatBuffer hook active");

  window.__fb = {
    buffers: [],
  };

  // -----------------------------
  // 1. Intercept .bin responses (SAFE)
  // -----------------------------
  const origFetch = window.fetch;

  window.fetch = async function (...args) {
    const res = await origFetch.apply(this, args);

    try {
      const url = args[0]?.toString() || "";

      if (url.includes(".bin")) {
        const clone = res.clone();
        const buf = await clone.arrayBuffer();
        const arr = new Uint8Array(buf);

        console.log("📦 BIN intercepted:", url);
        console.log("First 64 bytes:", arr.slice(0, 64));

        window.__fb.buffers.push({
          url,
          buffer: arr,
        });
      }
    } catch (e) {}

    return res;
  };

  // -----------------------------
  // 2. Detect FlatBuffer usage (PASSIVE)
  // -----------------------------
  const origArrayBuffer = Response.prototype.arrayBuffer;

  Response.prototype.arrayBuffer = async function () {
    const buf = await origArrayBuffer.call(this);

    try {
      if (this.url && this.url.includes(".bin")) {
        console.log("📦 arrayBuffer used for:", this.url);
      }
    } catch {}

    return buf;
  };

  // -----------------------------
  // 3. Helper to inspect buffers
  // -----------------------------
  window.inspectBuffers = function () {
    if (!window.__fb.buffers.length) {
      console.log("❌ No buffers captured yet");
      return;
    }

    for (const { url, buffer } of window.__fb.buffers) {
      console.log("🔎 Buffer:", url);

      // show ASCII strings inside buffer
      const text = new TextDecoder().decode(buffer);
      const matches = text.match(/[ -~]{4,}/g); // printable strings

      console.log("🧵 Strings found:", matches?.slice(0, 20));
    }
  };

  console.log("✅ Ready. Reload the page, then run:");
  console.log("👉 inspectBuffers()");
})();
