(function () {
  console.log("🔥 Hooking DataView (low-level binary reads)...");

  const origGetInt32 = DataView.prototype.getInt32;
  const origGetUint16 = DataView.prototype.getUint16;
  const origGetFloat32 = DataView.prototype.getFloat32;

  function logIfInteresting(method, offset, value, view) {
    try {
      // Heuristic: FlatBuffers root reads often happen at low offsets
      if (offset < 32) {
        console.log(`📦 ${method} @ offset ${offset}:`, value);

        // 🔥 THIS IS THE GOLD
        console.trace("🧠 Binary parsing stack");
      }
    } catch (e) {}
  }

  DataView.prototype.getInt32 = function (offset, littleEndian) {
    const val = origGetInt32.call(this, offset, littleEndian);
    logIfInteresting("getInt32", offset, val, this);
    return val;
  };

  DataView.prototype.getUint16 = function (offset, littleEndian) {
    const val = origGetUint16.call(this, offset, littleEndian);
    logIfInteresting("getUint16", offset, val, this);
    return val;
  };

  DataView.prototype.getFloat32 = function (offset, littleEndian) {
    const val = origGetFloat32.call(this, offset, littleEndian);
    logIfInteresting("getFloat32", offset, val, this);
    return val;
  };

  console.log("✅ DataView hooked. Reload page.");
})();
