import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

const canvasHost = document.getElementById("viewport");
const hudEl = document.getElementById("hud-values");
const $ = (id) => document.getElementById(id);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf4f6f8);
scene.fog = new THREE.Fog(0xf4f6f8, 400, 1400);

const camera = new THREE.PerspectiveCamera(50, 1, 0.5, 4000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = false;
canvasHost.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.maxPolarAngle = Math.PI * 0.49;
controls.listenToKeyEvents(window);

scene.add(new THREE.HemisphereLight(0xffffff, 0x6b7c5a, 1.15));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(80, 220, 40);
scene.add(sun);

const skeletonGroup = new THREE.Group();
scene.add(skeletonGroup);
const trailGeom = new THREE.BufferGeometry();
const trailLine = new THREE.Line(
  trailGeom,
  new THREE.LineBasicMaterial({ color: 0xff9e00, transparent: true, opacity: 0.9 })
);
scene.add(trailLine);

const ballMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.55, 16, 12),
  new THREE.MeshStandardMaterial({ color: 0xffd21e, roughness: 0.4, metalness: 0.1 })
);
ballMesh.visible = false;
scene.add(ballMesh);

const batGeom = new THREE.CylinderGeometry(0.12, 0.18, 1, 8);
batGeom.translate(0, 0.5, 0);
const batMesh = new THREE.Mesh(
  batGeom,
  new THREE.MeshStandardMaterial({ color: 0x8a5a2b, roughness: 0.7 })
);
batMesh.visible = false;
scene.add(batMesh);

let play = null;
let frame = 0;
let playhead = 0;
let playing = false;
let follow = false;
let lastPreset = "action";
let povUid = null;
let povLabel = null;
let actorLines = [];
let applyingSliders = false;
let suppressControlEvent = false;
const defaultNear = 0.5;

function resize() {
  const w = canvasHost.clientWidth || window.innerWidth;
  const h = canvasHost.clientHeight || window.innerHeight;
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}
window.addEventListener("resize", resize);
resize();

function sphericalFromCamera() {
  const t = controls.target;
  const off = camera.position.clone().sub(t);
  const dist = off.length();
  const elev = THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(off.y / dist, -1, 1)));
  const azim = THREE.MathUtils.radToDeg(Math.atan2(off.x, off.z));
  return { azim, elev, dist, target: t.clone() };
}

function setSpherical(azim, elev, dist, target) {
  const t = target || controls.target.clone();
  const el = THREE.MathUtils.degToRad(elev);
  const az = THREE.MathUtils.degToRad(azim);
  suppressControlEvent = true;
  camera.position.set(
    t.x + dist * Math.cos(el) * Math.sin(az),
    t.y + dist * Math.sin(el),
    t.z + dist * Math.cos(el) * Math.cos(az)
  );
  controls.target.copy(t);
  controls.update();
  suppressControlEvent = false;
}

function halfWidthAtTarget(dist, fovDeg) {
  return dist * Math.tan(THREE.MathUtils.degToRad(fovDeg) * 0.5);
}

function hudText() {
  const s = sphericalFromCamera();
  const p = camera.position;
  const t = controls.target;
  const hw = halfWidthAtTarget(s.dist, camera.fov);
  const time = play ? play.times[frame] : 0;
  return [
    `preset      ${povLabel || (follow ? "follow" : lastPreset)}`,
    `t           ${time.toFixed(3)} s`,
    `position    ${fmtVec(p)}`,
    `look-at     ${fmtVec(t)}`,
    `azimuth     ${s.azim.toFixed(2)} deg`,
    `elevation   ${s.elev.toFixed(2)} deg`,
    `distance    ${s.dist.toFixed(2)} ft`,
    `fov         ${camera.fov.toFixed(1)} deg`,
    `half-width  ${hw.toFixed(1)} ft`,
  ].join("\n");
}

function fmtVec(v) {
  const n = (x) => x.toFixed(2).padStart(8);
  return `x${n(v.x)}  y${n(v.y)}  z${n(v.z)}`;
}

function markCustom() {
  exitPov();
  follow = false;
  controls.enableDamping = true;
  lastPreset = "custom";
  document.querySelectorAll("#presets button").forEach((b) => b.classList.remove("active"));
}

function refreshHud() {
  hudEl.textContent = hudText();
  if (applyingSliders) return;
  const s = sphericalFromCamera();
  $("s-azim").value = s.azim.toFixed(1);
  $("s-elev").value = s.elev.toFixed(1);
  $("s-dist").value = Math.round(s.dist);
  $("s-fov").value = Math.round(camera.fov);
  $("s-tx").value = s.target.x.toFixed(1);
  $("s-ty").value = s.target.y.toFixed(1);
  $("s-tz").value = s.target.z.toFixed(1);
  $("v-azim").textContent = `${s.azim.toFixed(1)}°`;
  $("v-elev").textContent = `${s.elev.toFixed(1)}°`;
  $("v-dist").textContent = `${s.dist.toFixed(0)} ft`;
  $("v-fov").textContent = `${camera.fov.toFixed(0)}°`;
  $("v-tx").textContent = `${s.target.x.toFixed(1)}`;
  $("v-ty").textContent = `${s.target.y.toFixed(1)}`;
  $("v-tz").textContent = `${s.target.z.toFixed(1)}`;
}

function copyHud() {
  navigator.clipboard.writeText(hudText()).catch(() => {
    const ta = document.createElement("textarea");
    ta.value = hudText();
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  });
  $("btn-copy").textContent = "Copied";
  setTimeout(() => { $("btn-copy").textContent = "Copy"; }, 900);
}

function actorIndex(uid) {
  return play.actors.findIndex((a) => a.uid === uid);
}

function headAt(uid, i) {
  const ai = actorIndex(uid);
  if (ai < 0) return null;
  const h = play.actors[ai].head && play.actors[ai].head[i];
  if (!h || h.length < 9) return null;
  return {
    pos: new THREE.Vector3(h[0], h[1], h[2]),
    fwd: new THREE.Vector3(h[3], h[4], h[5]).normalize(),
    up: new THREE.Vector3(h[6], h[7], h[8]).normalize(),
  };
}

function applyPov() {
  if (povUid == null) return;
  const h = headAt(povUid, frame);
  if (!h) return;
  camera.near = 0.12;
  if (camera.fov < 60) {
    camera.fov = 70;
  }
  camera.updateProjectionMatrix();
  camera.up.copy(h.up);
  camera.position.copy(h.pos);
  const look = h.pos.clone().add(h.fwd.clone().multiplyScalar(40));
  camera.lookAt(look);
  suppressControlEvent = true;
  controls.target.copy(look);
  suppressControlEvent = false;
}

function exitPov() {
  if (povUid == null) return;
  povUid = null;
  povLabel = null;
  camera.near = defaultNear;
  camera.up.set(0, 1, 0);
  camera.updateProjectionMatrix();
  document.querySelectorAll("#views button").forEach((b) => b.classList.remove("active"));
  if (play) {
    showFrame(frame, false);
  }
}

function setPov(uid, label) {
  if (povUid === uid) {
    snap("action");
    return;
  }
  follow = false;
  document.querySelectorAll("#presets button").forEach((b) => b.classList.remove("active"));
  lastPreset = "pov";
  povUid = uid;
  povLabel = label || "pov";
  document.querySelectorAll("#views button").forEach((b) => {
    b.classList.toggle("active", Number(b.dataset.uid) === uid);
  });
  applyPov();
  refreshHud();
}

function buildViews() {
  const views = play.views || { players: [], officials: [] };
  const fill = (el, items) => {
    el.innerHTML = "";
    items.forEach((v) => {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.uid = String(v.uid);
      b.textContent = v.label;
      b.addEventListener("click", () => setPov(v.uid, v.label));
      el.appendChild(b);
    });
  };
  fill($("views-players"), views.players || []);
  fill($("views-officials"), views.officials || []);
}

function actorBounds() {
  const b = play.bounds.actors;
  return {
    min: new THREE.Vector3(...b.min),
    max: new THREE.Vector3(...b.max),
    center: new THREE.Vector3().addVectors(
      new THREE.Vector3(...b.min), new THREE.Vector3(...b.max)
    ).multiplyScalar(0.5),
  };
}

function snap(name) {
  exitPov();
  follow = name === "follow";
  controls.enableDamping = !follow;
  lastPreset = name;
  document.querySelectorAll("#presets button").forEach((b) => {
    b.classList.toggle("active", b.dataset.preset === name);
  });
  const fov = camera.fov;
  const ab = actorBounds();
  if (name === "action") {
    const span = Math.min(
      160,
      Math.max(ab.max.x - ab.min.x, Math.abs(ab.max.z - ab.min.z), 50)
    );
    const dist = (span * 0.7) / Math.tan(THREE.MathUtils.degToRad(fov) * 0.5);
    const target = new THREE.Vector3(
      THREE.MathUtils.clamp(ab.center.x, -40, 40),
      4,
      THREE.MathUtils.clamp(ab.center.z, -120, 10)
    );
    setSpherical(-72, 16, dist, target);
  } else if (name === "infield") {
    // Stay inside the bowl (roofed parks clip a behind-home camera).
    setSpherical(-48, 22, 165, new THREE.Vector3(0, 4, -45));
  } else if (name === "full") {
    // From CF looking in — pulling back on the 1B action azim hits the stands.
    setSpherical(170, 18, 210, new THREE.Vector3(0, 6, -55));
  } else if (name === "pitcher") {
    const look = new THREE.Vector3(...play.pitcherLook);
    setSpherical(180, 12, 90, look);
  } else if (name === "follow") {
    applyFollow(true);
  }
  refreshHud();
}

function ballVel(i) {
  const b = play.ball;
  let lo = i - 1;
  while (lo >= 0 && !b[lo]) lo -= 1;
  let hi = i + 1;
  while (hi < b.length && !b[hi]) hi += 1;
  const a = lo >= 0 ? b[lo] : null;
  const c = hi < b.length ? b[hi] : null;
  const cur = b[i];
  if (a && c && play.times[hi] - play.times[lo] > 1e-4) {
    const dt = play.times[hi] - play.times[lo];
    return [(c[0] - a[0]) / dt, (c[2] - a[2]) / dt];
  }
  if (cur && a && play.times[i] - play.times[lo] > 1e-4) {
    const dt = play.times[i] - play.times[lo];
    return [(cur[0] - a[0]) / dt, (cur[2] - a[2]) / dt];
  }
  return [0, 0];
}

function applyFollow(snapNow) {
  const b = play.ball[frame];
  const tRel = play.times[frame];
  const released = play.tRelease == null || tRel >= play.tRelease;
  let target;
  if (released && b) target = new THREE.Vector3(b[0], b[1], b[2]);
  else target = new THREE.Vector3(...play.pitcherLook);
  const dist = 95;
  const elev = 16;
  let azim = 180;
  if (released && b) {
    const v = ballVel(frame);
    if (Math.hypot(v[0], v[1]) >= 12) {
      azim = THREE.MathUtils.radToDeg(Math.atan2(-v[0], -v[1]));
    }
  }
  if (snapNow) {
    setSpherical(azim, elev, dist, target);
    return;
  }
  const cur = sphericalFromCamera();
  const k = 1 - Math.exp(-(1 / 60) / 0.12);
  const tMix = cur.target.clone().lerp(target, k);
  let dAz = ((azim - cur.azim + 540) % 360) - 180;
  const azMix = cur.azim + THREE.MathUtils.clamp(dAz, -140 / 60, 140 / 60);
  setSpherical(azMix, elev, dist, tMix);
}

function buildActors() {
  actorLines.forEach((l) => skeletonGroup.remove(l));
  actorLines = play.actors.map((a) => {
    const geom = new THREE.BufferGeometry();
    const maxSegs = Math.max(
      ...a.frames.map((f) => (f ? f.length / 6 : 0)),
      1
    );
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(maxSegs * 6), 3));
    const line = new THREE.LineSegments(
      geom,
      new THREE.LineBasicMaterial({ color: a.color, linewidth: 2 })
    );
    skeletonGroup.add(line);
    return line;
  });
}

function showFrame(i, syncHead = true) {
  if (!play) return;
  frame = Math.max(0, Math.min(play.times.length - 1, i));
  if (syncHead) playhead = frame;
  $("s-time").value = play.times.length <= 1 ? 0 : frame / (play.times.length - 1);
  $("v-time").textContent = `${play.times[frame].toFixed(2)}s`;

  play.actors.forEach((a, ai) => {
    const line = actorLines[ai];
    const segs = a.frames[frame];
    const attr = line.geometry.getAttribute("position");
    attr.array.fill(0);
    if (segs && segs.length) {
      attr.array.set(segs);
      line.geometry.setDrawRange(0, segs.length / 3);
      line.visible = a.uid !== povUid;
    } else {
      line.visible = false;
    }
    attr.needsUpdate = true;
  });

  const ball = play.ball[frame];
  if (ball) {
    ballMesh.position.set(ball[0], ball[1], ball[2]);
    ballMesh.visible = true;
  } else {
    ballMesh.visible = false;
  }

  const trail = [];
  for (let k = 0; k <= frame; k++) {
    const p = play.ball[k];
    if (!p) {
      if (trail.length) trail.push(trail[trail.length - 1]);
      continue;
    }
    trail.push(new THREE.Vector3(p[0], p[1], p[2]));
  }
  if (trail.length > 1) {
    trailGeom.setFromPoints(trail);
    trailLine.visible = true;
  } else {
    trailLine.visible = false;
  }

  const bat = play.bat[frame];
  if (bat) {
    const h = new THREE.Vector3(...bat.handle);
    const hd = new THREE.Vector3(...bat.head);
    const dir = hd.clone().sub(h);
    const len = dir.length();
    batMesh.scale.set(1, len, 1);
    batMesh.position.copy(h);
    batMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
    batMesh.visible = true;
  } else {
    batMesh.visible = false;
  }

  if (follow) applyFollow(false);
  else if (povUid != null) applyPov();
  refreshHud();
}

async function loadPark(info) {
  if (!info || !info.file) {
    const g = new THREE.Mesh(
      new THREE.PlaneGeometry(400, 400),
      new THREE.MeshStandardMaterial({ color: 0x5b9e4a })
    );
    g.rotation.x = -Math.PI / 2;
    scene.add(g);
    return;
  }
  const draco = new DRACOLoader();
  draco.setDecoderPath("https://www.gstatic.com/draco/versioned/decoders/1.5.7/");
  const loader = new GLTFLoader();
  loader.setDRACOLoader(draco);
  try {
    const gltf = await loader.loadAsync("data/ballpark.glb");
    const root = gltf.scene;
    root.scale.setScalar(info.mToFt || 3.28084);
    root.traverse((obj) => {
      const n = obj.name || "";
      if (n.includes("StartingCube") || n.includes("CameraCollider")) {
        obj.visible = false;
      }
      if (obj.isLight) obj.visible = false;
      if (obj.isMesh) {
        obj.castShadow = false;
        obj.receiveShadow = false;
      }
    });
    scene.add(root);
  } catch (err) {
    console.warn("ballpark glb failed", err);
    const g = new THREE.Mesh(
      new THREE.PlaneGeometry(400, 400),
      new THREE.MeshStandardMaterial({ color: 0x5b9e4a })
    );
    g.rotation.x = -Math.PI / 2;
    scene.add(g);
  }
}

function bindUi() {
  document.querySelectorAll("#presets [data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => snap(btn.dataset.preset));
  });
  $("btn-copy").addEventListener("click", copyHud);
  $("btn-play").addEventListener("click", togglePlay);
  $("s-time").addEventListener("input", () => {
    const u = Number($("s-time").value);
    showFrame(Math.round(u * (play.times.length - 1)));
  });
  const applySliderCam = () => {
    applyingSliders = true;
    markCustom();
    setSpherical(
      Number($("s-azim").value),
      Number($("s-elev").value),
      Number($("s-dist").value),
      new THREE.Vector3(
        Number($("s-tx").value),
        Number($("s-ty").value),
        Number($("s-tz").value)
      )
    );
    camera.fov = Number($("s-fov").value);
    camera.updateProjectionMatrix();
    applyingSliders = false;
    refreshHud();
  };
  ["s-azim", "s-elev", "s-dist", "s-fov", "s-tx", "s-ty", "s-tz"].forEach((id) => {
    $(id).addEventListener("input", applySliderCam);
  });
  controls.addEventListener("change", refreshHud);
  controls.addEventListener("start", () => {
    if (!suppressControlEvent && !applyingSliders) markCustom();
  });
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
      e.preventDefault();
      togglePlay();
    }
  });
}

function togglePlay() {
  playing = !playing;
  $("btn-play").textContent = playing ? "Pause" : "Play";
}

let lastTick = performance.now();
function tick(now) {
  requestAnimationFrame(tick);
  const dt = Math.min(0.05, (now - lastTick) / 1000);
  lastTick = now;
  if (playing && play) {
    playhead += play.fps * dt;
    if (playhead >= play.times.length - 1) {
      showFrame(play.times.length - 1);
      playing = false;
      $("btn-play").textContent = "Play";
    } else {
      const i = Math.floor(playhead);
      if (i !== frame) showFrame(i, false);
      else if (follow) applyFollow(false);
      else if (povUid != null) applyPov();
    }
  } else if (follow) {
    applyFollow(false);
  } else if (povUid != null) {
    applyPov();
  }
  if (povUid == null) controls.update();
  renderer.render(scene, camera);
}

async function main() {
  play = await fetch("data/play.json").then((r) => {
    if (!r.ok) throw new Error("data/play.json missing — run serve.py <play_dir>");
    return r.json();
  });
  document.title = `Gameday 3D  ${play.gamePk || ""}  ${play.playId || ""}`;
  $("s-time").max = 1;
  buildActors();
  buildViews();
  await loadPark(play.ballpark);
  bindUi();
  showFrame(0);
  snap("action");
  requestAnimationFrame(tick);
}

main().catch((err) => {
  hudEl.textContent = String(err);
  console.error(err);
});
