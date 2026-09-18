import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

// Code-only draw knobs (not in the GUI). Bump these, restart serve.py / hard-reload.
const LIMB_THICKEN = 2.8;     // bone cylinder radius vs BONE_RADIUS_FT
const TRAIL_THICKEN = 2.5;    // ball-path line width vs TRAIL_WIDTH_PX
const BALL_THICKEN = 1.5;     // ball sphere radius vs BALL_RADIUS_FT
const SHOW_BALL_TRAIL = true; // false hides the yellow ball-history line
const BONE_RADIUS_FT = 0.06;
const TRAIL_WIDTH_PX = 2;
const BALL_RADIUS_FT = 0.4;
const BAT_MODEL_LEN_FT = 2.843; // bat.glb knob -> barrel
// Only used at pitch release and bat contact. Other times are untouched.
const EASY_BLEND_S = 0.18;

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
const boneGeom = new THREE.CylinderGeometry(1, 1, 1, 6);
const boneDummy = new THREE.Object3D();

const trailMat = new LineMaterial({
  color: 0xff9e00,
  linewidth: TRAIL_WIDTH_PX * TRAIL_THICKEN,
  transparent: true,
  opacity: 0.9,
});
const trailLine = new Line2(new LineGeometry(), trailMat);
trailLine.visible = false;
scene.add(trailLine);

const ballMesh = new THREE.Mesh(
  new THREE.SphereGeometry(BALL_RADIUS_FT * BALL_THICKEN, 16, 12),
  new THREE.MeshStandardMaterial({ color: 0xffd21e, roughness: 0.4, metalness: 0.1 })
);
ballMesh.visible = false;
scene.add(ballMesh);

const batHolder = new THREE.Group();
batHolder.visible = false;
scene.add(batHolder);
const batFallbackGeom = new THREE.CylinderGeometry(0.045, 0.11, 1, 12);
batFallbackGeom.translate(0, 0.5, 0);
const batFallback = new THREE.Mesh(
  batFallbackGeom,
  new THREE.MeshStandardMaterial({ color: 0x8a5a2b, roughness: 0.65 })
);
batHolder.add(batFallback);
let batModelLen = 1;
let batIsMesh = false;

const HEAD_POSES = new Set(["FOLLOW_NECK", "ALWAYS_BALL", "SMART_VISION", "EASY_VISION"]);

let play = null;
let frame = 0;
let playhead = 0;
let playing = false;
let follow = false;
let lastPreset = "action";
let povUid = null;
let povLabel = null;
let headPose = "EASY_VISION";
let actorBones = [];
let applyingSliders = false;
let suppressControlEvent = false;
const defaultNear = 0.5;

function resize() {
  const w = canvasHost.clientWidth || window.innerWidth;
  const h = canvasHost.clientHeight || window.innerHeight;
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
  trailMat.resolution.set(w, h);
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
  const time = play ? timeAt(playing ? playhead : frame) : 0;
  return [
    `play        ${play ? (play.gamePk || "") : ""}  ${play ? shortPlayId(play.playId) : ""}`,
    `pitcher     ${playPitcher()}`,
    `preset      ${povLabel || (follow ? "follow" : lastPreset)}`,
    `head        ${headPose}`,
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

function shortPlayId(id) {
  if (!id) return "";
  const s = String(id);
  return s.length > 18 ? `${s.slice(0, 8)}…` : s;
}

function playPitcher() {
  const views = (play && play.views && play.views.players) || [];
  const p = views.find((v) => v.slot === "P");
  if (p) return p.label || p.name || "";
  return (window.PLAY_META && window.PLAY_META.pitcher) || "";
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

function slotFor(uid) {
  const ai = actorIndex(uid);
  if (ai >= 0 && play.actors[ai].slot) return play.actors[ai].slot;
  const views = play.views || {};
  for (const v of [...(views.players || []), ...(views.officials || [])]) {
    if (v.uid === uid) return v.slot || null;
  }
  return null;
}

function ballAt(i) {
  const b = play && play.ball && play.ball[i];
  return b ? new THREE.Vector3(b[0], b[1], b[2]) : null;
}

function roleFallback(slot) {
  if (slot === "1B runner") return new THREE.Vector3(0, 3, -127.28);
  if (slot === "2B runner") return new THREE.Vector3(-63.64, 3, -63.64);
  if (slot === "3B runner") return new THREE.Vector3(0, 2.5, 0);
  if (slot === "Batter" || slot === "C" || (slot && slot.endsWith("runner"))) {
    return new THREE.Vector3(0, 5, -60.5);
  }
  return new THREE.Vector3(0, 2.5, 0);
}

function basisFromFwd(eye, fwd, upHint) {
  const f = fwd.clone().normalize();
  let up = (upHint || new THREE.Vector3(0, 1, 0)).clone();
  if (Math.abs(f.dot(up.clone().normalize())) > 0.95) {
    up.set(0, 0, 1);
  }
  up.addScaledVector(f, -up.dot(f));
  if (up.lengthSq() < 1e-8) up.set(0, 1, 0);
  up.normalize();
  return { pos: eye.clone().addScaledVector(f, 0.35), fwd: f, up };
}

function turnToward(neck, want, maxDeg) {
  const a = neck.clone().normalize();
  const b = want.clone().normalize();
  const ang = THREE.MathUtils.radToDeg(a.angleTo(b));
  if (ang <= maxDeg) return b;
  if (ang < 1e-3) return a;
  const q = new THREE.Quaternion().setFromUnitVectors(a, b);
  const q2 = new THREE.Quaternion().slerpQuaternions(
    new THREE.Quaternion(),
    q,
    maxDeg / ang
  );
  return a.clone().applyQuaternion(q2).normalize();
}

function smartFwd(eye, neck, slot, ball) {
  const cone = 80;
  const ranked = [];
  if (ball) ranked.push({ pri: 0, tgt: ball });
  if (slot === "1B runner") ranked.push({ pri: 1, tgt: new THREE.Vector3(0, 3, -127.28) });
  if (slot === "2B runner") ranked.push({ pri: 1, tgt: new THREE.Vector3(-63.64, 3, -63.64) });
  if (slot === "3B runner") ranked.push({ pri: 1, tgt: new THREE.Vector3(0, 2.5, 0) });
  ranked.push({ pri: 2, tgt: roleFallback(slot) });
  const scored = [];
  for (const r of ranked) {
    const d = r.tgt.clone().sub(eye);
    if (d.lengthSq() < 1e-6) continue;
    d.normalize();
    scored.push({ pri: r.pri, ang: THREE.MathUtils.radToDeg(neck.angleTo(d)), d });
  }
  if (!scored.length) return neck.clone();
  const inCone = scored.filter((s) => s.ang <= cone);
  if (inCone.length) {
    inCone.sort((a, b) => a.pri - b.pri || a.ang - b.ang);
    return inCone[0].d;
  }
  scored.sort((a, b) => a.pri - b.pri);
  return turnToward(neck, scored[0].d, cone);
}

function timeAt(t) {
  const n = play.times.length;
  const x = Math.max(0, Math.min(n - 1, t));
  const i0 = Math.floor(x);
  const i1 = Math.min(n - 1, i0 + 1);
  const f = x - i0;
  return play.times[i0] * (1 - f) + play.times[i1] * f;
}

function mix3(a, b, f) {
  return [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ];
}

function catmull3(p0, p1, p2, p3, f) {
  const t2 = f * f;
  const t3 = t2 * f;
  const out = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    out[i] = 0.5 * (
      (2 * p1[i])
      + (-p0[i] + p2[i]) * f
      + (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * t2
      + (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * t3
    );
  }
  return out;
}

function lerpXyz(arr, t, spline = false) {
  if (!arr || !arr.length) return null;
  const x = Math.max(0, Math.min(arr.length - 1, t));
  const i0 = Math.floor(x);
  const i1 = Math.min(arr.length - 1, i0 + 1);
  const a = arr[i0];
  const b = arr[i1];
  if (!a) return b ? b.slice() : null;
  if (!b || i0 === i1) return a.slice();
  const f = x - i0;
  if (!spline) return mix3(a, b, f);
  const p0 = i0 > 0 && arr[i0 - 1] ? arr[i0 - 1] : a;
  const p3 = i1 + 1 < arr.length && arr[i1 + 1] ? arr[i1 + 1] : b;
  if ((i0 > 0 && !arr[i0 - 1]) || (i1 + 1 < arr.length && !arr[i1 + 1])) {
    return mix3(a, b, f);
  }
  return catmull3(p0, a, b, p3, f);
}

function lerpBat(t) {
  const arr = play.bat;
  if (!arr || !arr.length) return null;
  const x = Math.max(0, Math.min(arr.length - 1, t));
  const i0 = Math.floor(x);
  const i1 = Math.min(arr.length - 1, i0 + 1);
  const a = arr[i0];
  const b = arr[i1];
  if (!a) return b || null;
  if (!b || i0 === i1) return a;
  const f = x - i0;
  return { handle: mix3(a.handle, b.handle, f), head: mix3(a.head, b.head, f) };
}

function lerpSegs(a, b, f) {
  if (!a) return b;
  if (!b || a.length !== b.length) return a;
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * f;
  return out;
}

function eventWeight(tSec, eventT, tau = EASY_BLEND_S) {
  if (eventT == null || tSec == null) return null;
  if (tSec <= eventT) return 0;
  if (tSec >= eventT + tau) return 1;
  const u = (tSec - eventT) / tau;
  return u * u * (3 - 2 * u);
}

function playheadAtTime(sec) {
  if (!play || !play.times.length) return 0;
  const fps = play.fps || 20;
  return Math.max(0, Math.min(play.times.length - 1, sec * fps));
}

function slerpFwd(a, b, w) {
  const A = a.clone().normalize();
  const B = b.clone().normalize();
  const q = new THREE.Quaternion().setFromUnitVectors(A, B);
  const q2 = new THREE.Quaternion().slerpQuaternions(new THREE.Quaternion(), q, w);
  return A.applyQuaternion(q2).normalize();
}

function blendBasis(a, b, w) {
  if (!a) return b;
  if (!b) return a;
  const fwd = slerpFwd(a.fwd, b.fwd, w);
  let up = a.up.clone().lerp(b.up, w);
  up.addScaledVector(fwd, -up.dot(fwd));
  if (up.lengthSq() < 1e-8) up = b.up.clone();
  up.normalize();
  return { pos: a.pos.clone().lerp(b.pos, w), fwd, up };
}

function lookFollowNeck(neckSample) {
  return basisFromFwd(neckSample.pos, neckSample.fwd, neckSample.up);
}

function lookAlwaysBall(neckSample, uid, t) {
  const xyz = lerpXyz(play.ball, t, true);
  const ball = xyz ? new THREE.Vector3(...xyz) : null;
  const slot = slotFor(uid);
  const tgt = ball || roleFallback(slot);
  return basisFromFwd(neckSample.pos, tgt.clone().sub(neckSample.pos), null);
}

function resolveLook(neckSample, uid, t) {
  const eye = neckSample.pos;
  const neck = neckSample.fwd;
  const neckUp = neckSample.up;
  const tSec = timeAt(t);
  let mode = headPose;
  if (mode === "EASY_VISION") {
    const wC = eventWeight(tSec, play.tContact);
    const ballLook = lookAlwaysBall(neckSample, uid, t);
    const neckLook = lookFollowNeck(neckSample);
    if (wC == null) {
      return tSec >= (play.tContact ?? Infinity) ? neckLook : ballLook;
    }
    if (wC >= 1) return neckLook;
    if (wC > 0) {
      const preT = playheadAtTime(play.tContact - 1e-3);
      const preLook = lookAlwaysBall(neckSample, uid, preT);
      return blendBasis(preLook, neckLook, wC);
    }
    const wR = eventWeight(tSec, play.tRelease);
    if (wR != null && wR > 0 && wR < 1) {
      const preT = playheadAtTime(play.tRelease - 1e-3);
      const preLook = lookAlwaysBall(neckSample, uid, preT);
      return blendBasis(preLook, ballLook, wR);
    }
    return ballLook;
  }
  if (mode === "FOLLOW_NECK") {
    return basisFromFwd(eye, neck, neckUp);
  }
  const xyz = lerpXyz(play.ball, t, true);
  const ball = xyz ? new THREE.Vector3(...xyz) : null;
  const slot = slotFor(uid);
  if (mode === "ALWAYS_BALL") {
    const tgt = ball || roleFallback(slot);
    return basisFromFwd(eye, tgt.clone().sub(eye), null);
  }
  return basisFromFwd(eye, smartFwd(eye, neck, slot, ball), null);
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

function headAtF(uid, t) {
  if (!play) return null;
  const n = play.times.length;
  if (!n) return null;
  const x = Math.max(0, Math.min(n - 1, t));
  const i0 = Math.floor(x);
  const i1 = Math.min(n - 1, i0 + 1);
  const a = headAt(uid, i0);
  const b = headAt(uid, i1);
  if (!a) return b ? resolveLook(b, uid, x) : null;
  if (!b || i0 === i1) return resolveLook(a, uid, x);
  const f = x - i0;
  const neck = {
    pos: a.pos.clone().lerp(b.pos, f),
    fwd: a.fwd.clone().lerp(b.fwd, f).normalize(),
    up: a.up.clone().lerp(b.up, f).normalize(),
  };
  return resolveLook(neck, uid, x);
}

function applyPov() {
  if (povUid == null) return;
  const h = headAtF(povUid, playing ? playhead : frame);
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
  document.querySelectorAll("#views-players button, #views-officials button").forEach((b) => {
    b.classList.remove("active");
  });
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
  document.querySelectorAll("#views-players button, #views-officials button").forEach((b) => {
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

function playheadNow() {
  if (!play) return 0;
  return playing ? playhead : frame;
}

function ballVelAt(t) {
  const a = lerpXyz(play.ball, t - 1, true);
  const c = lerpXyz(play.ball, t + 1, true);
  if (a && c) {
    const dt = timeAt(t + 1) - timeAt(t - 1);
    if (dt > 1e-4) return [(c[0] - a[0]) / dt, (c[2] - a[2]) / dt];
  }
  return [0, 0];
}

function applyFollow(snapNow) {
  const t = playheadNow();
  const b = lerpXyz(play.ball, t, true);
  const tRel = timeAt(t);
  const released = play.tRelease == null || tRel >= play.tRelease;
  let target;
  if (released && b) target = new THREE.Vector3(b[0], b[1], b[2]);
  else target = new THREE.Vector3(...play.pitcherLook);
  const dist = 95;
  const elev = 16;
  let azim = 180;
  if (released && b) {
    const v = ballVelAt(t);
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
  actorBones.forEach((b) => {
    skeletonGroup.remove(b.mesh);
    b.mesh.material.dispose();
  });
  const radius = BONE_RADIUS_FT * LIMB_THICKEN;
  actorBones = play.actors.map((a) => {
    const maxSegs = Math.max(
      ...a.frames.map((f) => (f ? f.length / 6 : 0)),
      1
    );
    const mat = new THREE.MeshStandardMaterial({
      color: a.color,
      roughness: 0.55,
      metalness: 0.05,
    });
    const mesh = new THREE.InstancedMesh(boneGeom, mat, maxSegs);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    skeletonGroup.add(mesh);
    return { mesh, maxSegs, radius };
  });
}

function poseBat(bat) {
  if (!bat) {
    batHolder.visible = false;
    return;
  }
  const h = new THREE.Vector3(...bat.handle);
  const hd = new THREE.Vector3(...bat.head);
  const dir = hd.sub(h);
  const len = dir.length();
  if (len < 1e-3) {
    batHolder.visible = false;
    return;
  }
  batHolder.position.copy(h);
  batHolder.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  if (batIsMesh) {
    const s = len / batModelLen;
    batHolder.scale.set(s, s, s);
  } else {
    batHolder.scale.set(1, len, 1);
  }
  batHolder.visible = true;
}

function poseBones(entry, segs, hide) {
  const mesh = entry.mesh;
  if (hide || !segs || !segs.length) {
    mesh.visible = false;
    return;
  }
  mesh.visible = true;
  const n = Math.min(entry.maxSegs, Math.floor(segs.length / 6));
  const yAxis = new THREE.Vector3(0, 1, 0);
  const dir = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    const o = i * 6;
    const x0 = segs[o], y0 = segs[o + 1], z0 = segs[o + 2];
    const x1 = segs[o + 3], y1 = segs[o + 4], z1 = segs[o + 5];
    dir.set(x1 - x0, y1 - y0, z1 - z0);
    const len = dir.length();
    if (len < 1e-4) {
      boneDummy.scale.set(0, 0, 0);
    } else {
      boneDummy.position.set((x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5);
      boneDummy.scale.set(entry.radius, len, entry.radius);
      boneDummy.quaternion.setFromUnitVectors(yAxis, dir.multiplyScalar(1 / len));
    }
    boneDummy.updateMatrix();
    mesh.setMatrixAt(i, boneDummy.matrix);
  }
  for (let i = n; i < entry.maxSegs; i++) {
    boneDummy.scale.set(0, 0, 0);
    boneDummy.updateMatrix();
    mesh.setMatrixAt(i, boneDummy.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
  mesh.count = n;
}

function showFrame(i, syncHead = true) {
  if (!play) return;
  const n = play.times.length;
  const t = Math.max(0, Math.min(n - 1, i));
  frame = Math.floor(t);
  if (syncHead) playhead = t;
  $("s-time").value = n <= 1 ? 0 : t / (n - 1);
  $("v-time").textContent = `${timeAt(t).toFixed(2)}s`;

  const frac = t - frame;
  const i1 = Math.min(n - 1, frame + 1);
  play.actors.forEach((a, ai) => {
    const segs = lerpSegs(a.frames[frame], a.frames[i1], frac);
    poseBones(actorBones[ai], segs, a.uid === povUid);
  });

  const ball = lerpXyz(play.ball, t, true);
  if (ball) {
    ballMesh.position.set(ball[0], ball[1], ball[2]);
    ballMesh.visible = true;
  } else {
    ballMesh.visible = false;
  }

  const trail = [];
  for (let k = 0; k <= frame; k++) {
    const p = play.ball[k];
    if (!p) continue;
    trail.push(p[0], p[1], p[2]);
  }
  if (ball && (trail.length < 3 || frame < t)) {
    trail.push(ball[0], ball[1], ball[2]);
  }
  if (SHOW_BALL_TRAIL && trail.length >= 6) {
    const old = trailLine.geometry;
    trailLine.geometry = new LineGeometry();
    trailLine.geometry.setPositions(trail);
    trailLine.computeLineDistances();
    trailLine.visible = true;
    if (old && old.dispose) old.dispose();
  } else {
    trailLine.visible = false;
  }

  poseBat(lerpBat(t));

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
    const gltf = await loader.loadAsync(`data/ballpark.glb?v=${Date.now()}`);
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
  $("btn-save").addEventListener("click", saveVideo);
  $("s-time").addEventListener("input", () => {
    const u = Number($("s-time").value);
    showFrame(u * (play.times.length - 1));
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

let recording = false;

function togglePlay() {
  if (recording) return;
  playing = !playing;
  $("btn-play").textContent = playing ? "Pause" : "Play";
}

function pickRecorderMime() {
  const types = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  for (const t of types) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

function saveVideo() {
  if (!play || recording) return;
  if (!window.MediaRecorder) {
    $("btn-save").textContent = "No recorder";
    setTimeout(() => { $("btn-save").textContent = "Save video"; }, 1600);
    return;
  }
  const mime = pickRecorderMime();
  const canvas = renderer.domElement;
  const fps = Math.max(20, Number(play.fps) || 20);
  let stream;
  try {
    stream = canvas.captureStream(fps);
  } catch (err) {
    console.warn("captureStream failed", err);
    $("btn-save").textContent = "Capture failed";
    setTimeout(() => { $("btn-save").textContent = "Save video"; }, 1600);
    return;
  }
  const rec = mime
    ? new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8_000_000 })
    : new MediaRecorder(stream);
  const chunks = [];
  rec.ondataavailable = (e) => {
    if (e.data && e.data.size) chunks.push(e.data);
  };
  rec.onerror = (e) => {
    console.warn("MediaRecorder error", e);
    recording = false;
    playing = false;
    $("btn-save").disabled = false;
    $("btn-save").textContent = "Save failed";
    setTimeout(() => { $("btn-save").textContent = "Save video"; }, 1800);
  };
  rec.onstop = () => {
    stream.getTracks().forEach((tr) => tr.stop());
    const blob = new Blob(chunks, { type: rec.mimeType || "video/webm" });
    if (blob.size < 64) {
      $("btn-save").disabled = false;
      $("btn-save").textContent = "Empty video";
      recording = false;
      setTimeout(() => { $("btn-save").textContent = "Save video"; }, 1800);
      return;
    }
    const a = document.createElement("a");
    const id = (play.playId || "play").toString().slice(0, 8);
    a.href = URL.createObjectURL(blob);
    a.download = `gameday3d_${play.gamePk || "game"}_${id}.webm`;
    a.click();
    URL.revokeObjectURL(a.href);
    recording = false;
    $("btn-save").disabled = false;
    $("btn-save").textContent = "Save video";
    $("btn-play").textContent = "Play";
  };
  recording = true;
  $("btn-save").disabled = true;
  $("btn-save").textContent = "Recording…";
  showFrame(0);
  playing = true;
  $("btn-play").textContent = "Pause";
  rec.start(250);
  const tEnd = play.times[play.times.length - 1];
  const started = performance.now();
  const finish = () => {
    if (rec.state !== "inactive") rec.stop();
  };
  const watch = () => {
    if (!recording) return;
    const tNow = timeAt(playhead);
    const overtime = performance.now() - started > (tEnd * 3 + 5) * 1000;
    if (!playing || tNow >= tEnd - 1e-3 || overtime) {
      playing = false;
      showFrame(play.times.length - 1, false);
      renderer.render(scene, camera);
      finish();
      return;
    }
    requestAnimationFrame(watch);
  };
  requestAnimationFrame(watch);
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
      showFrame(playhead, false);
    }
  } else if (follow) {
    applyFollow(false);
  } else if (povUid != null) {
    applyPov();
  }
  if (povUid == null) controls.update();
  renderer.render(scene, camera);
}

async function loadBat() {
  try {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(`data/bat.glb?v=${Date.now()}`);
    batHolder.remove(batFallback);
    batFallback.geometry.dispose();
    batFallback.material.dispose();
    gltf.scene.traverse((obj) => {
      if (obj.isMesh) {
        obj.castShadow = false;
        obj.receiveShadow = false;
      }
    });
    batHolder.add(gltf.scene);
    batModelLen = BAT_MODEL_LEN_FT;
    batIsMesh = true;
    batHolder.visible = false;
  } catch (err) {
    console.warn("bat.glb failed, using cylinder", err);
  }
}

async function main() {
  const playUrl = (window.PLAY_META && window.PLAY_META.playId)
    ? `/api/play?id=${encodeURIComponent(window.PLAY_META.playId)}`
    : `/api/play?v=${Date.now()}`;
  play = await fetch(playUrl, { cache: "no-store" }).then((r) => {
    if (!r.ok) throw new Error("play JSON missing — run serve.py <play_dir>");
    return r.json();
  });
  document.title = `Gameday 3D  ${play.gamePk || ""}  ${play.playId || ""}`;
  const banner = document.getElementById("play-banner");
  if (banner) {
    const pit = playPitcher();
    banner.textContent = `${play.gamePk || ""}  ${play.playId || ""}  ${pit}`.trim();
  }
  $("s-time").max = 1;
  buildActors();
  buildViews();
  await loadPark(play.ballpark);
  await loadBat();
  bindUi();
  headPose = HEAD_POSES.has(play.headPose) ? play.headPose : "EASY_VISION";
  showFrame(0);
  snap("action");
  requestAnimationFrame(tick);
}

main().catch((err) => {
  hudEl.textContent = String(err);
  console.error(err);
});
