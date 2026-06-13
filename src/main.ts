/**
 * VinFast VFX Monitor – main.ts
 * Three.js scene loader + dashboard UI logic
 * Renderer: WebGPU  |  TSL reflection ground with hashBlur
 */

import './style.css';

// ── WebGPU + TSL imports ──────────────────────────────────────────────────────
import * as THREE from 'three/webgpu';
import {
  Fn,
  vec4,
  fract,
  abs,
  uniform,
  pow,
  color,
  max,
  length,
  rangeFogFactor,
  sub,
  reflector,
  time,
  mix,
  positionWorld,
  sample,
  float,
  sin,
  pass,
} from 'three/tsl';

import { hashBlur } from 'three/examples/jsm/tsl/display/hashBlur.js';
import { bloom } from 'three/examples/jsm/tsl/display/BloomNode.js';

// ── Standard addons ───────────────────────────────────────────────────────────
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';
import { FlakesTexture } from 'three/examples/jsm/textures/FlakesTexture.js';
import { Inspector } from 'three/examples/jsm/inspector/Inspector.js';

// ─────────────────────────────────────────────
//  SECTION 1 – Three.js Scene Setup
// ─────────────────────────────────────────────

const mount = document.getElementById('threejs-mount') as HTMLDivElement;

// Renderer – WebGPU
const renderer = new THREE.WebGPURenderer({
  antialias: true,
  powerPreference: 'high-performance',
});
await renderer.init();

// Initialize Inspector debug
const inspector = new Inspector();
document.body.appendChild(inspector.domElement);
(renderer as any).inspector = inspector;

renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.25;
// Shadow maps are not yet fully supported in WebGPU path; enable if your build supports it
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

mount.appendChild(renderer.domElement);


// Scene
const scene = new THREE.Scene();

// Camera – perspective matching a low studio angle
const camera = new THREE.PerspectiveCamera(
  35,
  mount.clientWidth / mount.clientHeight,
  0.01,
  1000,
);
camera.position.set(-10, 3, 10.5);
camera.lookAt(0, 0.5, 0);

// OrbitControls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.enablePan = false;
controls.minDistance = 2;
controls.maxDistance = 10;
controls.minPolarAngle = Math.PI * 0.15;
controls.maxPolarAngle = Math.PI * 0.50;
controls.target.set(0, 1, 0);
controls.autoRotate = false;
controls.autoRotateSpeed = 0.0;

// Cancel transitions on manual drag
controls.addEventListener('start', () => {
  targetCameraPos = null;
  targetControlsTarget = null;
});

// ── Post-processing (Bloom) Setup ───────────────────────────────────────────
const renderPipeline = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera);
const scenePassColor = scenePass.getTextureNode('output');

const bloomIntensity = uniform(0.15); // dynamically controlled by the sun dial
const bloomPass = bloom(scenePassColor, bloomIntensity as any, 0.35, 0.7);
// Params: inputNode, strength, radius, threshold

renderPipeline.outputNode = scenePassColor.add(bloomPass);

function triggerMaterialNeedsUpdate(): void {
  const currentEnvIntensity = scene.environmentIntensity;
  scene.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      const mesh = child as THREE.Mesh;
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      materials.forEach((mat) => {
        if (mat instanceof THREE.MeshStandardMaterial) {
          mat.envMapIntensity = currentEnvIntensity;
          mat.needsUpdate = true;
        }
      });
    }
  });
}

// ─────────────────────────────────────────────
//  SECTION 2 – Lighting & Environment
// ─────────────────────────────────────────────


const keyLight = new THREE.DirectionalLight(0xfff8ee, 0.5);
keyLight.position.set(5, 8, 6);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 30;
keyLight.shadow.camera.left = -6;
keyLight.shadow.camera.right = 6;
keyLight.shadow.camera.top = 6;
keyLight.shadow.camera.bottom = -6;
keyLight.shadow.bias = -0.0005;
keyLight.shadow.intensity = 3;
scene.add(keyLight);

// HDRI environment (with graceful fallback)
const rgbeLoader = new RGBELoader();
rgbeLoader.load(
  '/Textures/newman_locker_room_1k.hdr',
  (hdrTexture) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const pmrem = new THREE.PMREMGenerator(renderer as any);
    pmrem.compileEquirectangularShader();
    const envMap = pmrem.fromEquirectangular(hdrTexture).texture;
    scene.environment = envMap;
    scene.environmentIntensity = 0.35;
    triggerMaterialNeedsUpdate();
    hdrTexture.dispose();
    pmrem.dispose();
  },
  undefined
);

// ─────────────────────────────────────────────
//  SECTION 3 – GLB Model Loader
// ─────────────────────────────────────────────

let carModel: THREE.Object3D | null = null;
let modelReady = false;
let roofMixer: THREE.AnimationMixer | null = null;
let leftDoorMixer: THREE.AnimationMixer | null = null;
let rightDoorMixer: THREE.AnimationMixer | null = null;
let roofAction: THREE.AnimationAction | null = null;
let leftDoorAction: THREE.AnimationAction | null = null;
let rightDoorAction: THREE.AnimationAction | null = null;
let isRoofOpen = false;
let isLeftDoorOpen = false;
let isRightDoorOpen = false;

// ── Hotspot & Camera Transition Variables ──────────────────────────────
let targetCameraPos: THREE.Vector3 | null = null;
let targetControlsTarget: THREE.Vector3 | null = null;
const transitionSpeed = 0.05;

interface Hotspot {
  id: string;
  name: string;
  position: THREE.Vector3;
  element: HTMLDivElement;
}

let hotspots: Hotspot[] = [];
const hotspotsContainer = document.createElement('div');
hotspotsContainer.id = 'hotspots-container';
mount.appendChild(hotspotsContainer);

const loadingOverlay = createLoadingOverlay();
mount.appendChild(loadingOverlay);

const gltfLoader = new GLTFLoader();

gltfLoader.load('/Models/environment.glb', (gltf) => {
  const model = gltf.scene;
  model.traverse((child) => {
    console.log(child.name);
    if (child instanceof THREE.Mesh && child.name == "Plane014") {
      child.receiveShadow = true;
      child.castShadow = true;
    }

  });
  scene.add(model);
  model.position.set(0, 0, 0);
  triggerMaterialNeedsUpdate();
});
const normalMap3 = new THREE.CanvasTexture(new FlakesTexture());
normalMap3.wrapS = THREE.RepeatWrapping;
normalMap3.wrapT = THREE.RepeatWrapping;
normalMap3.repeat.x = 10;
normalMap3.repeat.y = 6;
normalMap3.anisotropy = 16;

let carPaintrmaterial = new THREE.MeshPhysicalMaterial({
  clearcoat: 1.0,
  clearcoatRoughness: 0.1,
  metalness: 0.9,
  roughness: 0.5,
  color: 0xe6c20e,
  normalMap: normalMap3,
  normalScale: new THREE.Vector2(0.15, 0.15),
  envMapIntensity: 1.0,
});

gltfLoader.load(
  '/Models/VinfastVFXCar.glb',
  (gltf) => {
    const model = gltf.scene;
    console.log('car', gltf);
    model.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        materials.forEach((mat) => {
          if (mat instanceof THREE.MeshStandardMaterial) {
            mat.envMapIntensity = 1.0;
            mat.needsUpdate = true;
          }
        });
        if (!Array.isArray(mesh.material) && mesh.material?.name === 'Procedural Car Paint') {
          mesh.material = carPaintrmaterial;
        }
      }
    });

    scene.add(model);
    carModel = model;
    modelReady = true;
    triggerMaterialNeedsUpdate();

    // Initialize hotspots based on the loaded model
    initHotspots(model);

    if (gltf.animations && gltf.animations.length > 0) {
      const filterClipTracks = (clip: THREE.AnimationClip, keywords: string[]): THREE.AnimationClip => {
        const filtered = clip.tracks.filter((track) => {
          const nameLower = track.name.toLowerCase();
          return keywords.some((kw) => nameLower.includes(kw.toLowerCase()));
        });
        return new THREE.AnimationClip(clip.name, clip.duration, filtered);
      };

      // Animation 0: Convertible Roof (Mixer 1)
      roofMixer = new THREE.AnimationMixer(model);
      const rawRoofClip = gltf.animations[3];
      const roofClip = filterClipTracks(rawRoofClip, ['cabo', 'top']);
      roofAction = roofMixer.clipAction(roofClip);
      roofAction.loop = THREE.LoopOnce;
      roofAction.clampWhenFinished = true;

      // Animation 1: Left Door (Mixer 2)
      if (gltf.animations.length > 4) {
        leftDoorMixer = new THREE.AnimationMixer(model);
        const rawLeftDoorClip = gltf.animations[4];
        const leftDoorClip = filterClipTracks(rawLeftDoorClip, ['door_l', 'doorfont_glass_l']);
        leftDoorAction = leftDoorMixer.clipAction(leftDoorClip);
        leftDoorAction.loop = THREE.LoopOnce;
        leftDoorAction.clampWhenFinished = true;
      }

      // Animation 2: Right Door (Mixer 3)
      if (gltf.animations.length > 5) {
        rightDoorMixer = new THREE.AnimationMixer(model);
        const rawRightDoorClip = gltf.animations[5];
        const rightDoorClip = filterClipTracks(rawRightDoorClip, ['door_r', 'doorfont_glass_r']);
        rightDoorAction = rightDoorMixer.clipAction(rightDoorClip);
        rightDoorAction.loop = THREE.LoopOnce;
        rightDoorAction.clampWhenFinished = true;
      }
    }

    loadingOverlay.style.opacity = '0';
    setTimeout(() => loadingOverlay.remove(), 600);
  },
  (progress) => {
    if (progress.total > 0) {
      const pct = (progress.loaded / progress.total) * 100;
      const bar = loadingOverlay.querySelector('.loading-bar-fill') as HTMLElement;
      if (bar) bar.style.width = `${pct}%`;
    }
  },
  (error) => {
    console.warn('GLB not found or failed to load:', error);
    const geometry = new THREE.BoxGeometry(4, 1.4, 2.2);
    const wireframe = new THREE.WireframeGeometry(geometry);
    const line = new THREE.LineSegments(
      wireframe,
      new THREE.LineBasicMaterial({ color: 0xD48F38, opacity: 0.4, transparent: true }),
    );
    line.position.y = 0.7;
    scene.add(line);
    carModel = line;
    modelReady = true;

    loadingOverlay.style.opacity = '0';
    setTimeout(() => loadingOverlay.remove(), 600);
  },
);

// ─────────────────────────────────────────────
//  SECTION 4 – Animated Reflection Ground (TSL)
// ─────────────────────────────────────────────

// ── Animated ripple circle effect ───────────────────────────────────────────
// TSL Fn receives individual args – NOT an array. Raw JS numbers must be float() nodes.
const drawCircle = Fn(([pos, radius, width, power, col, timer]: any[]) => {
  // Based on https://www.shadertoy.com/view/3tdSRn
  const dist1 = length(pos);
  dist1.assign(fract(dist1.mul(5.0).sub(fract(timer))));
  const dist2 = dist1.sub(radius);
  const intensity = pow(radius.div(abs(dist2).max(0.025)), width);
  const output = col.rgb.mul(intensity).mul(power).mul(max(sub(0.8, abs(dist2)), 0.0));
  return output;
});

const circleFadeY = positionWorld.y.mul(0.7).oneMinus().max(0);

// Oscillate between gold and bright green over time – each ring pulse cycles between the two
const goldColor = color(0xD4A017); // warm gold
const greenColor = color(0xA4AA17); // electric green
const colorT = sin(time.mul(1.2)).mul(0.5).add(0.5); // 0..1 sine wave
const animatedColor = mix(goldColor, greenColor, colorT);

// Call with individual TSL args – raw numbers wrapped in float() so .div()/.mul() work
const animatedCircle =
  drawCircle(
    positionWorld.xz.mul(0.05),
    float(0.5),
    float(0.8),
    float(0.01),
    animatedColor,
    time.mul(0.1),
  ).mul(circleFadeY);


// ── Reflector ────────────────────────────────────────────────────────────────
const roughness = uniform(0.5);
const radius = uniform(0.2);

const reflection = reflector({ resolutionScale: 0.8, depth: true, bounces: false });
const reflectionDepth = reflection.getDepthNode!();
reflection.target.rotateX(-Math.PI / 2);
scene.add(reflection.target);

// ── Floor material (TSL shader) ──────────────────────────────────────────────
const floorMaterial = new THREE.MeshStandardNodeMaterial();
floorMaterial.transparent = true;

floorMaterial.colorNode = Fn(() => {
  // Radius / roughness ranges
  const radiusRange = mix(0.01, 0.1, radius);   // [0.01, 0.10]
  const roughnessRange = mix(1, 0.08, roughness); // [0.03, 0.30]

  // Sample reflection + depth-mask the edges
  const maskReflection = sample((uv: any) => {
    const s = reflection.sample(uv);
    const mask = reflectionDepth.sample(uv);
    return vec4(s.rgb, s.a.mul(mask.r));
  }, reflection.uvNode);

  // Blur the reflection
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const reflectionBlurred = (hashBlur as any)(maskReflection, radiusRange, {
    repeats: 40,
    premultipliedAlpha: true,
  });

  // Composite
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const reflectionMask = (reflectionBlurred.a.mul(reflectionDepth) as any).remapClamp(0, roughnessRange);
  const reflectionIntensity = 0.15;
  const reflectionMixFactor = reflectionMask.mul(roughness.mul(2).min(1));
  const reflectionFinal = mix(
    reflection.rgb,
    reflectionBlurred.rgb,
    reflectionMixFactor,
  ).mul(reflectionIntensity);

  // Add animated ripple circles on top of reflection
  const outputColor = animatedCircle.add(reflectionFinal);

  // Distance-based opacity fade
  const opacity = rangeFogFactor(10.5, 15.5).oneMinus();

  return vec4(outputColor, opacity);
})();

// ── Floor mesh ───────────────────────────────────────────────────────────────
const floor = new THREE.Mesh(new THREE.BoxGeometry(50, 0.001, 50), floorMaterial);
floor.receiveShadow = true;

floor.position.set(0, 0, 0);
scene.add(floor);

// ─────────────────────────────────────────────
//  SECTION 5 – Resize Handler (Fixed 16:9 Container)
// ─────────────────────────────────────────────

function onResize(): void {
  const w = mount.clientWidth;
  const h = mount.clientHeight;
  if (w === 0 || h === 0) return;

  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}

const resizeObserver = new ResizeObserver(onResize);
resizeObserver.observe(mount);
onResize();

// ─────────────────────────────────────────────
//  SECTION 6 – Animation Loop
// ─────────────────────────────────────────────

const clock = new THREE.Clock();

// renderer.setAnimationLoop is preferred for WebGPU (async render)
renderer.setAnimationLoop(() => {
  const delta = clock.getDelta();
  if (roofMixer) roofMixer.update(delta);
  if (leftDoorMixer) leftDoorMixer.update(delta);
  if (rightDoorMixer) rightDoorMixer.update(delta);

  // Camera smooth transition (LERP)
  if (targetCameraPos) {
    camera.position.lerp(targetCameraPos, transitionSpeed);
    if (camera.position.distanceTo(targetCameraPos) < 0.01) {
      camera.position.copy(targetCameraPos);
      targetCameraPos = null;
    }
  }

  if (targetControlsTarget) {
    controls.target.lerp(targetControlsTarget, transitionSpeed);
    if (controls.target.distanceTo(targetControlsTarget) < 0.01) {
      controls.target.copy(targetControlsTarget);
      targetControlsTarget = null;
    }
  }

  controls.update();

  // Project 3D hotspot coordinates to 2D Screen
  if (hotspots.length > 0) {
    const tempV = new THREE.Vector3();
    const w = mount.clientWidth;
    const h = mount.clientHeight;

    hotspots.forEach((hotspot) => {
      tempV.copy(hotspot.position);
      tempV.project(camera);

      // Hide if hotspot is behind the camera
      if (tempV.z > 1) {
        hotspot.element.classList.add('hidden');
      } else {
        hotspot.element.classList.remove('hidden');
        const x = (tempV.x * 0.5 + 0.5) * w;
        const y = (-tempV.y * 0.5 + 0.5) * h;
        hotspot.element.style.transform = `translate(-50%, -50%) translate(${x}px, ${y}px)`;
      }
    });
  }

  renderPipeline.render();
});


// ─────────────────────────────────────────────
//  SECTION 8 – Sun & Time Dial Controller
// ─────────────────────────────────────────────

(function initTimeDial(): void {
  const dialSvg = document.getElementById('time-dial-svg') as SVGSVGElement | null;
  const dialHandle = document.getElementById('dial-handle') as SVGElement | null;
  const timeDisplay = document.getElementById('time-display');
  const timePeriod = document.getElementById('time-period');
  const sunIcon = document.getElementById('dial-sun-icon') as SVGElement | null;
  const moonIcon = document.getElementById('dial-moon-icon') as SVGElement | null;
  const ticksGroup = document.getElementById('dial-ticks') as SVGElement | null;

  if (!dialSvg || !dialHandle || !sunIcon || !moonIcon || !ticksGroup) {
    console.error('Time dial UI elements not found');
    return;
  }

  // Generate ticks dynamically
  const CX = 80, CY = 80, R = 62;
  for (let h = 0; h < 24; h++) {
    const angleRad = (h * 15 - 90) * Math.PI / 180;
    const isMajor = h % 6 === 0;

    // Draw tick line
    const startR = isMajor ? R - 6 : R - 4;
    const x1 = CX + R * Math.cos(angleRad);
    const y1 = CY + R * Math.sin(angleRad);
    const x2 = CX + startR * Math.cos(angleRad);
    const y2 = CY + startR * Math.sin(angleRad);

    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', String(x1));
    line.setAttribute('y1', String(y1));
    line.setAttribute('x2', String(x2));
    line.setAttribute('y2', String(y2));
    line.setAttribute('class', isMajor ? 'dial-tick major' : 'dial-tick');
    ticksGroup.appendChild(line);

    // Draw label for major ticks
    if (isMajor) {
      const labelR = R - 13;
      const lx = CX + labelR * Math.cos(angleRad);
      const ly = CY + labelR * Math.sin(angleRad);

      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', String(lx));
      text.setAttribute('y', String(ly));
      text.setAttribute('class', `dial-label label-${h}`);
      // Show "12h", "18h", "00h", "06h"
      text.textContent = h === 0 ? '00h' : (h < 10 ? `0${h}h` : `${h}h`);
      ticksGroup.appendChild(text);
    }
  }

  // Update light & UI based on hour (0 - 24 float)
  function updateSunTime(hour: number): void {
    // 1. Update Dial Knob rotation angle (degrees)
    // 12:00 -> 0 degrees, 18:00 -> 90 degrees, 00:00 -> 180 degrees, 06:00 -> 270 degrees
    const deg = (hour - 12) * 15;
    dialHandle!.setAttribute('transform', `rotate(${deg}, 80, 80)`);

    // 2. Update time display text
    const displayHour = Math.floor(hour);
    const displayMin = Math.floor((hour % 1) * 60);
    const period = displayHour >= 12 ? 'PM' : 'AM';
    let hour12 = displayHour % 12;
    if (hour12 === 0) hour12 = 12;
    const minStr = displayMin < 10 ? `0${displayMin}` : `${displayMin}`;

    if (timeDisplay) timeDisplay.textContent = `${hour12}:${minStr}`;
    if (timePeriod) timePeriod.textContent = period;

    // Highlight active quadrant label
    const labels = ticksGroup!.querySelectorAll('.dial-label');
    labels.forEach(l => l.classList.remove('active'));

    // Find closest major hour (0, 6, 12, 18)
    const closestMajor = [0, 6, 12, 18].reduce((prev, curr) => {
      const diffPrev = Math.min(Math.abs(hour - prev), 24 - Math.abs(hour - prev));
      const diffCurr = Math.min(Math.abs(hour - curr), 24 - Math.abs(hour - curr));
      return diffPrev < diffCurr ? prev : curr;
    });

    const activeLabel = ticksGroup!.querySelector(`.label-${closestMajor}`);
    if (activeLabel) activeLabel.classList.add('active');

    // 3. Toggle center icon (Sun during day 6-18, Moon during night)
    const isDay = hour >= 6 && hour < 18;
    if (isDay) {
      sunIcon!.style.display = 'block';
      moonIcon!.style.display = 'none';
    } else {
      sunIcon!.style.display = 'none';
      moonIcon!.style.display = 'block';
    }

    // 4. Update keyLight position and intensity
    // phi = (H - 6) * 2PI / 24. Sunrise (6) -> phi=0, Noon (12) -> phi=PI/2, Sunset (18) -> phi=PI
    const phi = (hour - 6) * (Math.PI * 2) / 24;

    // Position of the sun in the sky
    const R_orbit = 12;
    const sunY = R_orbit * Math.sin(phi);
    const sunX = R_orbit * Math.cos(phi);

    // Update keyLight position (keep Z constant to illuminate front/side nicely)
    keyLight.position.set(sunX, Math.max(0, sunY), 6);

    // Calculate strength factor (0 to 1 during the day, 0 at night)
    const sunFactor = Math.max(0, Math.sin(phi));

    // Smoothly scale keyLight intensity (max 0.7 at noon, 0 at night)
    keyLight.intensity = 0.7 * sunFactor;

    // 5. Adjust toneMappingExposure and environmentIntensity
    // Night exposure: 0.22, Noon exposure: 1.25
    renderer.toneMappingExposure = 0.5 + 1.03 * sunFactor;

    // Environment maps reflections: Night: 0.03, Noon: 0.3
    const envIntensity = 0.16 + 0.27 * sunFactor;
    scene.environmentIntensity = envIntensity;

    scene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        materials.forEach((mat) => {
          if (mat instanceof THREE.MeshStandardMaterial) {
            mat.envMapIntensity = envIntensity;
          }
        });
      }
    });

    // 6. Adjust bloom postprocessing intensity
    // Night bloom: 1.5, Day bloom: 0.0
    const nightFactor = 1.0 - sunFactor;
    console.log(nightFactor);
    bloomIntensity.value = 1.5 * nightFactor;
  }

  // Drag interaction logic
  let isDragging = false;

  function calculateHourFromCoord(clientX: number, clientY: number): number {
    const rect = dialSvg!.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    const dx = clientX - centerX;
    const dy = clientY - centerY;

    // Calculate angle in radians. atan2 gives angle from horizontal (right = 0)
    const angleRad = Math.atan2(dy, dx);
    const angleDeg = angleRad * 180 / Math.PI; // [-180, 180]

    // Rotate coordinate system so that top (12 o'clock) is 0 degrees
    let rotationDeg = angleDeg + 90;
    if (rotationDeg < 0) {
      rotationDeg += 360;
    }

    // Convert rotation degrees (0 - 360) to hour (0 - 24)
    // 360 deg = 24 hours -> 15 deg = 1 hour
    let hour = (12 + rotationDeg / 15) % 24;
    if (hour < 0) hour += 24;
    return hour;
  }

  function handleStart(e: MouseEvent | TouchEvent) {
    isDragging = true;
    document.getElementById('card-time-control')?.classList.add('dragging-dial');

    // Disable orbit controls while dragging
    controls.enabled = false;

    const clientX = 'touches' in e ? e.touches[0].clientX : (e as MouseEvent).clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : (e as MouseEvent).clientY;

    const newHour = calculateHourFromCoord(clientX, clientY);
    updateSunTime(newHour);
    e.preventDefault();
  }

  function handleMove(e: MouseEvent | TouchEvent) {
    if (!isDragging) return;

    const clientX = 'touches' in e ? e.touches[0].clientX : (e as MouseEvent).clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : (e as MouseEvent).clientY;

    const newHour = calculateHourFromCoord(clientX, clientY);
    updateSunTime(newHour);
    e.preventDefault();
  }

  function handleEnd() {
    if (!isDragging) return;
    isDragging = false;
    document.getElementById('card-time-control')?.classList.remove('dragging-dial');

    // Re-enable orbit controls
    controls.enabled = true;
  }

  // Bind events on the SVG dial container
  dialSvg.addEventListener('mousedown', handleStart);
  window.addEventListener('mousemove', handleMove);
  window.addEventListener('mouseup', handleEnd);

  dialSvg.addEventListener('touchstart', handleStart, { passive: false });
  window.addEventListener('touchmove', handleMove, { passive: false });
  window.addEventListener('touchend', handleEnd);

  // Initialize at 11:00 AM
  updateSunTime(14.0);
})();

// ─────────────────────────────────────────────
//  SECTION 9 – UI Utilities
// ─────────────────────────────────────────────

function createLoadingOverlay(): HTMLDivElement {
  const overlay = document.createElement('div');
  overlay.style.cssText = `
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 16px;
    background: rgba(10,10,12,0.75);
    backdrop-filter: blur(8px);
    z-index: 100;
    transition: opacity 0.5s ease;
    border-radius: inherit;
  `;

  const label = document.createElement('div');
  label.style.cssText = `
    font-family: 'Orbitron', monospace;
    font-size: 0.75em;
    color: #D48F38;
    letter-spacing: 0.2em;
    text-transform: uppercase;
  `;
  label.textContent = 'Connecting to VFX…';

  const barTrack = document.createElement('div');
  barTrack.style.cssText = `
    width: 30%; height: 2px;
    background: rgba(212,143,56,0.2);
    border-radius: 2px;
    overflow: hidden;
  `;

  const barFill = document.createElement('div');
  barFill.className = 'loading-bar-fill';
  barFill.style.cssText = `
    height: 100%; width: 0%;
    background: linear-gradient(90deg, #8B5E1A, #D48F38, #F4C96A);
    border-radius: 2px;
    transition: width 0.3s ease;
    box-shadow: 0 0 8px rgba(212,143,56,0.6);
  `;

  barTrack.appendChild(barFill);
  overlay.appendChild(label);
  overlay.appendChild(barTrack);
  return overlay;
}

// ─────────────────────────────────────────────
//  SECTION 10 – Clock & Live Updates
// ─────────────────────────────────────────────

function updateClock(): void {
  const el = document.querySelector('.header-time');
  if (!el) return;
  const now = new Date();
  const ampm = now.getHours() < 12 ? 'AM' : 'PM';
  const h12 = (now.getHours() % 12 || 12).toString().padStart(2, '0');
  const m = now.getMinutes().toString().padStart(2, '0');
  el.textContent = `${h12}:${m} ${ampm}`;
}

updateClock();
setInterval(updateClock, 60_000);

// ─────────────────────────────────────────────
//  SECTION 11 – Interactive Button Feedback
// ─────────────────────────────────────────────

const doorButtons = document.querySelectorAll<HTMLButtonElement>('.door-btn, .sq-btn');
doorButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    btn.style.transform = 'scale(0.96)';
    btn.style.boxShadow = '0 0 30px rgba(212, 143, 56, 0.5)';
    setTimeout(() => {
      btn.style.transform = '';
      btn.style.boxShadow = '';
    }, 200);
  });
});

// ─────────────────────────────────────────────
//  SECTION 11.1 – Open/Close Animations Control
// ─────────────────────────────────────────────

function toggleDoorOrRoof(
  action: THREE.AnimationAction | null,
  btnId: string,
  isOpen: boolean,
  openText: string,
  closeText: string
): boolean {
  if (!action) return isOpen;

  const nextOpen = !isOpen;
  const btn = document.getElementById(btnId);
  const label = btn?.querySelector('.btn-label');
  if (label) {
    label.textContent = nextOpen ? closeText : openText;
  }

  const duration = action.getClip().duration;
  action.paused = false;
  action.enabled = true;

  if (nextOpen) {
    action.timeScale = 1;
    if (action.time >= duration) {
      action.time = 0;
    }
  } else {
    action.timeScale = -1;
    if (action.time <= 0) {
      action.time = duration;
    }
  }
  action.play();

  return nextOpen;
}

const btnRoof = document.getElementById('btn-roof') as HTMLButtonElement | null;
if (btnRoof) {
  btnRoof.addEventListener('click', () => {
    isRoofOpen = toggleDoorOrRoof(roofAction, 'btn-roof', isRoofOpen, 'Open Roof', 'Close Roof');
  });
}

const btnLeftDoor = document.getElementById('btn-door-left') as HTMLButtonElement | null;
if (btnLeftDoor) {
  btnLeftDoor.addEventListener('click', () => {
    isLeftDoorOpen = toggleDoorOrRoof(leftDoorAction, 'btn-door-left', isLeftDoorOpen, 'Open Left Door', 'Close Left Door');
  });
}

const btnRightDoor = document.getElementById('btn-door-right') as HTMLButtonElement | null;
if (btnRightDoor) {
  btnRightDoor.addEventListener('click', () => {
    isRightDoorOpen = toggleDoorOrRoof(rightDoorAction, 'btn-door-right', isRightDoorOpen, 'Open Right Door', 'Close Right Door');
  });
}

// ─────────────────────────────────────────────
//  SECTION 12 – OrbitControls Auto-rotate Resume
// ─────────────────────────────────────────────

let autoRotateTimer: ReturnType<typeof setTimeout>;

renderer.domElement.addEventListener('pointerdown', () => {
  controls.autoRotate = false;
  clearTimeout(autoRotateTimer);
});

renderer.domElement.addEventListener('pointerup', () => {
  autoRotateTimer = setTimeout(() => {
    controls.autoRotate = true;
  }, 4000);
});

controls.touches = {
  ONE: THREE.TOUCH.ROTATE,
  TWO: THREE.TOUCH.DOLLY_ROTATE,
};

// Keep track of carModel usage to avoid lint errors
void modelReady;
void carModel;



console.info('%c VinFast VFX Monitor – WebGPU Loaded', 'color:#D48F38;font-weight:bold;font-size:14px;');

// ── Hotspot System Initialization Function ──────────────────────────
function initHotspots(model: THREE.Object3D): void {
  // Clear any existing hotspots
  hotspotsContainer.innerHTML = '';
  hotspots = [];

  const box = new THREE.Box3().setFromObject(model);
  const center = new THREE.Vector3();
  box.getCenter(center);
  const size = new THREE.Vector3();
  box.getSize(size);

  let steerPos: THREE.Vector3 | null = null;
  let frontPos: THREE.Vector3 | null = null;
  let rearPos: THREE.Vector3 | null = null;

  model.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      const name = child.name.toLowerCase();
      // Find steering wheel
      if (!steerPos && (name.includes('steering') || name.includes('steer') || name.includes('volant'))) {
        steerPos = new THREE.Vector3();
        child.getWorldPosition(steerPos);
      }
      // Find headlight / front bumpers
      if (!frontPos && (name.includes('headlight') || name.includes('grille') || name.includes('hood') || name.includes('bumper_f') || name.includes('lamp_f') || name.includes('capo'))) {
        frontPos = new THREE.Vector3();
        child.getWorldPosition(frontPos);
      }
      // Find taillight / rear bumpers / exhaust
      if (!rearPos && (name.includes('taillight') || name.includes('trunk') || name.includes('bumper_r') || name.includes('lamp_r') || name.includes('exhaust'))) {
        rearPos = new THREE.Vector3();
        child.getWorldPosition(rearPos);
      }
    }
  });

  const isZLong = size.z >= size.x;
  const longAxis = isZLong ? 'z' : 'x';
  let frontDirection = 1;

  if (frontPos && rearPos) {
    frontDirection = frontPos[longAxis] > rearPos[longAxis] ? 1 : -1;
  } else if (steerPos) {
    frontDirection = steerPos[longAxis] > center[longAxis] ? 1 : -1;
  } else {
    frontDirection = 1; // Default to +Z or +X as front
  }

  // Define Hotspot Positions
  const finalSteer = steerPos || new THREE.Vector3(
    center.x + (isZLong ? 0.35 : 0),
    center.y + 0.3,
    center.z + (isZLong ? 0.4 * frontDirection : 0.35 * frontDirection)
  );

  const finalFront = new THREE.Vector3(center.x, center.y + 0.1, 0);
  if (frontDirection === 1) {
    finalFront[longAxis] = box.max[longAxis] - 0.1;
  } else {
    finalFront[longAxis] = box.min[longAxis] + 0.1;
  }
  if (frontPos) {
    finalFront.x = (frontPos as any).x;
    finalFront.y = (frontPos as any).y;
  }

  const finalRear = new THREE.Vector3(center.x, center.y + 0.1, 0);
  if (frontDirection === 1) {
    finalRear[longAxis] = box.min[longAxis] + 0.1;
  } else {
    finalRear[longAxis] = box.max[longAxis] - 0.1;
  }
  if (rearPos) {
    finalRear.x = (rearPos as any).x;
    finalRear.y = (rearPos as any).y;
  }

  const hotspotData = [
    { id: 'front', name: 'Front View', position: finalFront },
    { id: 'rear', name: 'Rear View', position: finalRear },
    { id: 'steer', name: 'Steering Wheel', position: finalSteer },
  ];

  hotspotData.forEach((data) => {
    const el = document.createElement('div');
    el.className = 'hotspot';
    el.innerHTML = `
      <div class="hotspot-ring"></div>
      <div class="hotspot-dot"></div>
      <div class="hotspot-label">${data.name}</div>
    `;

    el.addEventListener('click', () => {
      // Calculate best camera angle
      let camPos = new THREE.Vector3();
      let targetPos = new THREE.Vector3();

      if (data.id === 'front') {
        targetPos.copy(finalFront);
        if (isZLong) {
          camPos.set(finalFront.x - 2.5, finalFront.y + 1.2, finalFront.z + 3.2 * frontDirection);
        } else {
          camPos.set(finalFront.x + 3.2 * frontDirection, finalFront.y + 1.2, finalFront.z - 2.5);
        }
      } else if (data.id === 'rear') {
        targetPos.copy(finalRear);
        if (isZLong) {
          camPos.set(finalRear.x + 2.5, finalRear.y + 1.2, finalRear.z - 3.2 * frontDirection);
        } else {
          camPos.set(finalRear.x - 3.2 * frontDirection, finalRear.y + 1.2, finalRear.z + 2.5);
        }
      } else if (data.id === 'steer') {
        targetPos.copy(finalSteer);
        // Position camera to look at the steering wheel from slightly above/outside
        if (isZLong) {
          const xOffset = finalSteer.x > center.x ? 1.4 : -1.4;
          camPos.set(finalSteer.x + xOffset, finalSteer.y + 0.8, finalSteer.z + 1.0 * frontDirection);
        } else {
          const zOffset = finalSteer.z > center.z ? 1.4 : -1.4;
          camPos.set(finalSteer.x + 1.0 * frontDirection, finalSteer.y + 0.8, finalSteer.z + zOffset);
        }
      }

      targetCameraPos = camPos;
      targetControlsTarget = targetPos;
    });

    hotspotsContainer.appendChild(el);
    hotspots.push({
      id: data.id,
      name: data.name,
      position: data.position,
      element: el,
    });
  });
}
