import * as THREE from 'three';

const mount = document.querySelector('#scene');
const ui = {
	speed: document.querySelector('#speed-value'),
	target: document.querySelector('#target-value'),
	distance: document.querySelector('#distance-value'),
	episode: document.querySelector('#episode-value'),
	sensorMetric: document.querySelector('.metric-sensors'),
	sensorValue: document.querySelector('#sensor-value'),
	alert: document.querySelector('#scene-alert'),
	radar: document.querySelector('#radar-distance'),
	radarState: document.querySelector('#radar-state'),
	clearance: document.querySelector('#clearance-value'),
	clearanceFill: document.querySelector('#clearance-fill'),
	steer: document.querySelector('#steer-value'),
	steerIndicator: document.querySelector('#steer-indicator'),
	throttle: document.querySelector('#throttle-slider'),
	throttleValue: document.querySelector('#throttle-value'),
	pause: document.querySelector('#pause-toggle'),
	reset: document.querySelector('#reset-button'),
	frame: document.querySelector('#frame-status'),
	best: document.querySelector('#best-distance'),
	collisions: document.querySelector('#collision-count'),
};

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x829080, 72, 220);
scene.background = new THREE.Color(0x829080);

const camera = new THREE.PerspectiveCamera(43, 1, 0.1, 320);
camera.position.set(0, 9.5, 19);
camera.lookAt(0, 0, -20);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
mount.appendChild(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xdce8dc, 0x27342c, 2.0));
const sun = new THREE.DirectionalLight(0xffedce, 3.2);
sun.position.set(-15, 28, 8);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
sun.shadow.camera.left = -45;
sun.shadow.camera.right = 45;
sun.shadow.camera.top = 40;
sun.shadow.camera.bottom = -40;
scene.add(sun);

const groundMaterial = new THREE.MeshStandardMaterial({ color: 0x526451, roughness: 1 });
const ground = new THREE.Mesh(new THREE.PlaneGeometry(500, 500), groundMaterial);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.22;
ground.position.z = -80;
ground.receiveShadow = true;
scene.add(ground);

const roadMaterial = new THREE.MeshStandardMaterial({ color: 0x343b3c, roughness: 0.92 });
const shoulderMaterial = new THREE.MeshStandardMaterial({ color: 0x55594c, roughness: 1 });
const curbMaterials = [
	new THREE.MeshStandardMaterial({ color: 0xd9775e, roughness: 0.82 }),
	new THREE.MeshStandardMaterial({ color: 0xd9ddd0, roughness: 0.82 }),
];
const centerMaterial = new THREE.MeshStandardMaterial({ color: 0xe1e4d7, roughness: 0.8 });
const roadSegments = 180;
const roadLength = 270;
const roadWidth = 12;
const segmentLength = roadLength / roadSegments;
const roadBehind = 40;
const roadAhead = roadLength - roadBehind;
const safeLateralLimit = roadWidth / 2 - 1.5;
const roadRoot = new THREE.Group();
const roadStrips = [];
scene.add(roadRoot);

function trackCenter(z) {
	return Math.sin((z + 18) * 0.024) * 13 + Math.sin((z - 10) * 0.061) * 3.3;
}

function stripMesh(side, innerOffset, width, material, y) {
	const positions = new Float32Array((roadSegments + 1) * 2 * 3);
	const indices = [];
	for (let index = 0; index <= roadSegments; index += 1) {
		const offset = roadBehind - index * segmentLength;
		const center = trackCenter(offset);
		const vertex = index * 6;
		positions.set([
			center + side * innerOffset, y, 5 + offset,
			center + side * (innerOffset + width), y, 5 + offset,
		], vertex);
		if (index < roadSegments) {
			const vertex = index * 2;
			indices.push(vertex, vertex + 1, vertex + 2, vertex + 1, vertex + 3, vertex + 2);
		}
	}
	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
	geometry.setIndex(indices);
	geometry.computeVertexNormals();
	const mesh = new THREE.Mesh(geometry, material);
	mesh.receiveShadow = true;
	roadStrips.push({ geometry, side, innerOffset, width });
	return mesh;
}

roadRoot.add(stripMesh(-1, 0, roadWidth / 2, roadMaterial, 0));
roadRoot.add(stripMesh(1, 0, roadWidth / 2, roadMaterial, 0));
roadRoot.add(stripMesh(-1, roadWidth / 2, 2.5, shoulderMaterial, 0.005));
roadRoot.add(stripMesh(1, roadWidth / 2, 2.5, shoulderMaterial, 0.005));

function createRoadMarkings() {
	const group = new THREE.Group();
	group.userData.movingFeatures = [];
	const dashGeometry = new THREE.BoxGeometry(0.12, 0.035, 3.1);
	for (let z = 36; z > -roadLength + 45; z -= 9) {
		const dash = new THREE.Mesh(dashGeometry, centerMaterial);
		dash.userData.trackZ = z - 5;
		dash.userData.laneOffset = 0;
		dash.position.set(trackCenter(dash.userData.trackZ), 0.035, z);
		group.add(dash);
		group.userData.movingFeatures.push(dash);
	}
	const edgeGeometry = new THREE.BoxGeometry(0.12, 0.04, 0.9);
	for (let z = 42; z > -roadLength + 45; z -= 1.8) {
		const stripeMaterial = curbMaterials[Math.floor((42 - z) / 1.8) % 2];
		for (const side of [-1, 1]) {
			const curb = new THREE.Mesh(edgeGeometry, stripeMaterial);
			curb.userData.trackZ = z - 5;
			curb.userData.laneOffset = side * (roadWidth / 2 - 0.08);
			curb.position.set(trackCenter(curb.userData.trackZ) + curb.userData.laneOffset, 0.045, z);
			group.add(curb);
			group.userData.movingFeatures.push(curb);
		}
	}
	return group;
}
const roadMarkings = createRoadMarkings();
scene.add(roadMarkings);
const scenery = [];

function makeTree(x, z, size) {
	const group = new THREE.Group();
	const trunk = new THREE.Mesh(
		new THREE.CylinderGeometry(size * 0.08, size * 0.12, size * 0.65, 6),
		new THREE.MeshStandardMaterial({ color: 0x554536, roughness: 1 }),
	);
	trunk.position.y = size * 0.32;
	group.add(trunk);
	for (let tier = 0; tier < 3; tier += 1) {
		const radius = size * (0.42 - tier * 0.095);
		const leaves = new THREE.Mesh(
			new THREE.ConeGeometry(radius, size * 0.72, 7),
			new THREE.MeshStandardMaterial({
				color: [0x2d4738, 0x3c5a43, 0x526c4c][tier],
				roughness: 1,
			}),
		);
		leaves.position.y = size * (0.72 + tier * 0.43);
		leaves.castShadow = true;
		group.add(leaves);
	}
	group.position.set(x, 0, z);
	return group;
}

for (let z = 40; z > -roadLength; z -= 13) {
	const distance = -z;
	const center = trackCenter(z);
	for (const side of [-1, 1]) {
		const jitter = Math.sin(distance * 7.17 + side) * 5;
		const tree = makeTree(
			center + side * (roadWidth / 2 + 5 + Math.abs(jitter)),
			z,
			2.3 + ((Math.sin(distance * 0.29) + 1) * 1.5),
		);
		tree.userData.trackZ = z - 5;
		tree.userData.laneOffset = side * (roadWidth / 2 + 5 + Math.abs(jitter));
		scene.add(tree);
		scenery.push(tree);
	}
}

const cars = [];
const carColors = [0x79b7c8, 0xd8b967, 0xd67a65, 0x94a7d0, 0xe5e0d1];
function trafficStartDistance(index, offset = 0) {
	return 35 + index * 32 + offset;
}

function makeCar(color, isPlayer = false) {
	const car = new THREE.Group();
	const paint = new THREE.MeshStandardMaterial({
		color,
		metalness: 0.24,
		roughness: 0.42,
	});
	const darkPaint = new THREE.MeshStandardMaterial({ color: 0x17201d, roughness: 0.52 });
	const glass = new THREE.MeshStandardMaterial({
		color: 0x8eb9be,
		metalness: 0.12,
		roughness: 0.2,
		transparent: true,
		opacity: 0.9,
	});
	const body = new THREE.Mesh(new THREE.BoxGeometry(1.85, 0.5, 3.8), paint);
	body.position.y = 0.63;
	body.castShadow = true;
	car.add(body);
	const hood = new THREE.Mesh(new THREE.BoxGeometry(1.68, 0.22, 1.0), paint);
	hood.position.set(0, 0.9, -1.17);
	hood.castShadow = true;
	car.add(hood);
	const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.38, 0.64, 1.75), darkPaint);
	cabin.position.set(0, 1.07, 0.12);
	cabin.castShadow = true;
	car.add(cabin);
	const windshield = new THREE.Mesh(new THREE.BoxGeometry(1.26, 0.43, 0.08), glass);
	windshield.position.set(0, 1.18, -0.57);
	windshield.rotation.x = -0.22;
	car.add(windshield);
	const rearGlass = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.36, 0.07), glass);
	rearGlass.position.set(0, 1.15, 0.83);
	rearGlass.rotation.x = 0.26;
	car.add(rearGlass);
	const wheels = [];
	const wheelGeometry = new THREE.CylinderGeometry(0.34, 0.34, 0.2, 12);
	const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x111514, roughness: 0.83 });
	for (const x of [-0.94, 0.94]) {
		for (const z of [-1.18, 1.2]) {
			const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
			wheel.rotation.z = Math.PI / 2;
			wheel.position.set(x, 0.36, z);
			wheel.castShadow = true;
			car.add(wheel);
			wheels.push(wheel);
		}
	}
	const lampMaterial = new THREE.MeshStandardMaterial({
		color: isPlayer ? 0xffedc4 : 0xffe4af,
		emissive: 0xf2c78a,
		emissiveIntensity: 0.35,
	});
	for (const x of [-0.58, 0.58]) {
		const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.12, 0.08), lampMaterial);
		lamp.position.set(x, 0.72, -1.92);
		car.add(lamp);
	}
	const tailMaterial = new THREE.MeshStandardMaterial({ color: 0xdb5645, emissive: 0x71180e, emissiveIntensity: 0.5 });
	for (const x of [-0.62, 0.62]) {
		const tail = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.11, 0.07), tailMaterial);
		tail.position.set(x, 0.69, 1.93);
		car.add(tail);
	}
	const shadow = new THREE.Mesh(
		new THREE.PlaneGeometry(2.7, 4.8),
		new THREE.MeshBasicMaterial({ color: 0x101411, transparent: true, opacity: 0.42, depthWrite: false }),
	);
	shadow.rotation.x = -Math.PI / 2;
	shadow.position.y = 0.025;
	car.add(shadow);
	car.userData.wheels = wheels;
	return car;
}

const playerCar = makeCar(0xe97961, true);
playerCar.position.set(trackCenter(5), 0, 5);
scene.add(playerCar);

const state = {
	distance: 0,
	speed: 20,
	safeSpeed: 22,
	lateral: 0,
	steering: 0,
	targetSteering: 0,
	throttle: 0.62,
	paused: false,
	manualUntil: 0,
	avoidanceCar: null,
	avoidanceTarget: null,
	episode: 1,
	collisions: 0,
	best: Number(localStorage.getItem('apex-best-distance') || 0),
	nearby: 24,
	hazard: false,
	lastFrame: performance.now(),
	frameTimes: [],
	droppedTimestamp: 0,
};

function spawnTraffic(index, offset = 0) {
	const car = makeCar(carColors[index % carColors.length]);
	car.userData.speed = 5.55 + (index % 4) * 0.22;
	car.userData.lane = [-3.4, 0, 3.4][index % 3];
	car.userData.distance = trafficStartDistance(index, offset);
	car.userData.index = index;
	car.position.set(trackCenter(-car.userData.distance) + car.userData.lane, 0, 5 - car.userData.distance * 1.1);
	scene.add(car);
	cars.push(car);
}
for (let index = 0; index < 14; index += 1) spawnTraffic(index);

const clock = new THREE.Clock();
let engineContext = null;
let engineOscillator = null;
let engineGain = null;
let soundEnabled = false;

function updateEngineSound() {
	if (!engineContext || !engineOscillator || !engineGain) return;
	const frequency = 52 + state.speed * 2.1 + Math.max(0, state.throttle) * 45;
	engineOscillator.frequency.setTargetAtTime(frequency, engineContext.currentTime, 0.12);
	engineGain.gain.setTargetAtTime(state.paused ? 0 : 0.025 + state.throttle * 0.022, engineContext.currentTime, 0.15);
}

function setSoundEnabled(enabled) {
	soundEnabled = enabled;
	if (enabled && !engineContext) {
		const AudioContextType = window.AudioContext || window.webkitAudioContext;
		if (!AudioContextType) return;
		engineContext = new AudioContextType();
		engineOscillator = engineContext.createOscillator();
		engineGain = engineContext.createGain();
		engineOscillator.type = 'sawtooth';
		engineOscillator.connect(engineGain).connect(engineContext.destination);
		engineOscillator.start();
	}
	if (engineContext?.state === 'suspended') engineContext.resume();
	updateEngineSound();
	document.querySelector('#sound-toggle').setAttribute('aria-pressed', String(enabled));
}

function resize() {
	const width = Math.max(1, mount.clientWidth);
	const height = Math.max(1, mount.clientHeight);
	renderer.setSize(width, height, false);
	camera.aspect = width / height;
	camera.fov = width < 540 ? 49 : 43;
	camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(mount);
resize();

function smooth(current, target, rate, dt) {
	return THREE.MathUtils.damp(current, target, rate, dt);
}

function nearestCar() {
	let nearestGap = Infinity;
	let nearestLaneGap = Infinity;
	let nearestLane = null;
	let nearestLaneSpeed = 0;
	let nearestLaneCar = null;
	for (const car of cars) {
		const gap = car.userData.distance - state.distance;
		if (gap <= 0) continue;
		nearestGap = Math.min(nearestGap, gap);
		if (Math.abs(car.userData.lane - state.lateral) < 3 && gap < nearestLaneGap) {
			nearestLaneGap = gap;
			nearestLane = car.userData.lane;
			nearestLaneSpeed = car.userData.speed;
			nearestLaneCar = car;
		}
	}
	return {
		gap: nearestGap,
		laneGap: nearestLaneGap,
		lane: nearestLane,
		laneSpeed: nearestLaneSpeed,
		laneCar: nearestLaneCar,
	};
}

function autonomousControl(trafficAhead, dt) {
	const curve = Math.sin((state.distance + 16) * 0.024) * 0.4;
	let steeringTarget = THREE.MathUtils.clamp(-state.lateral * 0.1 - state.steering * 0.55 - curve, -0.6, 0.6);
	if (state.avoidanceCar && state.avoidanceCar.userData.distance - state.distance <= -6) {
		state.avoidanceCar = null;
		state.avoidanceTarget = null;
	}
	if (!state.avoidanceCar && trafficAhead.laneGap < 30) {
		const candidates = [-3.4, 0, 3.4]
			.filter((lane) => Math.abs(lane - trafficAhead.lane) > 2.6 && Math.abs(lane) <= safeLateralLimit - 0.5)
			.map((lane) => {
				const gap = cars.reduce((nearest, car) => {
					const carGap = car.userData.distance - state.distance;
					return car !== trafficAhead.laneCar
						&& carGap > 0
						&& Math.abs(car.userData.lane - lane) < 1.5
						? Math.min(nearest, carGap)
						: nearest;
				}, Infinity);
				return { lane, gap };
			})
			.filter((candidate) => candidate.gap > 30)
			.sort((left, right) => right.gap - left.gap);
		if (candidates.length) {
			state.avoidanceCar = trafficAhead.laneCar;
			state.avoidanceTarget = candidates[0].lane;
		}
	}
	if (state.avoidanceCar) {
		steeringTarget = THREE.MathUtils.clamp(
			(state.avoidanceTarget - state.lateral) * 0.45 - state.steering * 0.5,
			-0.8,
			0.8,
		);
	}
	state.targetSteering = steeringTarget;
	const curveLimit = 22 - Math.min(7, Math.abs(curve) * 12);
	const trafficLimit = trafficAhead.laneGap < 30
		? Math.max(11.2, trafficAhead.laneSpeed + Math.max(0, trafficAhead.laneGap - 6) * 0.25)
		: 32;
	state.safeSpeed = Math.max(11.2, Math.min(curveLimit, trafficLimit));
	const throttleDemand = THREE.MathUtils.clamp((state.safeSpeed - state.speed) * 0.4, -1, 1);
	state.throttle = smooth(state.throttle, throttleDemand, 2.2, dt);
}

function collidesWithTraffic(car, startDistance, startLateral, endDistance, endLateral) {
	const relativeStart = new THREE.Vector2(startLateral - car.userData.lane, car.userData.distance - startDistance);
	const relativeEnd = new THREE.Vector2(endLateral - car.userData.lane, car.userData.distance - endDistance);
	const radiusX = 2.2;
	const radiusZ = 4.1;
	const delta = relativeEnd.clone().sub(relativeStart);
	const lengthSq = delta.lengthSq();
	const t = lengthSq === 0 ? 0 : THREE.MathUtils.clamp(-relativeStart.dot(delta) / lengthSq, 0, 1);
	const closest = relativeStart.addScaledVector(delta, t);
	return (closest.x / radiusX) ** 2 + (closest.y / radiusZ) ** 2 <= 1;
}

function update(dt) {
	const startDistance = state.distance;
	const startLateral = state.lateral;
	if (performance.now() > state.manualUntil && manualSteeringInputs.size === 0) {
		autonomousControl(nearestCar(), dt);
	}
	state.steering = smooth(state.steering, state.targetSteering, 2.2, dt);
	state.steering = THREE.MathUtils.clamp(state.steering, -0.9, 0.9);
	state.lateral += state.steering * dt * 4.1;
	state.lateral = THREE.MathUtils.clamp(state.lateral, -safeLateralLimit, safeLateralLimit);
	state.speed = THREE.MathUtils.clamp(state.speed + state.throttle * 5.3 * dt, 11.2, 32);
	state.distance += state.speed * dt;

	let crash = false;
	for (const car of cars) {
		car.userData.distance += car.userData.speed * dt;
		const gap = car.userData.distance - state.distance;
		car.position.x = trackCenter(-car.userData.distance) + car.userData.lane;
		car.position.z = 5 - gap * 1.1;
		if (collidesWithTraffic(car, startDistance, startLateral, state.distance, state.lateral)) crash = true;
	}
	if (crash) {
		state.collisions += 1;
		state.episode += 1;
		state.distance = 0;
		state.lateral = 0;
		state.speed = 11.2;
		state.steering = 0;
		state.targetSteering = 0;
		state.avoidanceCar = null;
		state.avoidanceTarget = null;
		for (const car of cars) {
			car.userData.distance += 90 + Math.random() * 35;
		}
	}
	for (const car of cars) {
		if (car.userData.distance - state.distance < -18) {
			car.userData.distance += 220;
			car.userData.lane = [-3.4, 0, 3.4][Math.floor(Math.random() * 3)];
		}
		car.position.x = trackCenter(-car.userData.distance) + car.userData.lane;
		car.position.z = 5 - (car.userData.distance - state.distance) * 1.1;
	}

	playerCar.position.x = trackCenter(-state.distance) + state.lateral;
	playerCar.position.z = 5;
	playerCar.rotation.y = -Math.atan2(state.steering * 4.1, Math.max(11.2, state.speed));
	for (const wheel of playerCar.userData.wheels) wheel.rotation.x += state.speed * dt * 0.45;
	for (const car of cars) {
		for (const wheel of car.userData.wheels) wheel.rotation.x += car.userData.speed * dt * 0.45;
	}

	updateTrackView();
	const curve = trackCenter(-state.distance - 18) - trackCenter(-state.distance);
	const cameraX = trackCenter(-state.distance) + state.lateral * 0.25 + curve * 0.15;
	camera.position.x = smooth(camera.position.x, cameraX, 2.5, dt);
	camera.position.y = smooth(camera.position.y, 9.2 + Math.abs(curve) * 0.035, 2, dt);
	camera.position.z = smooth(camera.position.z, 19, 2, dt);
	camera.lookAt(trackCenter(-state.distance - 24) + state.lateral * 0.1, 0.3, -23);
	ground.position.x = trackCenter(-state.distance - 60) * 0.3;

	const nearest = nearestCar();
	state.nearby = nearest.gap;
	state.hazard = nearest.laneGap < 18;
	updateHud();
	updateEngineSound();
}

function updateTrackView() {
	for (const strip of roadStrips) {
		const positions = strip.geometry.attributes.position;
		for (let index = 0; index <= roadSegments; index += 1) {
			const offset = roadBehind - index * segmentLength;
			const center = trackCenter(-state.distance + offset);
			const vertex = index * 2;
			positions.setXYZ(vertex, center + strip.side * strip.innerOffset, positions.getY(vertex), 5 + offset);
			positions.setXYZ(vertex + 1, center + strip.side * (strip.innerOffset + strip.width), positions.getY(vertex + 1), 5 + offset);
		}
		positions.needsUpdate = true;
		strip.geometry.computeVertexNormals();
	}

	for (const feature of roadMarkings.userData.movingFeatures) {
		let offset = feature.userData.trackZ + state.distance;
		while (offset > roadBehind) {
			feature.userData.trackZ -= roadLength;
			offset -= roadLength;
		}
		while (offset < -roadAhead) {
			feature.userData.trackZ += roadLength;
			offset += roadLength;
		}
		feature.position.x = trackCenter(-state.distance + offset) + feature.userData.laneOffset;
		feature.position.z = 5 + offset;
	}

	for (const tree of scenery) {
		let offset = tree.userData.trackZ + state.distance;
		while (offset > roadBehind) {
			tree.userData.trackZ -= roadLength;
			offset -= roadLength;
		}
		while (offset < -roadAhead) {
			tree.userData.trackZ += roadLength;
			offset += roadLength;
		}
		tree.position.x = trackCenter(-state.distance + offset) + tree.userData.laneOffset;
		tree.position.z = 5 + offset;
	}
}

function updateHud() {
	ui.speed.textContent = String(Math.round(state.speed * 3.6)).padStart(3, '0');
	ui.target.textContent = String(Math.round(state.safeSpeed * 3.6)).padStart(3, '0');
	ui.distance.textContent = state.distance.toFixed(1).padStart(5, '0');
	ui.episode.textContent = String(state.episode).padStart(3, '0');
	ui.sensorValue.textContent = state.hazard ? 'ALERT' : 'CLEAR';
	ui.sensorMetric.classList.toggle('alert', state.hazard);
	ui.alert.classList.toggle('danger', state.hazard);
	ui.alert.querySelector('strong').textContent = state.hazard ? 'PROXIMITY ALERT' : 'SENSORS CLEAR';
	ui.alert.querySelector('small').textContent = state.hazard ? 'COLLISION PATH DETECTED' : 'FORWARD RADAR ACTIVE';
	ui.radar.textContent = Number.isFinite(state.nearby) ? state.nearby.toFixed(1) : '—';
	ui.radarState.textContent = state.hazard ? 'BRAKING' : 'CLEAR';
	ui.radarState.style.color = state.hazard ? 'var(--coral)' : 'var(--lime)';
	const clearance = Math.max(0, Math.min(1, (safeLateralLimit - Math.abs(state.lateral)) / safeLateralLimit));
	ui.clearance.textContent = `${Math.round(clearance * 100)}%`;
	ui.clearanceFill.style.width = `${clearance * 100}%`;
	ui.steer.textContent = `${state.steering >= 0 ? '+' : '−'}${String(Math.round(Math.abs(state.steering) * 30)).padStart(2, '0')}°`;
	ui.steerIndicator.style.left = `${50 + state.steering * 45}%`;
	ui.best.textContent = state.best ? state.best.toFixed(0) : '—';
	ui.collisions.textContent = String(state.collisions).padStart(2, '0');
	const throttlePercent = Math.round(THREE.MathUtils.clamp((state.throttle + 0.25) / 1.3, 0, 1) * 100);
	ui.throttleValue.textContent = `${throttlePercent}%`;
	ui.throttle.value = String(throttlePercent);
	ui.throttle.style.background = `linear-gradient(90deg, #c7e27b ${throttlePercent}%, #354239 ${throttlePercent}%)`;
}

function render(now) {
	requestAnimationFrame(render);
	state.frameTimes.push(now);
	while (state.frameTimes.length && now - state.frameTimes[0] > 1000) state.frameTimes.shift();
	ui.frame.textContent = `${state.frameTimes.length} FPS / WEBGL`;
	const dt = Math.min(clock.getDelta(), 0.05);
	if (!state.paused) update(dt);
	renderer.render(scene, camera);
}

function applyManualSteer(direction) {
	state.manualUntil = performance.now() + 800;
	state.targetSteering = manualSteeringInputs.size
		? [...manualSteeringInputs.values()].at(-1)
		: direction;
}
const manualSteeringInputs = new Map();
function setManualSteerInput(source, direction) {
	if (direction === 0) manualSteeringInputs.delete(source);
	else manualSteeringInputs.set(source, direction);
	applyManualSteer(0);
}
function bindHold(button, direction, source) {
	button.addEventListener('pointerdown', (event) => {
		event.preventDefault();
		button.setPointerCapture(event.pointerId);
		setManualSteerInput(source, direction);
	});
	button.addEventListener('pointerup', () => setManualSteerInput(source, 0));
	button.addEventListener('pointercancel', () => setManualSteerInput(source, 0));
	button.addEventListener('lostpointercapture', () => setManualSteerInput(source, 0));
}
bindHold(document.querySelector('#steer-left'), -0.8, 'button-left');
bindHold(document.querySelector('#steer-right'), 0.8, 'button-right');

document.addEventListener('keydown', (event) => {
	if (event.target instanceof HTMLInputElement) return;
	if (event.key.toLowerCase() === 'a' || event.key === 'ArrowLeft') {
		event.preventDefault();
		setManualSteerInput('keyboard-left', -0.8);
		return;
	}
	if (event.key.toLowerCase() === 'd' || event.key === 'ArrowRight') {
		event.preventDefault();
		setManualSteerInput('keyboard-right', 0.8);
		return;
	}
	if (event.repeat) return;
	if (event.key.toLowerCase() === 'w' || event.key === 'ArrowUp') {
		state.manualUntil = performance.now() + 800;
		state.throttle = Math.min(1, state.throttle + 0.18);
		ui.throttle.value = Math.round(state.throttle * 100);
	}
	if (event.key.toLowerCase() === 's' || event.key === 'ArrowDown') {
		state.manualUntil = performance.now() + 800;
		state.throttle = Math.max(-1, state.throttle - 0.3);
		ui.throttle.value = Math.round(Math.max(0, state.throttle) * 100);
	}
	if (event.code === 'Space') {
		event.preventDefault();
		resetRun();
	}
});
document.addEventListener('keyup', (event) => {
	if (event.key.toLowerCase() === 'a' || event.key === 'ArrowLeft') setManualSteerInput('keyboard-left', 0);
	if (event.key.toLowerCase() === 'd' || event.key === 'ArrowRight') setManualSteerInput('keyboard-right', 0);
});
window.addEventListener('blur', () => {
	manualSteeringInputs.clear();
	applyManualSteer(0);
});
ui.throttle.addEventListener('input', () => {
	state.manualUntil = performance.now() + 1200;
	state.throttle = (Number(ui.throttle.value) / 100) * 1.3 - 0.25;
	updateHud();
});
ui.pause.addEventListener('click', () => {
	state.paused = !state.paused;
	ui.pause.setAttribute('aria-label', state.paused ? 'Resume simulation' : 'Pause simulation');
	ui.pause.title = state.paused ? 'Resume simulation' : 'Pause simulation';
	ui.pause.innerHTML = state.paused
		? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 5 11 7-11 7V5Z"/></svg>'
		: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14"/></svg>';
	clock.getDelta();
});
document.querySelector('#sound-toggle').addEventListener('click', () => setSoundEnabled(!soundEnabled));
ui.reset.addEventListener('click', resetRun);

function resetRun() {
	if (state.distance > state.best) {
		state.best = state.distance;
		localStorage.setItem('apex-best-distance', String(state.best));
	}
	state.distance = 0;
	state.lateral = 0;
	state.speed = 20;
	state.steering = 0;
	state.targetSteering = 0;
	state.avoidanceCar = null;
	state.avoidanceTarget = null;
	state.throttle = 0.62;
	state.episode += 1;
	state.manualUntil = 0;
	for (const car of cars) {
		car.userData.distance = trafficStartDistance(car.userData.index, Math.random() * 14);
		car.userData.lane = [-3.4, 0, 3.4][Math.floor(Math.random() * 3)];
	}
	ui.throttle.value = 62;
	clock.getDelta();
	updateHud();
}

updateHud();
requestAnimationFrame(render);
