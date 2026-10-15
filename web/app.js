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
	modeToggle: document.querySelector('#mode-toggle'),
	pause: document.querySelector('#pause-toggle'),
	reset: document.querySelector('#reset-button'),
	frame: document.querySelector('#frame-status'),
	best: document.querySelector('#best-distance'),
	collisions: document.querySelector('#collision-count'),
};

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x090b1b, 105, 285);
scene.background = new THREE.Color(0x090b1b);

const camera = new THREE.PerspectiveCamera(43, 1, 0.1, 320);
camera.position.set(0, 9.5, 19);
camera.lookAt(0, 0, -20);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.22;
mount.appendChild(renderer.domElement);

const ambientLight = new THREE.HemisphereLight(0x8198e8, 0x211b3b, 1.65);
scene.add(ambientLight);
const fillLight = new THREE.AmbientLight(0x514477, 0.9);
scene.add(fillLight);
const moonlight = new THREE.DirectionalLight(0x9dbdff, 2.2);
moonlight.position.set(-18, 34, 15);
moonlight.castShadow = true;
moonlight.shadow.mapSize.set(1024, 1024);
moonlight.shadow.camera.left = -45;
moonlight.shadow.camera.right = 45;
moonlight.shadow.camera.top = 40;
moonlight.shadow.camera.bottom = -40;
scene.add(moonlight);

const groundMaterial = new THREE.MeshStandardMaterial({ color: 0x080a15, roughness: 0.92, metalness: 0.08 });
const ground = new THREE.Mesh(new THREE.PlaneGeometry(500, 500), groundMaterial);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.22;
ground.position.z = -80;
ground.receiveShadow = true;
scene.add(ground);

const roadMaterial = new THREE.MeshStandardMaterial({
	color: 0x414457,
	emissive: 0x161a30,
	emissiveIntensity: 0.55,
	roughness: 0.34,
	metalness: 0.42,
});
const shoulderMaterial = new THREE.MeshStandardMaterial({ color: 0x22243a, roughness: 0.62, metalness: 0.2 });
const sidewalkMaterial = new THREE.MeshStandardMaterial({ color: 0x33374b, roughness: 0.72, metalness: 0.16 });
const curbMaterials = [
	new THREE.MeshStandardMaterial({ color: 0xff3e99, emissive: 0xff167e, emissiveIntensity: 2.4, roughness: 0.35, side: THREE.DoubleSide }),
	new THREE.MeshStandardMaterial({ color: 0x65f8ff, emissive: 0x12cfff, emissiveIntensity: 2.3, roughness: 0.35, side: THREE.DoubleSide }),
];
const centerMaterial = new THREE.MeshStandardMaterial({
	color: 0xb9faff,
	emissive: 0x42d9ff,
	emissiveIntensity: 1.25,
	roughness: 0.38,
	side: THREE.DoubleSide,
});
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

function trackSlope(z) {
	const sampleDistance = 0.25;
	return (
		trackCenter(z + sampleDistance) - trackCenter(z - sampleDistance)
	) / (2 * sampleDistance);
}

function trackPoint(distance, laneOffset, referenceDistance) {
	const slope = trackSlope(-distance);
	const normalLength = Math.hypot(1.1, slope);
	return new THREE.Vector3(
		trackCenter(-distance) + laneOffset * 1.1 / normalLength,
		0,
		5 - (distance - referenceDistance) * 1.1 - laneOffset * slope / normalLength,
	);
}

function dampAngle(current, target, rate, dt) {
	const difference = Math.atan2(Math.sin(target - current), Math.cos(target - current));
	return current + difference * (1 - Math.exp(-rate * dt));
}

function smoothstep(value) {
	const clamped = THREE.MathUtils.clamp(value, 0, 1);
	return clamped * clamped * (3 - 2 * clamped);
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
			if (side < 0) {
				indices.push(vertex, vertex + 2, vertex + 1, vertex + 1, vertex + 2, vertex + 3);
			} else {
				indices.push(vertex, vertex + 1, vertex + 2, vertex + 1, vertex + 3, vertex + 2);
			}
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
roadRoot.add(stripMesh(-1, roadWidth / 2 + 2.5, 4.5, sidewalkMaterial, -0.005));
roadRoot.add(stripMesh(1, roadWidth / 2 + 2.5, 4.5, sidewalkMaterial, -0.005));

function createRoadMarkings() {
	const group = new THREE.Group();
	group.userData.movingFeatures = [];
	group.userData.edgeLines = [];
	const addMarking = (length, width, material, laneOffset, trackZ, y) => {
		const marking = createCurvedMarking(length, width, material, laneOffset, trackZ, y);
		group.add(marking);
		group.userData.movingFeatures.push(marking);
	};
	for (let z = 36; z > -roadLength + 45; z -= 9) {
		addMarking(4.4, 0.22, centerMaterial, 0, z - 5, 0.075);
	}
	for (let z = 42; z > -roadLength + 45; z -= 2.2) {
		const stripeMaterial = curbMaterials[Math.floor((42 - z) / 1.8) % 2];
		for (const side of [-1, 1]) {
			addMarking(
				1.35,
				0.2,
				stripeMaterial,
				side * (roadWidth / 2 - 0.48),
				z - 5,
				0.085,
			);
		}
	}
	for (const side of [-1, 1]) {
		const edgeLine = createRoadEdgeLine(side);
		group.add(edgeLine.mesh);
		group.userData.edgeLines.push(edgeLine);
	}
	return group;
}

function createRoadEdgeLine(side) {
	const positions = new Float32Array((roadSegments + 1) * 2 * 3);
	const indices = [];
	for (let index = 0; index < roadSegments; index += 1) {
		const vertex = index * 2;
		indices.push(vertex, vertex + 2, vertex + 1, vertex + 1, vertex + 2, vertex + 3);
	}
	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
	geometry.setIndex(indices);
	const mesh = new THREE.Mesh(geometry, curbMaterials[side < 0 ? 0 : 1]);
	mesh.frustumCulled = false;
	mesh.receiveShadow = false;
	return { mesh, geometry, side };
}

function createCurvedMarking(length, width, material, laneOffset, trackZ, y) {
	const segments = Math.max(1, Math.ceil(length / 0.45));
	const positions = new Float32Array((segments + 1) * 2 * 3);
	const indices = [];
	for (let index = 0; index < segments; index += 1) {
		const vertex = index * 2;
		indices.push(vertex, vertex + 2, vertex + 1, vertex + 1, vertex + 2, vertex + 3);
	}
	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
	geometry.setIndex(indices);
	geometry.computeVertexNormals();
	const marking = new THREE.Mesh(geometry, material);
	marking.userData.trackZ = trackZ;
	marking.userData.laneOffset = laneOffset;
	marking.userData.length = length;
	marking.userData.width = width;
	marking.userData.segments = segments;
	marking.userData.y = y;
	marking.userData.movingMarking = true;
	marking.frustumCulled = false;
	marking.receiveShadow = false;
	return marking;
}
const roadMarkings = createRoadMarkings();
scene.add(roadMarkings);
const scenery = [];
const buildingMaterials = [
	new THREE.MeshStandardMaterial({ color: 0x17182c, roughness: 0.72, metalness: 0.3 }),
	new THREE.MeshStandardMaterial({ color: 0x20203a, roughness: 0.68, metalness: 0.36 }),
	new THREE.MeshStandardMaterial({ color: 0x191d32, roughness: 0.75, metalness: 0.25 }),
];
const windowMaterials = [
	new THREE.MeshStandardMaterial({ color: 0x67e8ff, emissive: 0x35b6ff, emissiveIntensity: 2.6 }),
	new THREE.MeshStandardMaterial({ color: 0xff6ac1, emissive: 0xff258d, emissiveIntensity: 2.4 }),
	new THREE.MeshStandardMaterial({ color: 0xffd784, emissive: 0xffa93d, emissiveIntensity: 1.8 }),
];
const cityAccents = [
	new THREE.MeshStandardMaterial({ color: 0x4cf5ff, emissive: 0x12c9ff, emissiveIntensity: 3.2 }),
	new THREE.MeshStandardMaterial({ color: 0xff58bb, emissive: 0xff1686, emissiveIntensity: 3.2 }),
];
const buildingWindowGeometry = new THREE.BoxGeometry(0.07, 0.48, 0.2);
const neonBarGeometry = new THREE.BoxGeometry(0.1, 1, 0.18);

function makeSignTexture(label, color) {
	const canvas = document.createElement('canvas');
	canvas.width = 256;
	canvas.height = 96;
	const context = canvas.getContext('2d');
	context.fillStyle = '#101020';
	context.fillRect(0, 0, canvas.width, canvas.height);
	context.strokeStyle = color;
	context.lineWidth = 5;
	context.strokeRect(5, 5, canvas.width - 10, canvas.height - 10);
	context.shadowColor = color;
	context.shadowBlur = 18;
	context.fillStyle = '#ffffff';
	context.font = '800 46px sans-serif';
	context.textAlign = 'center';
	context.textBaseline = 'middle';
	context.fillText(label, canvas.width / 2, canvas.height / 2);
	const texture = new THREE.CanvasTexture(canvas);
	texture.colorSpace = THREE.SRGBColorSpace;
	return texture;
}

const signTextures = [
	makeSignTexture('NOVA', '#42edff'),
	makeSignTexture('夜市', '#ff4fae'),
	makeSignTexture('APEX', '#ffc46b'),
	makeSignTexture('NEXUS', '#a384ff'),
];
const signMaterial = signTextures.map((map) => new THREE.SpriteMaterial({
	map,
	color: 0xffffff,
	toneMapped: false,
}));

function makeCityBuilding(index, side, trackZ) {
	const group = new THREE.Group();
	const width = 8 + (index % 4) * 1.6;
	const depth = 8 + (index % 3) * 2;
	const distantTower = index >= 20 && index < 40;
	const height = distantTower
		? 34 + ((index * 17) % 48)
		: 18 + ((index * 13) % 30);
	const body = new THREE.Mesh(
		new THREE.BoxGeometry(width, height, depth),
		buildingMaterials[index % buildingMaterials.length],
	);
	body.position.y = height / 2;
	body.castShadow = false;
	body.receiveShadow = false;
	group.add(body);

	const floors = Math.floor(height / 2.3);
	const columns = Math.floor(depth / 1.35);
	const windows = new THREE.InstancedMesh(
		buildingWindowGeometry,
		windowMaterials[index % windowMaterials.length],
		floors * columns * 2,
	);
	const windowDummy = new THREE.Object3D();
	let windowIndex = 0;
	for (const face of [side > 0 ? -1 : 1, side > 0 ? 1 : -1]) {
		for (let floor = 0; floor < floors; floor += 1) {
			for (let column = 0; column < columns; column += 1) {
				if ((floor * 7 + column * 11 + index + (face < 0 ? 2 : 0)) % 5 === 0) continue;
				windowDummy.position.set(
					face * (width / 2 + 0.045),
					1.45 + floor * 2.25,
					(column - (columns - 1) / 2) * 1.25,
				);
				windowDummy.updateMatrix();
				windows.setMatrixAt(windowIndex, windowDummy.matrix);
				windowIndex += 1;
			}
		}
	}
	windows.count = windowIndex;
	group.add(windows);

	const accent = cityAccents[index % cityAccents.length];
	const trim = new THREE.Mesh(neonBarGeometry, accent);
	trim.position.set(side > 0 ? -width / 2 - 0.1 : width / 2 + 0.1, height / 2, 0);
	trim.scale.y = height;
	group.add(trim);
	const roof = new THREE.Mesh(
		new THREE.BoxGeometry(width * 0.42, 0.35, depth * 0.35),
		accent,
	);
	roof.position.set(0, height + 0.18, 0);
	group.add(roof);

	if (index % 2 === 0) {
		const sign = new THREE.Sprite(signMaterial[index % signMaterial.length]);
		sign.position.set(side > 0 ? -width / 2 - 0.18 : width / 2 + 0.18, height * 0.62, 0);
		sign.scale.set(5.4, 2.05, 1);
		group.add(sign);
	}
	group.userData.trackZ = trackZ;
	const roadsideOffset = distantTower
		? 76 + (index % 3) * 18
		: index >= 40
			? 15 + (index % 4) * 3
			: 38 + (index % 3) * 10;
	group.userData.laneOffset = side * (roadWidth / 2 + roadsideOffset);
	scene.add(group);
	scenery.push(group);
}

for (let index = 0; index < 20; index += 1) {
	makeCityBuilding(index, index % 2 === 0 ? -1 : 1, 34 - index * 14);
}
for (let index = 0; index < 16; index += 1) {
	makeCityBuilding(index + 20, index % 2 === 0 ? 1 : -1, 22 - index * 17);
}
for (let index = 0; index < 26; index += 1) {
	makeCityBuilding(index + 40, index % 2 === 0 ? -1 : 1, 30 - index * 10);
}

const roadLights = [];
const streetlightMetal = new THREE.MeshStandardMaterial({
	color: 0x30374b,
	metalness: 0.78,
	roughness: 0.32,
});
const streetlightFixture = new THREE.MeshStandardMaterial({
	color: 0xa7f9ff,
	emissive: 0x48dfff,
	emissiveIntensity: 2.4,
});

function makeStreetlight(index) {
	const side = index % 2 === 0 ? -1 : 1;
	const group = new THREE.Group();
	const base = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.38, 0.22, 8), streetlightMetal);
	base.position.y = 0.11;
	group.add(base);
	const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.15, 5.5, 8), streetlightMetal);
	pole.position.y = 2.86;
	pole.castShadow = true;
	group.add(pole);
	const arm = new THREE.Mesh(new THREE.BoxGeometry(1.05, 0.12, 0.14), streetlightMetal);
	arm.position.set(-side * 0.48, 5.52, 0);
	group.add(arm);
	const fixture = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.16, 0.38), streetlightMetal);
	fixture.position.set(-side * 0.9, 5.42, 0);
	group.add(fixture);
	const bulb = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.045, 0.27), streetlightFixture);
	bulb.position.set(-side * 0.9, 5.32, 0);
	group.add(bulb);
	const light = new THREE.PointLight(0x91eaff, 30, 38, 1.6);
	light.position.set(-side * 0.9, 5.15, 0);
	light.castShadow = false;
	light.visible = false;
	group.add(light);

	group.userData.trackZ = 36 - index * 23;
	group.userData.laneOffset = side * 8.2;
	group.userData.side = side;
	group.userData.light = light;
	group.userData.bulb = bulb;
	group.userData.active = false;
	group.userData.distance = Infinity;
	scene.add(group);
	roadLights.push(group);
}

for (let index = 0; index < 12; index += 1) makeStreetlight(index);

const daySkyObjects = new THREE.Group();
const sunDisc = new THREE.Mesh(
	new THREE.SphereGeometry(10, 24, 16),
	new THREE.MeshBasicMaterial({ color: 0xffe7a6, toneMapped: false }),
);
sunDisc.position.set(-88, 74, -225);
daySkyObjects.add(sunDisc);
const cloudMaterial = new THREE.MeshStandardMaterial({
	color: 0xe5edf5,
	roughness: 1,
	flatShading: true,
});
for (let index = 0; index < 10; index += 1) {
	const cloud = new THREE.Group();
	const scale = 1.8 + (index % 4) * 0.65;
	for (let puff = 0; puff < 4; puff += 1) {
		const shape = new THREE.Mesh(
			new THREE.IcosahedronGeometry(scale * (0.48 + (puff % 2) * 0.12), 1),
			cloudMaterial,
		);
		shape.position.set((puff - 1.5) * scale * 0.65, Math.sin(puff * 2) * scale * 0.1, (puff % 2) * scale * 0.2);
		cloud.add(shape);
	}
	cloud.position.set(-105 + index * 23, 42 + (index % 3) * 9, -90 - (index % 4) * 42);
	daySkyObjects.add(cloud);
}
daySkyObjects.visible = false;
scene.add(daySkyObjects);

const cars = [];
const carColors = [0x79b7c8, 0xd8b967, 0xd67a65, 0x94a7d0, 0xe5e0d1];
function trafficStartDistance(index, offset = 0) {
	return 35 + index * 32 + offset;
}

function resetTrafficCars() {
	const lanes = [-3.4, 0, 3.4];
	for (const car of cars) {
		car.userData.distance = trafficStartDistance(car.userData.index, Math.random() * 14);
		car.userData.lane = lanes[Math.floor(Math.random() * lanes.length)];
	}
}

function makeCar(color, isPlayer = false) {
	const car = new THREE.Group();
	car.rotation.order = 'YXZ';
	const accentColor = isPlayer ? 0x43f5ff : 0xff4f9b;
	const paint = new THREE.MeshStandardMaterial({
		color,
		metalness: 0.68,
		roughness: 0.27,
		emissive: color,
		emissiveIntensity: 0.075,
	});
	const darkPaint = new THREE.MeshStandardMaterial({ color: 0x111522, metalness: 0.6, roughness: 0.28 });
	const glass = new THREE.MeshStandardMaterial({
		color: 0x52c9e8,
		metalness: 0.48,
		roughness: 0.16,
		emissive: 0x0b4560,
		emissiveIntensity: 0.75,
	});
	const profile = new THREE.Shape();
	profile.moveTo(-1.95, 0.34);
	profile.lineTo(1.95, 0.34);
	profile.lineTo(1.88, 0.62);
	profile.lineTo(1.53, 0.76);
	profile.lineTo(0.78, 0.83);
	profile.lineTo(0.38, 1.02);
	profile.lineTo(-0.48, 1.02);
	profile.lineTo(-0.96, 0.78);
	profile.lineTo(-1.78, 0.72);
	profile.closePath();
	const bodyGeometry = new THREE.ExtrudeGeometry(profile, {
		depth: 1.75,
		bevelEnabled: true,
		bevelSegments: 2,
		steps: 1,
		bevelSize: 0.055,
		bevelThickness: 0.055,
	});
	bodyGeometry.translate(0, 0, -0.875);
	bodyGeometry.rotateY(Math.PI / 2);
	const body = new THREE.Mesh(bodyGeometry, paint);
	body.castShadow = true;
	body.receiveShadow = true;
	car.add(body);
	const cabinProfile = new THREE.Shape();
	cabinProfile.moveTo(-0.87, 0.78);
	cabinProfile.lineTo(0.87, 0.78);
	cabinProfile.lineTo(0.48, 1.36);
	cabinProfile.lineTo(-0.43, 1.36);
	cabinProfile.closePath();
	const cabinGeometry = new THREE.ExtrudeGeometry(cabinProfile, {
		depth: 1.22,
		bevelEnabled: true,
		bevelSegments: 2,
		steps: 1,
		bevelSize: 0.04,
		bevelThickness: 0.04,
	});
	cabinGeometry.translate(0, 0, -0.61);
	cabinGeometry.rotateY(Math.PI / 2);
	const cabin = new THREE.Mesh(cabinGeometry, darkPaint);
	cabin.castShadow = true;
	car.add(cabin);

	const windshield = new THREE.Mesh(new THREE.BoxGeometry(1.12, 0.42, 0.045), glass);
	windshield.position.set(0, 1.08, -0.73);
	windshield.rotation.x = -0.28;
	car.add(windshield);
	const rearGlass = new THREE.Mesh(new THREE.BoxGeometry(1.05, 0.38, 0.045), glass);
	rearGlass.position.set(0, 1.08, 0.58);
	rearGlass.rotation.x = 0.3;
	car.add(rearGlass);
	for (const side of [-1, 1]) {
		for (const z of [-0.28, 0.25]) {
			const sideWindow = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.34, 0.43), glass);
			sideWindow.position.set(side * 0.64, 1.1, z);
			car.add(sideWindow);
		}
		const sill = new THREE.Mesh(
			new THREE.BoxGeometry(0.09, 0.12, 2.7),
			new THREE.MeshStandardMaterial({
				color: accentColor,
				emissive: accentColor,
				emissiveIntensity: 2.2,
			}),
		);
		sill.position.set(side * 0.9, 0.36, 0.05);
		car.add(sill);
	}
	const roof = new THREE.Mesh(new THREE.BoxGeometry(0.94, 0.07, 0.7), paint);
	roof.position.set(0, 1.39, 0.02);
	roof.castShadow = true;
	car.add(roof);
	const frontSplitter = new THREE.Mesh(
		new THREE.BoxGeometry(1.95, 0.09, 0.28),
		new THREE.MeshStandardMaterial({ color: 0x090d18, metalness: 0.65, roughness: 0.25 }),
	);
	frontSplitter.position.set(0, 0.34, -1.88);
	car.add(frontSplitter);
	const rearWing = new THREE.Mesh(
		new THREE.BoxGeometry(2.05, 0.12, 0.3),
		new THREE.MeshStandardMaterial({ color: accentColor, emissive: accentColor, emissiveIntensity: 1.25, metalness: 0.5 }),
	);
	rearWing.position.set(0, 1.02, 1.68);
	car.add(rearWing);
	for (const x of [-0.62, 0.62]) {
		const support = new THREE.Mesh(
			new THREE.BoxGeometry(0.09, 0.34, 0.1),
			new THREE.MeshStandardMaterial({ color: 0x121522, metalness: 0.7, roughness: 0.28 }),
		);
		support.position.set(x, 0.84, 1.6);
		car.add(support);
	}
	const diffuser = new THREE.Mesh(
		new THREE.BoxGeometry(1.35, 0.2, 0.18),
		new THREE.MeshStandardMaterial({ color: 0x0b0e18, metalness: 0.65, roughness: 0.3 }),
	);
	diffuser.position.set(0, 0.33, 1.88);
	car.add(diffuser);
	const grille = new THREE.Mesh(
		new THREE.BoxGeometry(0.72, 0.2, 0.055),
		new THREE.MeshStandardMaterial({ color: 0x101321, metalness: 0.7, roughness: 0.24 }),
	);
	grille.position.set(0, 0.55, -1.91);
	car.add(grille);

	const wheels = [];
	const frontWheelPivots = [];
	const wheelGeometry = new THREE.CylinderGeometry(0.37, 0.37, 0.24, 16);
	const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x080a11, roughness: 0.68 });
	const rimMaterial = new THREE.MeshStandardMaterial({ color: 0x93a9bd, metalness: 0.86, roughness: 0.24 });
	for (const x of [-0.94, 0.94]) {
		for (const z of [-1.18, 1.2]) {
			const front = z < 0;
			const wheelPivot = new THREE.Group();
			wheelPivot.position.set(x, 0.36, z);
			const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
			wheel.rotation.z = Math.PI / 2;
			wheel.castShadow = true;
			const rim = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.035, 12), rimMaterial);
			rim.rotation.z = Math.PI / 2;
			rim.position.x = x < 0 ? -0.13 : 0.13;
			wheel.add(rim);
			wheelPivot.add(wheel);
			car.add(wheelPivot);
			wheels.push(wheel);
			if (front) frontWheelPivots.push(wheelPivot);
		}
	}
	const lampMaterial = new THREE.MeshBasicMaterial({ color: 0xc9fbff, toneMapped: false });
	const headlight = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.1, 0.07), lampMaterial);
	headlight.position.set(0, 0.67, -1.91);
	car.add(headlight);
	for (const x of [-0.58, 0.58]) {
		const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.1, 0.08), lampMaterial);
		lamp.position.set(x, 0.67, -1.9);
		car.add(lamp);
	}
	const tailMaterial = new THREE.MeshBasicMaterial({ color: 0xff397f, toneMapped: false });
	for (const x of [-0.62, 0.62]) {
		const tail = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.11, 0.07), tailMaterial);
		tail.position.set(x * 0.72, 0.62, 1.89);
		car.add(tail);
	}
	const underglow = new THREE.Mesh(
		new THREE.PlaneGeometry(1.7, 3.3),
		new THREE.MeshBasicMaterial({
			color: accentColor,
			transparent: true,
			opacity: 0.2,
			depthWrite: false,
			toneMapped: false,
		}),
	);
	underglow.rotation.x = -Math.PI / 2;
	underglow.position.set(0, 0.12, 0);
	car.add(underglow);
	const shadow = new THREE.Mesh(
		new THREE.PlaneGeometry(2.7, 4.8),
		new THREE.MeshBasicMaterial({ color: 0x101411, transparent: true, opacity: 0.42, depthWrite: false }),
	);
	shadow.rotation.x = -Math.PI / 2;
	shadow.position.y = 0.025;
	car.add(shadow);
	car.userData.wheels = wheels;
	car.userData.frontWheelPivots = frontWheelPivots;
	car.userData.roll = 0;
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
	laneChange: null,
	throttle: 0.62,
	dayMode: localStorage.getItem('apex-drive-mode') === 'day',
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
	let steeringTarget = THREE.MathUtils.clamp(-state.lateral * 0.1 - state.steering * 0.55, -0.6, 0.6);
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
			state.laneChange = {
				start: state.lateral,
				target: state.avoidanceTarget,
				elapsed: 0,
				duration: Math.max(
					0.8,
					1.5 * Math.abs(state.avoidanceTarget - state.lateral)
						/ (Math.max(state.speed, 11.2) * Math.tan(12 * Math.PI / 180)),
				),
			};
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

function laneChangeOffset(secondsAhead = 0) {
	const change = state.laneChange;
	if (!change) return state.lateral;
	const progress = (change.elapsed + secondsAhead) / change.duration;
	return THREE.MathUtils.lerp(change.start, change.target, smoothstep(progress));
}

function updateCarPose(car, distance, laneOffset, speed, dt, referenceDistance, offsetAtDistance = () => laneOffset, steering = 0) {
	const ahead = 1.5 + Math.min(speed, 32) * 0.12;
	const sampleSpacing = 0.8;
	const offsetAtTime = (distanceAhead) => offsetAtDistance(distanceAhead / Math.max(speed, 1));
	const pointAt = (distanceAhead) => trackPoint(
		distance + distanceAhead,
		offsetAtTime(distanceAhead),
		referenceDistance,
	);
	const headingBetween = (distanceAhead, varyingOffset) => {
		const pointAtHeading = (sampleDistance) => varyingOffset
			? pointAt(sampleDistance)
			: trackPoint(distance + sampleDistance, laneOffset, referenceDistance);
		const start = pointAtHeading(distanceAhead - sampleSpacing / 2);
		const end = pointAtHeading(distanceAhead + sampleSpacing / 2);
		return Math.atan2(-(end.x - start.x), -(end.z - start.z));
	};
	const pathYaw = headingBetween(ahead, true);
	const laneYaw = headingBetween(ahead, false);
	const laneChangeYaw = Math.atan2(Math.sin(pathYaw - laneYaw), Math.cos(pathYaw - laneYaw));
	const targetYaw = state.laneChange
		? laneYaw + THREE.MathUtils.clamp(laneChangeYaw, -12 * Math.PI / 180, 12 * Math.PI / 180)
		: pathYaw;
	const curvature = Math.atan2(
		Math.sin(headingBetween(ahead + 0.8, true) - headingBetween(ahead - 0.8, true)),
		Math.cos(headingBetween(ahead + 0.8, true) - headingBetween(ahead - 0.8, true)),
	) / 1.6;

	car.position.copy(trackPoint(distance, laneOffset, referenceDistance));
	car.rotation.y = dampAngle(car.rotation.y, targetYaw, 6, dt);
	car.userData.roll = THREE.MathUtils.damp(
		car.userData.roll,
		THREE.MathUtils.clamp(-curvature * speed * 0.48, -0.12, 0.12),
		4.5,
		dt,
	);
	car.rotation.z = car.userData.roll;
	for (const pivot of car.userData.frontWheelPivots) {
		pivot.rotation.y = THREE.MathUtils.damp(
			pivot.rotation.y,
			THREE.MathUtils.clamp(-steering * 0.24, -0.22, 0.22),
			9,
			dt,
		);
	}
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
	if (state.laneChange) {
		state.laneChange.elapsed += dt;
		const progress = THREE.MathUtils.clamp(
			state.laneChange.elapsed / state.laneChange.duration,
			0,
			1,
		);
		state.lateral = laneChangeOffset();
		state.steering = THREE.MathUtils.clamp(
			(state.laneChange.target - state.laneChange.start)
				* 6 * progress * (1 - progress)
				/ (state.laneChange.duration * 4.1),
			-0.9,
			0.9,
		);
		if (state.laneChange.elapsed >= state.laneChange.duration) {
			state.lateral = state.laneChange.target;
			state.steering = 0;
			state.laneChange = null;
		}
	} else {
		state.lateral += state.steering * dt * 4.1;
	}
	state.lateral = THREE.MathUtils.clamp(state.lateral, -safeLateralLimit, safeLateralLimit);
	state.speed = THREE.MathUtils.clamp(state.speed + state.throttle * 5.3 * dt, 11.2, 32);
	state.distance += state.speed * dt;

	let crash = false;
	for (const car of cars) {
		car.userData.distance += car.userData.speed * dt;
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
		state.laneChange = null;
		state.avoidanceCar = null;
		state.avoidanceTarget = null;
		resetTrafficCars();
	}
	for (const car of cars) {
		if (car.userData.distance - state.distance < -18) {
			car.userData.distance += 220;
			car.userData.lane = [-3.4, 0, 3.4][Math.floor(Math.random() * 3)];
		}
		updateCarPose(car, car.userData.distance, car.userData.lane, car.userData.speed, dt, state.distance);
		for (const wheel of car.userData.wheels) wheel.rotation.x += car.userData.speed * dt * 0.45;
	}

	updateCarPose(
		playerCar,
		state.distance,
		state.lateral,
		state.speed,
		dt,
		state.distance,
		(secondsAhead) => state.laneChange
			? laneChangeOffset(secondsAhead)
			: THREE.MathUtils.clamp(
				state.lateral + state.steering * 4.1 * secondsAhead,
				-safeLateralLimit,
				safeLateralLimit,
			),
		state.steering,
	);
	for (const wheel of playerCar.userData.wheels) wheel.rotation.x += state.speed * dt * 0.45;

	updateTrackView();
	const curve = trackCenter(-state.distance - 18) - trackCenter(-state.distance);
	const cameraX = playerCar.position.x * 0.85 + trackCenter(-state.distance) * 0.15 + curve * 0.15;
	camera.position.x = smooth(camera.position.x, cameraX, 2.5, dt);
	camera.position.y = smooth(camera.position.y, 9.2 + Math.abs(curve) * 0.035, 2, dt);
	camera.position.z = smooth(camera.position.z, 19, 2, dt);
	camera.userData.yaw = dampAngle(camera.userData.yaw ?? playerCar.rotation.y, playerCar.rotation.y, 2.2, dt);
	camera.lookAt(
		camera.position.x - Math.sin(camera.userData.yaw) * 30,
		0.3,
		camera.position.z - Math.cos(camera.userData.yaw) * 30,
	);
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
		const data = feature.userData;
		const positions = feature.geometry.attributes.position;
		for (let index = 0; index <= data.segments; index += 1) {
			const along = (index / data.segments - 0.5) * data.length;
			const centerlineDistance = data.trackZ + along;
			const slope = trackSlope(centerlineDistance);
			const normalScale = 1 / Math.sqrt(1 + slope * slope);
			for (let edge = 0; edge < 2; edge += 1) {
				const lateral = data.laneOffset + (edge === 0 ? -data.width / 2 : data.width / 2);
				const vertex = index * 2 + edge;
				positions.setXYZ(
					vertex,
					trackCenter(centerlineDistance) + lateral * normalScale,
					data.y,
					5 + data.trackZ + state.distance + along - lateral * slope * normalScale,
				);
			}
		}
		positions.needsUpdate = true;
		feature.geometry.computeVertexNormals();
	}

	for (const edgeLine of roadMarkings.userData.edgeLines) {
		const positions = edgeLine.geometry.attributes.position;
		for (let index = 0; index <= roadSegments; index += 1) {
			const offset = roadBehind - index * segmentLength;
			const trackZ = -state.distance + offset;
			const slope = trackSlope(trackZ);
			const normalScale = 1 / Math.sqrt(1 + slope * slope);
			for (let edge = 0; edge < 2; edge += 1) {
				const lateral = edgeLine.side * (roadWidth / 2 - 0.12 + (edge === 0 ? -0.1 : 0.1));
				const vertex = index * 2 + edge;
				positions.setXYZ(
					vertex,
					trackCenter(trackZ) + lateral * normalScale,
					0.07,
					5 + offset - lateral * slope * normalScale,
				);
			}
		}
		positions.needsUpdate = true;
		edgeLine.geometry.computeVertexNormals();
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

	for (const lamp of roadLights) {
		let offset = lamp.userData.trackZ + state.distance;
		while (offset > roadBehind) {
			lamp.userData.trackZ -= roadLength;
			offset -= roadLength;
		}
		while (offset < -roadAhead) {
			lamp.userData.trackZ += roadLength;
			offset += roadLength;
		}
		const trackZ = lamp.userData.trackZ;
		const laneOffset = lamp.userData.laneOffset;
		const slope = trackSlope(trackZ);
		const normalLength = Math.hypot(1.1, slope);
		lamp.position.set(
			trackCenter(trackZ) + laneOffset * 1.1 / normalLength,
			0,
			5 + offset - laneOffset * slope / normalLength,
		);
		const before = trackPoint(-trackZ - 0.5, laneOffset, 0);
		const after = trackPoint(-trackZ + 0.5, laneOffset, 0);
		lamp.rotation.y = Math.atan2(-(after.x - before.x), -(after.z - before.z));
		lamp.userData.distance = Math.abs(offset);
	}
	const nearestLamps = [...roadLights]
		.sort((left, right) => left.userData.distance - right.userData.distance)
		.slice(0, 4);
	for (const lamp of roadLights) {
		lamp.userData.active = nearestLamps.includes(lamp);
		lamp.userData.light.visible = !state.dayMode && lamp.userData.active;
	}
}

function applyDriveMode() {
	const isDay = state.dayMode;
	document.body.classList.toggle('day-mode', isDay);
	scene.background.set(isDay ? 0x9fc5e8 : 0x090b1b);
	scene.fog.color.set(isDay ? 0x9fc5e8 : 0x090b1b);
	daySkyObjects.visible = isDay;
	scene.fog.near = isDay ? 150 : 105;
	scene.fog.far = isDay ? 330 : 285;
	ambientLight.color.set(isDay ? 0xd7e9ff : 0x8198e8);
	ambientLight.groundColor.set(isDay ? 0x586379 : 0x211b3b);
	ambientLight.intensity = isDay ? 2.0 : 1.65;
	fillLight.color.set(isDay ? 0xffe9cf : 0x514477);
	fillLight.intensity = isDay ? 0.8 : 0.9;
	moonlight.color.set(isDay ? 0xffedcf : 0x9dbdff);
	moonlight.intensity = isDay ? 3.2 : 2.2;
	renderer.toneMappingExposure = isDay ? 1.12 : 1.22;

	groundMaterial.color.set(isDay ? 0x697386 : 0x080a15);
	roadMaterial.color.set(isDay ? 0x5d6878 : 0x414457);
	roadMaterial.emissive.set(isDay ? 0x080a10 : 0x161a30);
	roadMaterial.emissiveIntensity = isDay ? 0.08 : 0.55;
	roadMaterial.roughness = isDay ? 0.72 : 0.34;
	roadMaterial.metalness = isDay ? 0.08 : 0.42;
	shoulderMaterial.color.set(isDay ? 0x727988 : 0x22243a);
	sidewalkMaterial.color.set(isDay ? 0x9ca5b1 : 0x33374b);
	curbMaterials[0].color.set(isDay ? 0xe5e7ed : 0xff3e99);
	curbMaterials[0].emissive.set(isDay ? 0x000000 : 0x9e124f);
	curbMaterials[0].emissiveIntensity = isDay ? 0 : 1.5;
	curbMaterials[1].color.set(isDay ? 0xf0c85e : 0x65f8ff);
	curbMaterials[1].emissive.set(isDay ? 0x000000 : 0x147d9b);
	curbMaterials[1].emissiveIntensity = isDay ? 0 : 1.35;
	centerMaterial.color.set(isDay ? 0xf2f0dc : 0x8debf4);
	centerMaterial.emissive.set(isDay ? 0x000000 : 0x216b83);
	centerMaterial.emissiveIntensity = isDay ? 0 : 0.65;

	for (const material of buildingMaterials) {
		material.color.set(isDay ? 0x9aa9bd : material === buildingMaterials[0] ? 0x17182c : material === buildingMaterials[1] ? 0x20203a : 0x191d32);
	}
	const nightWindowColors = [0x67e8ff, 0xff6ac1, 0xffd784];
	const nightWindowEmissives = [0x35b6ff, 0xff258d, 0xffa93d];
	windowMaterials.forEach((material, index) => {
		material.color.set(isDay ? 0xbac9d9 : nightWindowColors[index]);
		material.emissive.set(isDay ? 0x526071 : nightWindowEmissives[index]);
		material.emissiveIntensity = isDay ? 0.12 : [2.6, 2.4, 1.8][index];
	});
	cityAccents.forEach((material, index) => {
		material.color.set(isDay ? 0x7393a8 : index === 0 ? 0x4cf5ff : 0xff58bb);
		material.emissive.set(isDay ? 0x1b3447 : index === 0 ? 0x12c9ff : 0xff1686);
		material.emissiveIntensity = isDay ? 0.35 : 3.2;
	});
	for (const material of signMaterial) material.color.set(isDay ? 0xc5d6e8 : 0xffffff);
	streetlightMetal.color.set(isDay ? 0x9ba8b8 : 0x30374b);
	streetlightFixture.color.set(isDay ? 0x78879a : 0xa7f9ff);
	streetlightFixture.emissiveIntensity = isDay ? 0 : 2.4;
	for (const lamp of roadLights) {
		lamp.userData.light.visible = !isDay && lamp.userData.active;
	}

	const buttonLabel = isDay ? 'NIGHT MODE' : 'DAY MODE';
	const targetLabel = isDay ? 'Switch to night mode' : 'Switch to day mode';
	ui.modeToggle.querySelector('span').textContent = buttonLabel;
	ui.modeToggle.setAttribute('aria-label', targetLabel);
	ui.modeToggle.setAttribute('aria-pressed', String(isDay));
	ui.modeToggle.title = targetLabel;
	ui.modeToggle.querySelector('svg').innerHTML = isDay
		? '<path d="M20.5 15.5A8.5 8.5 0 0 1 8.5 3.5 8.5 8.5 0 1 0 20.5 15.5Z"></path><path d="M16 3v3m-1.5-1.5h3"></path>'
		: '<circle cx="12" cy="12" r="4"></circle><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42"></path>';
	document.querySelector('#weather-label').textContent = isDay ? 'CLEAR / DAYLIGHT' : 'MIDNIGHT / NIGHT DRIVE';
	document.querySelector('meta[name="theme-color"]').content = isDay ? '#e7eef7' : '#111815';
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
	const throttleFill = state.dayMode ? '#138e9d' : '#b779ed';
	const throttleTrack = state.dayMode ? '#d6dfe8' : '#30263f';
	ui.throttle.style.background = `linear-gradient(90deg, ${throttleFill} ${throttlePercent}%, ${throttleTrack} ${throttlePercent}%)`;
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
	if (manualSteeringInputs.size) state.laneChange = null;
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
ui.modeToggle.addEventListener('click', () => {
	state.dayMode = !state.dayMode;
	localStorage.setItem('apex-drive-mode', state.dayMode ? 'day' : 'night');
	applyDriveMode();
});

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
	state.laneChange = null;
	state.avoidanceCar = null;
	state.avoidanceTarget = null;
	state.throttle = 0.62;
	state.episode += 1;
	state.manualUntil = 0;
	resetTrafficCars();
	ui.throttle.value = 62;
	clock.getDelta();
	updateHud();
}

updateHud();
applyDriveMode();
requestAnimationFrame(render);
