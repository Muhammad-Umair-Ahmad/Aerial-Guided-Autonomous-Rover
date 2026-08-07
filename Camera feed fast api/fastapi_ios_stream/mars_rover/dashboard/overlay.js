// overlay.js

let drawModeActive = false;
let isDrawing = false;
let gridStart = null;
let gridEnd = null;
let currentGridEnd = null;
let terrainGridData = null;

const overlayCanvas = document.getElementById('overlayCanvas');
const ctx = overlayCanvas.getContext('2d');
const video = document.getElementById('videoPlayer');

// Make canvas support pseudo-3D isometric view via CSS
overlayCanvas.style.transform = "perspective(800px) rotateX(60deg) rotateZ(-45deg)";
overlayCanvas.style.transformOrigin = "center";
overlayCanvas.style.transition = "transform 0.5s ease";

const minimapCanvas = document.getElementById('minimapCanvas');
if (minimapCanvas) {
    minimapCanvas.style.transform = "perspective(600px) rotateX(60deg) rotateZ(-45deg)";
    minimapCanvas.style.transformOrigin = "center";
}

// 3D Isometric View transformation
function drawRoverPrism(ctx, pose) {
    if (!pose) return;

    const x = pose.x || 0;
    const y = pose.y || 0;
    const heading = pose.heading || 0;
    const confidence = pose.confidence || 1.0;
    
    const w = 30; // width
    const l = 50; // length
    const h = 20; // height

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(heading);

    // Because the canvas is already rotated in 3D via CSS, "Y" on canvas maps to the flat plane.
    // However, if we just draw a box on canvas, it will look flat on the floor.
    // To make it look like a 3D prism standing UP from the CSS-rotated floor, we would need 
    // to offset vertices towards the CSS camera. But doing true 3D projection on a CSS-rotated canvas 
    // using 2D context is tricky.
    // Instead, since the canvas is CSS transformed, we can simulate height by drawing 
    // a stacked set of rectangles or applying a 2D offset that visually represents height.
    // For a CSS transform of rotateX(60deg) rotateZ(-45deg), the "up" direction in the viewport
    // corresponds to moving along the canvas +X and -Y axes in a specific way.
    // Let's use a simple fake-3D offset:
    const zOffsetX = h * Math.sin(Math.PI / 4);
    const zOffsetY = -h * Math.cos(Math.PI / 4);

    // Draw bottom face (shadow/base)
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(-l/2, -w/2, l, w);

    // Draw body
    ctx.fillStyle = 'rgba(56, 189, 248, 0.8)';
    ctx.beginPath();
    ctx.moveTo(-l/2, -w/2);
    ctx.lineTo(l/2, -w/2);
    ctx.lineTo(l/2 + zOffsetX, -w/2 + zOffsetY);
    ctx.lineTo(-l/2 + zOffsetX, -w/2 + zOffsetY);
    ctx.fill();

    ctx.fillStyle = 'rgba(45, 212, 191, 0.9)'; // Side
    ctx.beginPath();
    ctx.moveTo(-l/2, -w/2);
    ctx.lineTo(-l/2, w/2);
    ctx.lineTo(-l/2 + zOffsetX, w/2 + zOffsetY);
    ctx.lineTo(-l/2 + zOffsetX, -w/2 + zOffsetY);
    ctx.fill();

    // Top face
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.fillRect(-l/2 + zOffsetX, -w/2 + zOffsetY, l, w);

    // Front/Battery End (Colored block)
    ctx.fillStyle = 'rgba(239, 68, 68, 1.0)'; // Red block for battery end
    ctx.fillRect(l/2 - 10 + zOffsetX, -w/2 + zOffsetY, 10, w);

    // Display heading angle and turning radius on the top face
    ctx.save();
    ctx.translate(zOffsetX, zOffsetY);
    ctx.fillStyle = '#000';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.rotate(-heading); // counter rotate text to stay readable? Or keep it aligned.
    ctx.fillText(`HDG:${Math.round((heading * 180 / Math.PI) % 360)}°`, 0, -5);
    ctx.fillText(`R:${Math.round(pose.turningRadius || 0)}px`, 0, 8);
    ctx.restore();

    ctx.restore();
}

function drawTerrainGridAndPath() {
    if (!terrainGridData) return;
    const { rows, cols, cells, pathWaypoints } = terrainGridData;
    
    ctx.save();
    
    // Draw red boundary
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#ff2222';
    ctx.setLineDash([]);
    const tw = gridEnd.x - gridStart.x;
    const th = gridEnd.y - gridStart.y;
    ctx.strokeRect(gridStart.x, gridStart.y, tw, th);
    
    // Draw thin white lines dividing the boundary into cells
    ctx.strokeStyle = 'rgba(255,255,255,0.4)';
    ctx.lineWidth = 1.5;
    for(let i=1; i<cols; i++) {
        const x = gridStart.x + (tw / cols) * i;
        ctx.beginPath();
        ctx.moveTo(x, gridStart.y);
        ctx.lineTo(x, gridStart.y + th);
        ctx.stroke();
    }
    for(let i=1; i<rows; i++) {
        const y = gridStart.y + (th / rows) * i;
        ctx.beginPath();
        ctx.moveTo(gridStart.x, y);
        ctx.lineTo(gridStart.x + tw, y);
        ctx.stroke();
    }
    
    cells.forEach((cell, i) => {
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        const rowLabel = rows - cell.r; 
        const colLabel = String.fromCharCode(65 + cell.c); 
        ctx.fillText(`${colLabel}${rowLabel}`, cell.x + 5, cell.y + 5);
    });
    
    if (pathWaypoints.length > 0) {
        ctx.beginPath();
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.8)'; // Cyan
        
        const offset = (Date.now() / 20) % 20;
        ctx.setLineDash([15, 10]);
        ctx.lineDashOffset = -offset;
        
        pathWaypoints.forEach((wp, i) => {
            if (i === 0) ctx.moveTo(wp.x, wp.y);
            else ctx.lineTo(wp.x, wp.y);
        });
        ctx.stroke();
    }
    ctx.restore();
}

function drawRoverTracking(cvDetections, cvOriginalSize) {
    if (!cvDetections || cvDetections.length === 0 || !cvOriginalSize) return;
    const r = getVideoRect();
    if (!r) return;
    const scaleX = r.w / cvOriginalSize.width;
    const scaleY = r.h / cvOriginalSize.height;

    const rover = cvDetections[0];
    if (!rover) return;

    const b  = rover.box;
    const x  = r.x + b.x * scaleX;
    const y  = r.y + b.y * scaleY;
    const bw = b.width  * scaleX;
    const bh = b.height * scaleY;
    const cx = x + bw / 2;
    const cy = y + bh / 2;

    const pose = {
        x: cx,
        y: cy,
        heading: rover.heading || 0,
        confidence: rover.confidence || 1.0,
        turningRadius: rover.turningRadius || 15
    };

    drawRoverPrism(ctx, pose);
}

function drawGridOverlay() {
    if (gridStart && (gridEnd || currentGridEnd)) {
        const endPos = currentGridEnd || gridEnd;
        ctx.lineWidth = 4;
        ctx.strokeStyle = '#ff0000';
        
        const w = endPos.x - gridStart.x;
        const h = endPos.y - gridStart.y;
        
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(gridStart.x, gridStart.y, w, h);
        ctx.fillStyle = 'rgba(255, 0, 0, 0.05)';
        ctx.fillRect(gridStart.x, gridStart.y, w, h);
        ctx.setLineDash([]);
        
        ctx.fillStyle = '#fb923c';
        ctx.beginPath(); ctx.arc(gridStart.x, gridStart.y, 4, 0, 2*Math.PI); ctx.fill();
        ctx.beginPath(); ctx.arc(endPos.x, endPos.y, 4, 0, 2*Math.PI); ctx.fill();
    }
}

function renderLoop(state) {
    requestAnimationFrame(() => renderLoop(state));
    if (!state.feedActive) return;

    if (overlayCanvas.width !== video.clientWidth || overlayCanvas.height !== video.clientHeight) {
        overlayCanvas.width  = video.clientWidth;
        overlayCanvas.height = video.clientHeight;
    }
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const hasManualGrid = gridStart && (gridEnd || currentGridEnd);

    if (terrainGridData) {
        drawTerrainGridAndPath();
    } else if (hasManualGrid) {
        drawGridOverlay();
    }

    if (state.aiEnabled) {
        drawRoverTracking(state.cvDetections, state.cvOriginalSize);
    }
}

function getVideoRect() {
    if (!video.videoWidth || !video.videoHeight) return null;
    const cw = video.clientWidth;
    const ch = video.clientHeight;
    const vr = video.videoWidth / video.videoHeight;
    const cr = cw / ch;
    let dw, dh, ox, oy;
    if (vr > cr) { dw = cw; dh = cw / vr; ox = 0; oy = (ch - dh) / 2; }
    else         { dh = ch; dw = ch * vr; ox = (cw - dw) / 2; oy = 0; }
    return { x: ox, y: oy, w: dw, h: dh };
}

window.Overlay = {
    renderLoop,
    setTerrainGridData: (data) => terrainGridData = data,
    getTerrainGridData: () => terrainGridData,
    setGridStart: (pos) => gridStart = pos,
    setGridEnd: (pos) => gridEnd = pos,
    setCurrentGridEnd: (pos) => currentGridEnd = pos,
    getGridStart: () => gridStart,
    getGridEnd: () => gridEnd,
    getCurrentGridEnd: () => currentGridEnd,
    setDrawModeActive: (active) => {
        drawModeActive = active;
        overlayCanvas.style.cursor = drawModeActive ? 'crosshair' : 'default';
        overlayCanvas.style.pointerEvents = drawModeActive ? 'auto' : 'none';
    },
    isDrawModeActive: () => drawModeActive,
    setIsDrawing: (drawing) => isDrawing = drawing,
    getIsDrawing: () => isDrawing
};

// Canvas events
function getMousePos(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    return { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
}

overlayCanvas.addEventListener('mousedown', (e) => {
    if (!Overlay.isDrawModeActive()) return;
    const pos = getMousePos(overlayCanvas, e);
    Overlay.setIsDrawing(true);
    Overlay.setGridStart(pos);
    Overlay.setGridEnd(null);
    Overlay.setCurrentGridEnd(pos);
});

overlayCanvas.addEventListener('mousemove', (e) => {
    if (!Overlay.isDrawModeActive() || !Overlay.getIsDrawing()) return;
    Overlay.setCurrentGridEnd(getMousePos(overlayCanvas, e));
});

overlayCanvas.addEventListener('mouseup', (e) => {
    if (!Overlay.isDrawModeActive() || !Overlay.getIsDrawing()) return;
    Overlay.setIsDrawing(false);
    const pos = getMousePos(overlayCanvas, e);
    Overlay.setGridEnd(pos);
    Overlay.setCurrentGridEnd(null);
    const start = Overlay.getGridStart();
    const tX = Math.min(start.x, pos.x), tY = Math.min(start.y, pos.y);
    const bX = Math.max(start.x, pos.x), bY = Math.max(start.y, pos.y);
    Overlay.setGridStart({x: tX, y: tY}); 
    Overlay.setGridEnd({x: bX, y: bY});
});

overlayCanvas.addEventListener('mouseleave', (e) => {
    if (Overlay.getIsDrawing()) { 
        Overlay.setIsDrawing(false); 
        Overlay.setGridEnd(getMousePos(overlayCanvas, e)); 
        Overlay.setCurrentGridEnd(null); 
    }
});
