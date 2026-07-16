#!/bin/bash

# 1. Create and enter project directory
mkdir -p fastapi_ios_stream
cd fastapi_ios_stream

echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install -q fastapi uvicorn websockets

# 2. Get local IP address
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    LOCAL_IP=$(ipconfig getifaddr en0)
else
    # Linux
    LOCAL_IP=$(hostname -I | awk '{print $1}')
fi

if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP="localhost"
fi

# 3. Generate Self-Signed SSL Certificates (Fixes the iPhone camera error)
echo "Generating SSL certificates for HTTPS (Required by iOS Safari)..."
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout key.pem -out cert.pem \
    -subj "/C=US/ST=State/L=City/O=Dev/CN=$LOCAL_IP" 2>/dev/null

# 4. Create FastAPI Backend (main.py)
# This acts as a signaling server to connect the phone and laptop via WebRTC
cat << 'EOF' > main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()
connections = []

@app.get("/")
async def get_index():
    return HTMLResponse("<h1>Dashboards</h1><a href='/broadcaster'>1. iPhone Broadcaster</a><br><br><a href='/viewer'>2. Laptop Viewer</a>")

@app.get("/broadcaster")
async def get_broadcaster():
    with open("broadcaster.html", "r") as f:
        return HTMLResponse(f.read())

@app.get("/viewer")
async def get_viewer():
    with open("viewer.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Relay signaling messages to the other peer
            for conn in connections:
                if conn != websocket:
                    await conn.send_text(data)
    except WebSocketDisconnect:
        connections.remove(websocket)
EOF

# 5. Create iPhone Broadcaster Interface
cat << 'EOF' > broadcaster.html
<!DOCTYPE html>
<html>
<head>
    <title>iPhone Camera</title>
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <style>
        body { background: #000; color: #fff; font-family: -apple-system, sans-serif; text-align: center; margin: 0; padding: 20px;}
        video { width: 100%; border-radius: 12px; background: #222; }
        button { padding: 15px; width: 100%; font-size: 18px; margin-top: 20px; background: #0A84FF; border: none; color: white; border-radius: 12px; font-weight: bold;}
        #status { margin-top: 15px; color: #30D158; }
    </style>
</head>
<body>
    <video id="video" autoplay muted playsinline></video>
    <button id="startBtn">Start Broadcast</button>
    <div id="status">Ready.</div>

    <script>
        const ws = new WebSocket(`wss://${window.location.host}/ws`);
        let peerConnection;
        const config = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };

        document.getElementById('startBtn').onclick = async () => {
            try {
                // Constraints: Targets back camera, caps resolution at 720p 
                // This keeps bandwidth low for CV modeling while forcing the wide angle
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { 
                        facingMode: { ideal: "environment" },
                        width: { ideal: 1280, max: 1280 },
                        height: { ideal: 720, max: 720 },
                        frameRate: { ideal: 24, max: 30 }
                    },
                    audio: false
                });
                
                document.getElementById('video').srcObject = stream;
                document.getElementById('status').innerText = "Camera active. Sending...";

                peerConnection = new RTCPeerConnection(config);
                stream.getTracks().forEach(track => peerConnection.addTrack(track, stream));

                peerConnection.onicecandidate = e => {
                    if (e.candidate) ws.send(JSON.stringify({ candidate: e.candidate }));
                };

                const offer = await peerConnection.createOffer();
                await peerConnection.setLocalDescription(offer);
                ws.send(JSON.stringify({ offer: offer }));

            } catch (err) {
                document.getElementById('status').innerText = "Error: " + err.message;
                document.getElementById('status').style.color = "#FF453A";
            }
        };

        ws.onmessage = async (msg) => {
            const data = JSON.parse(msg.data);
            if (data.answer) {
                await peerConnection.setRemoteDescription(new RTCSessionDescription(data.answer));
                document.getElementById('status').innerText = "Live: Streaming to laptop!";
            } else if (data.candidate && peerConnection) {
                await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
            }
        };
    </script>
</body>
</html>
EOF

# 6. Create Laptop Viewer Interface
cat << 'EOF' > viewer.html
<!DOCTYPE html>
<html>
<head>
    <title>CV Dashboard Stream</title>
    <style>
        body { background: #1e1e1e; color: #fff; font-family: monospace; padding: 20px; }
        video { width: 100%; max-width: 900px; border: 2px solid #333; border-radius: 8px; }
    </style>
</head>
<body>
    <h2>Computer Vision Feed Intake</h2>
    <video id="video" autoplay playsinline></video>
    <div id="status" style="color: #4CAF50; margin-top:10px;">Awaiting WebRTC connection...</div>

    <script>
        const ws = new WebSocket(`wss://${window.location.host}/ws`);
        let peerConnection;
        const config = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };

        ws.onmessage = async (msg) => {
            const data = JSON.parse(msg.data);

            if (data.offer) {
                document.getElementById('status').innerText = "Negotiating secure stream...";
                peerConnection = new RTCPeerConnection(config);
                
                peerConnection.ontrack = e => {
                    document.getElementById('video').srcObject = e.streams[0];
                    document.getElementById('status').innerText = "Stream established. Ready for CV pipeline.";
                };

                peerConnection.onicecandidate = e => {
                    if (e.candidate) ws.send(JSON.stringify({ candidate: e.candidate }));
                };

                await peerConnection.setRemoteDescription(new RTCSessionDescription(data.offer));
                const answer = await peerConnection.createAnswer();
                await peerConnection.setLocalDescription(answer);
                ws.send(JSON.stringify({ answer: answer }));

            } else if (data.candidate && peerConnection) {
                await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
            }
        };
    </script>
</body>
</html>
EOF

# 7. Start the FastAPI Server
echo ""
echo "============================================================"
echo "🚀 SYSTEM READY"
echo "============================================================"
echo "1. On your LAPTOP, go to: https://$LOCAL_IP:8000/viewer"
echo "2. On your IPHONE, go to: https://$LOCAL_IP:8000/broadcaster"
echo "============================================================"
echo "⚠️ IMPORTANT iOS INSTRUCTION:"
echo "Because the SSL certificate is self-signed for local use, Safari will show a 'Connection Not Private' warning."
echo "Tap 'Show Details' -> 'visit this website' at the bottom to allow camera access."
echo "============================================================"
echo ""

# Run uvicorn with the generated SSL certificates
uvicorn main:app --host 0.0.0.0 --port 8000 --ssl-keyfile=key.pem --ssl-certfile=cert.pem