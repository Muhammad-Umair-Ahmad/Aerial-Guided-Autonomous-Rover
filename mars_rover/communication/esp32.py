import asyncio
import time
import logging
from enum import Enum
from mars_rover.config import Config

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger("MarsRover.ESP32")

class Direction(str, Enum):
    FORWARD  = "forward"
    REVERSE  = "reverse"
    LEFT     = "left"
    RIGHT    = "right"
    STOP     = "stop"

class MotorController:
    def __init__(self, esp32_ip: str = Config.ESP32_IP, default_speed: int = Config.DEFAULT_SPEED):
        self.esp32_ip = esp32_ip
        self.speed = default_speed
        self.last_command: str = "stop"
        self.last_command_time: float = 0
        self.min_command_interval: float = 0.10
        self.invert_forward_reverse: bool = True
        self._connected = False
        self._client = None

    @property
    def base_url(self) -> str:
        return f"http://{self.esp32_ip}"

    @property
    def is_configured(self) -> bool:
        return bool(self.esp32_ip)

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def _get_client(self):
        if self._client is None:
            if not HTTPX_AVAILABLE:
                return None
            self._client = httpx.AsyncClient(timeout=2.0)
        return self._client

    async def send_command(self, direction: Direction, duration_ms: int = 0) -> bool:
        if not self.is_configured:
            logger.warning("ESP32 IP not configured — command not sent")
            return False

        now = time.time()
        if now - self.last_command_time < self.min_command_interval:
            return True

        physical_direction = direction.value
        if self.invert_forward_reverse:
            if direction == Direction.FORWARD:
                physical_direction = "reverse"
            elif direction == Direction.REVERSE:
                physical_direction = "forward"

        url = f"{self.base_url}/{physical_direction}"
        if duration_ms > 0:
            url += f"?ms={duration_ms}"

        try:
            client = await self._get_client()
            if client:
                resp = await client.get(url)
                success = resp.status_code == 200
            else:
                success = await self._fallback_request(url)

            if success:
                self.last_command = physical_direction
                self.last_command_time = now
                self._connected = True
                return True
            else:
                logger.error(f"ESP32 returned non-200 for {physical_direction}")
                return False

        except Exception as e:
            self._connected = False
            logger.error(f"ESP32 communication failed: {e}")
            return False

    async def _fallback_request(self, url: str) -> bool:
        import urllib.request
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: urllib.request.urlopen(url, timeout=2))
            return True
        except Exception:
            return False

    async def stop(self) -> bool:
        return await self.send_command(Direction.STOP)

    async def set_speed(self, speed: int) -> bool:
        speed = max(0, min(255, speed))
        self.speed = speed

        if not self.is_configured:
            return False

        url = f"{self.base_url}/speed?v={speed}"
        try:
            client = await self._get_client()
            if client:
                resp = await client.get(url)
                return resp.status_code == 200
            else:
                return await self._fallback_request(url)
        except Exception as e:
            logger.error(f"Failed to set speed: {e}")
            return False

    async def check_heartbeat(self) -> bool:
        if not self.is_configured:
            return False

        url = f"{self.base_url}/status"
        try:
            client = await self._get_client()
            if client:
                resp = await client.get(url)
                self._connected = resp.status_code == 200
            else:
                self._connected = await self._fallback_request(url)
            return self._connected
        except Exception:
            self._connected = False
            return False

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
