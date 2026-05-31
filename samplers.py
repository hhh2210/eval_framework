import json
import os
import time
import urllib.request
import fcntl
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


Message = Dict[str, Any]
MessageList = List[Message]


class SkipExampleError(Exception):
    """Raised when a single eval example should be skipped (e.g., content filter)."""
    pass


@dataclass
class SamplerResponse:
    response_text: str
    response_metadata: dict
    actual_queried_message_list: MessageList


class SamplerBase:
    def __call__(self, message_list: MessageList) -> SamplerResponse:
        raise NotImplementedError


class GlobalRequestThrottle:
    """Simple global throttle shared across threads/processes on one machine."""

    _thread_lock = threading.Lock()

    def __init__(self, min_interval_s: float, state_path: Optional[str] = None) -> None:
        self.min_interval_s = max(0.0, min_interval_s)
        self.state_path = state_path or os.getenv(
            "EVAL_THROTTLE_STATE_PATH",
            "/tmp/eval_framework_global_throttle.state",
        )

    def acquire(self) -> None:
        if self.min_interval_s <= 0:
            return

        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)

        # Book a time slot under lock, then sleep outside the lock so other
        # threads/processes can book their own slots concurrently.
        with self._thread_lock:
            with open(self.state_path, "a+", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    raw = f.read().strip()
                    next_allowed = float(raw) if raw else 0.0
                    now = time.monotonic()
                    my_slot = max(now, next_allowed)
                    new_next = my_slot + self.min_interval_s
                    f.seek(0)
                    f.truncate()
                    f.write(f"{new_next:.9f}")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        sleep_time = my_slot - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)


class VLLMChatSampler(SamplerBase):
    _global_throttle: Optional[GlobalRequestThrottle] = None

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: Optional[int] = 20,
        max_tokens: int = 2048,
        timeout: int = 1800,
        local: bool = False,
        tp_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_mem_util: float = 0.95,
        trust_remote_code: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "dummy")
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.local = local
        self.tp_size = tp_size
        self.max_model_len = max_model_len
        self.gpu_mem_util = gpu_mem_util
        self.trust_remote_code = trust_remote_code
        self._local_llm = None
        min_interval_s = float(os.getenv("MIN_INTERVAL_S", "0.005"))
        if VLLMChatSampler._global_throttle is None:
            VLLMChatSampler._global_throttle = GlobalRequestThrottle(min_interval_s=min_interval_s)

    def _init_local(self) -> None:
        if self._local_llm is not None:
            return
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError("vllm is required for local sampling") from exc
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "tensor_parallel_size": self.tp_size,
            "gpu_memory_utilization": self.gpu_mem_util,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        self._local_llm = LLM(**kwargs)

    def _call_local(self, message_list: MessageList) -> SamplerResponse:
        self._init_local()
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
        )
        outputs = self._local_llm.chat(message_list, params)
        text = outputs[0].outputs[0].text if outputs else ""
        return SamplerResponse(
            response_text=text or "",
            response_metadata={},
            actual_queried_message_list=message_list,
        )

    def _call_server(self, message_list: MessageList) -> SamplerResponse:
        url = f"{self.base_url.rstrip('/')}/chat/completions".replace("/v1/v1", "/v1")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": message_list,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        
        max_retries = 3
        last_exception: Exception | None = None

        for attempt in range(max_retries):
            try:
                return self._make_request(url, payload, message_list)
            except SkipExampleError:
                # This example should be skipped entirely; propagate up
                raise
            except urllib.error.HTTPError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    print(
                        f"Request failed with HTTP {e.code}, retrying in {wait_time}s... "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    # Last attempt failed, log detailed error info
                    self._log_error(e, url, payload, message_list)
                    raise
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(
                        f"Request failed with error: {e}, retrying in {wait_time}s... "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    raise

        # Should not reach here, but just in case
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Request failed after all retries")
    
    def _make_request(
        self,
        url: str,
        payload: Dict[str, Any],
        message_list: MessageList,
    ) -> SamplerResponse:
        if self._global_throttle is not None:
            self._global_throttle.acquire()
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            # Inspect error body to see if this example should be skipped
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass

            if error_body:
                try:
                    data = json.loads(error_body)
                except Exception:
                    data = {}
                error_obj = data.get("error") if isinstance(data, dict) else None
                # Aliyun DashScope content filter case:
                # message contains "Input data may contain inappropriate content"
                # and code is "data_inspection_failed"
                msg = ""
                code = ""
                if isinstance(error_obj, dict):
                    msg = str(error_obj.get("message") or "")
                    code = str(error_obj.get("code") or "")
                elif isinstance(error_obj, str):
                    msg = error_obj

                msg_lower = msg.lower()
                if (
                    "input data may contain inappropriate content" in msg_lower
                    or code == "data_inspection_failed"
                ):
                    # For this specific case, we skip this example entirely instead of failing
                    print(
                        "Skipping example due to content filter violation from judge API "
                        "(data_inspection_failed)."
                    )
                    raise SkipExampleError(msg)

            # For all other HTTP errors, re-raise for the retry logic
            raise

        elapsed = time.time() - start
        data = json.loads(raw)
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        return SamplerResponse(
            response_text=content or "",
            response_metadata={
                "usage": data.get("usage"),
                "latency": elapsed,
            },
            actual_queried_message_list=message_list,
        )
    
    def _log_error(
        self,
        e: urllib.error.HTTPError,
        url: str,
        payload: Dict[str, Any],
        message_list: MessageList
    ) -> None:
        """Log detailed error information for debugging."""
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            pass
        
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        
        # Log debugging information
        print(f"\n=== HTTP Error {e.code} Debug Info ===")
        print(f"URL: {url}")
        print(f"Model: {self.model}")
        print(f"Message count: {len(message_list)}")
        print(f"Payload size: {len(body)} bytes")
        print(f"Request headers: {dict(req.headers)}")
        if error_body:
            print(f"Error response body: {error_body}")
        else:
            print(f"Error response: {e.reason}")
        
        # Try to parse error body as JSON for better error message
        if error_body:
            try:
                error_data = json.loads(error_body)
                if "error" in error_data:
                    error_msg = error_data["error"]
                    if isinstance(error_msg, dict) and "message" in error_msg:
                        print(f"Error message: {error_msg['message']}")
                    elif isinstance(error_msg, str):
                        print(f"Error message: {error_msg}")
            except Exception:
                pass
        
        print("=" * 50)

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.local:
            return self._call_local(message_list)
        return self._call_server(message_list)
