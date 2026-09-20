from __future__ import annotations

import base64
import json
import math
import mimetypes
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULTS = {
    "minimax": {
        "endpoint": "https://api.minimax.io",
        "model": "MiniMax-Hailuo-2.3",
        "resolution": "768P",
        "api_key_env": "MINIMAX_API_KEY",
    },
    "seedance": {
        "endpoint": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seedance-2-0-260128",
        "resolution": "720p",
        "api_key_env": "ARK_API_KEY",
    },
}


class ProviderError(RuntimeError):
    pass


def data_uri(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if resolved.stat().st_size >= 20 * 1024 * 1024:
        raise ValueError(f"reference frame exceeds the 20 MB provider-safe limit: {resolved}")
    mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    if not mime.startswith("image/"):
        raise ValueError(f"reference frame is not an image: {resolved}")
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def resolve_reference(
    value: str | None, allow_upload: bool, base_dir: Path | None = None
) -> str | None:
    if not value:
        return None
    if not allow_upload:
        raise ValueError("reference upload requires --allow-reference-upload")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme in {"http", "https", "data"}:
        return value
    local_path = Path(value)
    if not local_path.is_absolute() and base_dir is not None:
        local_path = base_dir / local_path
    return data_uri(local_path)


def minimax_duration(requested: float) -> int:
    if requested <= 6:
        return 6
    if requested <= 10:
        return 10
    raise ValueError("MiniMax v1 supports at most 10 seconds per generated shot")


def seedance_duration(requested: float) -> int:
    if requested <= 0 or requested > 12:
        raise ValueError("Seedance shot duration must be within 12 seconds")
    return max(2, int(math.ceil(requested)))


def build_payload(
    provider: str,
    shot: dict,
    model: str,
    resolution: str,
    ratio: str,
    reference: str | None = None,
    generate_audio: bool = False,
    watermark: bool = False,
) -> dict:
    prompt = str(shot.get("visual_prompt", "")).strip()
    if not prompt:
        raise ValueError(f"shot {shot.get('id')} has no visual_prompt")
    requested = float(shot.get("requested_duration", 0))
    if provider == "minimax":
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "prompt_optimizer": False,
            "duration": minimax_duration(requested),
            "resolution": resolution,
        }
        if reference:
            payload["first_frame_image"] = reference
        return payload
    if provider == "seedance":
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if reference:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": reference},
                    "role": "first_frame",
                }
            )
        return {
            "model": model,
            "content": content,
            "duration": seedance_duration(requested),
            "ratio": ratio,
            "resolution": resolution,
            "generate_audio": generate_audio,
            "watermark": watermark,
        }
    raise ValueError(f"unsupported provider: {provider}")


def redact_payload(payload: dict) -> dict:
    rendered = json.loads(json.dumps(payload))
    if str(rendered.get("first_frame_image", "")).startswith("data:"):
        rendered["first_frame_image"] = "[local image data omitted]"
    for item in rendered.get("content", []):
        image_url = item.get("image_url", {}).get("url") if isinstance(item, dict) else None
        if isinstance(image_url, str) and image_url.startswith("data:"):
            item["image_url"]["url"] = "[local image data omitted]"
    return rendered


def request_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict | None = None,
    timeout: float = 60.0,
) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "video-evidence/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ProviderError(f"provider returned HTTP {error.code}: {detail[:1000]}") from error
    except urllib.error.URLError as error:
        raise ProviderError(f"provider request failed: {error.reason}") from error


def check_minimax_response(response: dict) -> None:
    base = response.get("base_resp", {})
    status_code = base.get("status_code", 0)
    if status_code not in (0, None):
        raise ProviderError(
            f"MiniMax error {status_code}: {base.get('status_msg', 'unknown error')}"
        )


def submit_job(provider: str, endpoint: str, api_key: str, payload: dict) -> dict:
    base = endpoint.rstrip("/")
    if provider == "minimax":
        response = request_json("POST", f"{base}/v1/video_generation", api_key, payload)
        check_minimax_response(response)
        task_id = response.get("task_id")
    elif provider == "seedance":
        response = request_json(
            "POST", f"{base}/contents/generations/tasks", api_key, payload
        )
        task_id = response.get("id")
    else:
        raise ValueError(f"unsupported provider: {provider}")
    if not task_id:
        raise ProviderError(f"provider response has no task id: {response}")
    return {"task_id": str(task_id), "response": response}


def normalize_status(value: str | None) -> str:
    status = str(value or "unknown").casefold()
    if status in {"preparing", "queueing", "queued", "pending"}:
        return "queued"
    if status in {"processing", "running", "in_progress"}:
        return "running"
    if status in {"success", "succeeded", "completed"}:
        return "succeeded"
    if status in {"fail", "failed", "cancelled", "canceled"}:
        return "failed"
    return "unknown"


def query_job(provider: str, endpoint: str, api_key: str, task_id: str) -> dict:
    base = endpoint.rstrip("/")
    if provider == "minimax":
        query = urllib.parse.urlencode({"task_id": task_id})
        response = request_json(
            "GET", f"{base}/v1/query/video_generation?{query}", api_key
        )
        check_minimax_response(response)
        status = normalize_status(response.get("status"))
        download_url = None
        if status == "succeeded" and response.get("file_id"):
            file_query = urllib.parse.urlencode({"file_id": response["file_id"]})
            file_response = request_json(
                "GET", f"{base}/v1/files/retrieve?{file_query}", api_key
            )
            check_minimax_response(file_response)
            download_url = file_response.get("file", {}).get("download_url")
            response["file_response"] = file_response
    elif provider == "seedance":
        response = request_json(
            "GET", f"{base}/contents/generations/tasks/{urllib.parse.quote(task_id)}", api_key
        )
        status = normalize_status(response.get("status"))
        download_url = response.get("content", {}).get("video_url")
    else:
        raise ValueError(f"unsupported provider: {provider}")
    return {
        "status": status,
        "provider_status": response.get("status"),
        "download_url": download_url,
        "response": response,
    }


def download_file(url: str, output: Path, timeout: float = 300.0) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "video-evidence/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as destination:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
        temporary.replace(output)
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        raise ProviderError(f"video download failed: {error}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
