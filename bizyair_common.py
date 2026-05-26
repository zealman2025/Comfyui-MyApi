import time
import asyncio
import requests
import json
import base64
import numpy as np
import io
from PIL import Image
import torch
import traceback
from typing import Tuple, List, Union

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    print("[Comfyui-MyApi] 警告: 未安装 aiohttp，BizyAir 节点将退化为同步模式（不影响功能但无法并发）")

try:
    from .bizyair_upload import upload_image_to_bizyair
except ImportError:
    from bizyair_upload import upload_image_to_bizyair

# ModelZoo OpenAPI 任务创建与轮询
MODELZOO_BASE_URL = "https://api.bizyair.cn/x/v1/modelzoo/tasks/openapi"

# NanoBanana2 标准 10 种比例（O.2 渠道版同样仅支持这 10 种）
NANOBANANA2_STANDARD_ASPECT_RATIOS = [
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
]

# NanoBanana2 B.2 官方/渠道 API 额外支持的 4 种比例
NANOBANANA2_EXTENDED_ASPECT_RATIOS = NANOBANANA2_STANDARD_ASPECT_RATIOS + [
    "4:1", "1:4", "8:1", "1:8",
]

# O.2 官方版 width/height 约束
O2_OFFICIAL_MIN_EDGE = 480
O2_OFFICIAL_MAX_EDGE = 3840
O2_OFFICIAL_MIN_PIXELS = 655_360
O2_OFFICIAL_MAX_PIXELS = 8_294_400
O2_OFFICIAL_MAX_RATIO = 3.0


def normalize_aspect_ratio(aspect_ratio: str, allowed: list, fallback: str = "1:1") -> str:
    """非法比例兜底为 fallback。"""
    return aspect_ratio if aspect_ratio in allowed else fallback


def validate_prompt_length(prompt: str, max_len: int, label: str = "提示词") -> str:
    """校验并返回 strip 后的 prompt。"""
    text = (prompt or "").strip()
    if not text:
        raise ValueError(f"请输入{label}")
    if len(text) > max_len:
        raise ValueError(f"{label}长度不能超过 {max_len} 字符（当前 {len(text)} 字符）")
    return text


def _modelzoo_task_payload(result: dict) -> dict:
    """
    解析 ModelZoo API JSON 响应体。
    BizyAir 常见格式：{ code, message, status: true, data: { request_id / status / outputs } }
    """
    if not isinstance(result, dict):
        return {}
    data = result.get("data")
    if isinstance(data, dict) and (
        "request_id" in data
        or "outputs" in data
        or isinstance(data.get("status"), str)
    ):
        return data
    return result


def _extract_request_id(result: dict) -> str:
    payload = _modelzoo_task_payload(result)
    request_id = payload.get("request_id") or result.get("request_id")
    return str(request_id).strip() if request_id else ""


def _extract_output_images(result: dict):
    payload = _modelzoo_task_payload(result)
    for source in (payload, result):
        outputs = source.get("outputs")
        if isinstance(outputs, dict):
            images = outputs.get("images")
            if images:
                return images
    return None


def _extract_task_status(result: dict) -> str:
    payload = _modelzoo_task_payload(result)
    status = payload.get("status")
    return status if isinstance(status, str) else ""


def _extract_error_message(result: dict) -> str:
    payload = _modelzoo_task_payload(result)
    for source in (payload, result):
        msg = source.get("message")
        if msg and msg not in ("Ok", "OK", "ok"):
            return str(msg)
        err = source.get("error")
        if isinstance(err, dict):
            em = err.get("message") or err.get("msg")
            if em:
                return str(em)
    return "未知错误"


def call_modelzoo_task(
    api_key: str,
    endpoint: str,
    payload: dict,
    log_prefix: str = "BizyAir"
) -> Tuple[List[str], str]:
    """
    提交 ModelZoo 异步任务，并在排队时重试，成功后进行轮询直到任务成功并获取图像 URL 列表。
    
    :param api_key: BizyAir API 密钥
    :param endpoint: 模型 endpoint，如 'bza-image-b2-base/text-to-image'
    :param payload: 请求参数体
    :param log_prefix: 打印日志时的前缀
    :return: (图像URL列表, request_id)
    """
    auth = api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"
    url = f"{MODELZOO_BASE_URL}/{endpoint}"
    headers = {
        "Authorization": auth,
        "Content-Type": "application/json",
        "X-Bizyair-Task-Async": "enable"
    }

    # 处理 30039(排队上限) / 30040(并行上限) 重试机制
    max_retries = 6
    retry_delay = 8
    response = None

    for attempt in range(max_retries + 1):
        print(f"[{log_prefix}] 发送请求到 ModelZoo API: {url} (尝试 {attempt + 1})")
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                raise Exception(f"任务提交连接错误: {e}")
            print(f"[{log_prefix}] 网络错误，将在 {retry_delay} 秒后重试: {e}")
            time.sleep(retry_delay)
            continue

        if resp.status_code in (200, 202):
            response = resp
            break

        # 尝试解析错误码以判断是否可以重试
        try:
            error_data = resp.json()
            code = error_data.get("code")
            if code is None and isinstance(error_data.get("error"), dict):
                code = error_data["error"].get("code")

            if code in (30039, 30040) or "30039" in str(error_data) or "30040" in str(error_data):
                print(f"[{log_prefix}] 服务排队或并行超限 ({code})，将在 {retry_delay} 秒后重试...")
                if attempt == max_retries:
                    raise Exception(f"任务创建失败 (排队或并行超限，已重试 {max_retries} 次): {error_data}")
                time.sleep(retry_delay)
                continue
        except Exception as parse_err:
            print(f"[{log_prefix}] 尝试解析重试错误码时异常: {parse_err}")

        # 如果是非 30039/30040 错误或已经试完，则直接抛出 API 错误
        raise Exception(f"ModelZoo API 请求失败 (HTTP {resp.status_code}): {resp.text[:500]}")

    if not response:
        raise Exception("未获得有效的 API 响应")

    result = response.json()
    api_code = result.get("code")
    if api_code not in (None, 20000) and result.get("status") is not True:
        raise Exception(f"ModelZoo 任务创建失败: {_extract_error_message(result)}")

    request_id = _extract_request_id(result)
    if not request_id:
        images = _extract_output_images(result)
        if images:
            return images, "sync"
        raise Exception(
            f"API 响应中没有 request_id 也没有 outputs.images 字段: {result}"
        )

    print(f"[{log_prefix}] 异步任务创建成功, request_id: {request_id}. 开始轮询状态...")

    # 开始轮询，最长等待 180 秒
    poll_url = f"{MODELZOO_BASE_URL}/{request_id}"
    poll_headers = {
        "Authorization": auth
    }
    
    start_time = time.time()
    max_wait = 180
    poll_interval = 2.5

    while time.time() - start_time < max_wait:
        time.sleep(poll_interval)
        try:
            poll_resp = requests.get(poll_url, headers=poll_headers, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"[{log_prefix}] 轮询网络异常，稍后重试: {e}")
            continue

        if not poll_resp.ok:
            print(f"[{log_prefix}] 轮询请求失败 (HTTP {poll_resp.status_code}), 继续等待")
            continue

        poll_data = poll_resp.json()
        status = _extract_task_status(poll_data)
        print(f"[{log_prefix}] 任务轮询状态: {status or poll_data.get('data', {}).get('status', 'unknown')}")

        if status == "Success":
            images = _extract_output_images(poll_data)
            if not images:
                raise Exception(f"任务状态为 Success, 但 outputs.images 为空: {poll_data}")
            return images, request_id
        elif status == "Failed":
            msg = _extract_error_message(poll_data)
            raise Exception(f"任务执行失败 (status=Failed): {msg}")
        elif status in ("Pending", "Running", "Saving"):
            continue
        elif not status:
            # 顶层 status:true 但 data 内尚无任务状态时继续等待
            continue
        else:
            print(f"[{log_prefix}] 获得未知任务状态: {status}，继续轮询")

    raise Exception(f"任务轮询超时，超过了最长等待时间 ({max_wait} 秒)。请到 BizyAir 官网检查 request_id: {request_id}")


def calculate_official_wh(aspect_ratio: str, resolution: str) -> Tuple[int, int]:
    """
    按 UI 的 aspect_ratio / resolution 计算 O.2 官方版 width / height。
    约束：16 倍数；单边 480–3840；宽高比 ≤ 3:1；总像素 655,360–8,294,400。
    """
    base_map = {"1K": 1024, "2K": 2048, "4K": 3840}
    long_edge = base_map.get(str(resolution).upper(), 2048)

    try:
        aw, ah = map(int, aspect_ratio.split(":"))
        if aw <= 0 or ah <= 0:
            raise ValueError
    except Exception:
        aw, ah = 1, 1

    ratio = max(aw / ah, ah / aw)
    if ratio > O2_OFFICIAL_MAX_RATIO:
        if aw >= ah:
            aw = int(round(ah * O2_OFFICIAL_MAX_RATIO))
        else:
            ah = int(round(aw * O2_OFFICIAL_MAX_RATIO))

    if aw >= ah:
        w = long_edge
        h = int(round(long_edge * ah / aw))
    else:
        h = long_edge
        w = int(round(long_edge * aw / ah))

    w = int(round(w / 16.0)) * 16
    h = int(round(h / 16.0)) * 16
    w = max(O2_OFFICIAL_MIN_EDGE, min(w, O2_OFFICIAL_MAX_EDGE))
    h = max(O2_OFFICIAL_MIN_EDGE, min(h, O2_OFFICIAL_MAX_EDGE))

    pixels = w * h
    if pixels > O2_OFFICIAL_MAX_PIXELS:
        scale = (O2_OFFICIAL_MAX_PIXELS / pixels) ** 0.5
        w = max(O2_OFFICIAL_MIN_EDGE, int(w * scale // 16) * 16)
        h = max(O2_OFFICIAL_MIN_EDGE, int(h * scale // 16) * 16)

    pixels = w * h
    if pixels < O2_OFFICIAL_MIN_PIXELS:
        scale = (O2_OFFICIAL_MIN_PIXELS / max(pixels, 1)) ** 0.5
        w = min(O2_OFFICIAL_MAX_EDGE, int(round(w * scale / 16)) * 16)
        h = min(O2_OFFICIAL_MAX_EDGE, int(round(h * scale / 16)) * 16)
        w = max(O2_OFFICIAL_MIN_EDGE, w)
        h = max(O2_OFFICIAL_MIN_EDGE, h)

    return w, h


def decode_image_from_url(image_url: str) -> torch.Tensor:
    """从给定的 URL 下载并解码图像，返回 ComfyUI 标准 Tensor [1, H, W, C]"""
    print(f"[BizyAir] 正在从 URL 下载并解码输出图片: {image_url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(image_url, headers=headers, timeout=60)
    response.raise_for_status()

    img = Image.open(io.BytesIO(response.content))
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = img_np[np.newaxis, ...]
    
    # 转换为 PyTorch Tensor (ComfyUI 需要)
    img_tensor = torch.from_numpy(img_tensor)
    return img_tensor


async def call_modelzoo_task_async(
    api_key: str,
    endpoint: str,
    payload: dict,
    log_prefix: str = "BizyAir",
) -> Tuple[List[str], str]:
    """
    异步版任务提交 + 轮询。使用 aiohttp + asyncio.sleep，让 ComfyUI 调度器可在等待期间并发执行其他节点。
    若未安装 aiohttp 则自动回退到 asyncio.to_thread 包装的同步版本。
    """
    if not HAS_AIOHTTP:
        return await asyncio.to_thread(call_modelzoo_task, api_key, endpoint, payload, log_prefix)

    auth = api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"
    submit_url = f"{MODELZOO_BASE_URL}/{endpoint}"
    submit_headers = {
        "Authorization": auth,
        "Content-Type": "application/json",
        "X-Bizyair-Task-Async": "enable",
    }
    poll_headers = {"Authorization": auth}

    max_retries = 6
    retry_delay = 8
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession() as session:
        result = None
        for attempt in range(max_retries + 1):
            print(f"[{log_prefix}] 异步提交 ModelZoo: {submit_url} (尝试 {attempt + 1})")
            try:
                async with session.post(
                    submit_url, headers=submit_headers, json=payload, timeout=timeout
                ) as resp:
                    if resp.status in (200, 202):
                        result = await resp.json()
                        break
                    body_text = await resp.text()
                    retried = False
                    try:
                        err = json.loads(body_text)
                        code = err.get("code")
                        if code is None and isinstance(err.get("error"), dict):
                            code = err["error"].get("code")
                        if code in (30039, 30040):
                            print(f"[{log_prefix}] 排队/并行超限 ({code})，{retry_delay}s 后重试...")
                            if attempt == max_retries:
                                raise Exception(f"任务创建失败 (重试 {max_retries} 次仍超限): {err}")
                            await asyncio.sleep(retry_delay)
                            retried = True
                    except (json.JSONDecodeError, ValueError):
                        pass
                    if retried:
                        continue
                    raise Exception(f"ModelZoo API 请求失败 (HTTP {resp.status}): {body_text[:500]}")
            except aiohttp.ClientError as e:
                if attempt == max_retries:
                    raise Exception(f"任务提交连接错误: {e}") from e
                print(f"[{log_prefix}] 网络错误，{retry_delay}s 后重试: {e}")
                await asyncio.sleep(retry_delay)

        if result is None:
            raise Exception("未获得有效的 API 响应")

        api_code = result.get("code")
        if api_code not in (None, 20000) and result.get("status") is not True:
            raise Exception(f"ModelZoo 任务创建失败: {_extract_error_message(result)}")

        request_id = _extract_request_id(result)
        if not request_id:
            images = _extract_output_images(result)
            if images:
                return images, "sync"
            raise Exception(f"API 响应中没有 request_id 也没有 outputs.images 字段: {result}")

        print(f"[{log_prefix}] 异步任务创建成功, request_id: {request_id}. 开始轮询...")

        poll_url = f"{MODELZOO_BASE_URL}/{request_id}"
        poll_timeout = aiohttp.ClientTimeout(total=30)
        start_time = time.time()
        max_wait = 180
        poll_interval = 2.5

        while time.time() - start_time < max_wait:
            await asyncio.sleep(poll_interval)
            try:
                async with session.get(
                    poll_url, headers=poll_headers, timeout=poll_timeout
                ) as poll_resp:
                    if not (200 <= poll_resp.status < 300):
                        print(f"[{log_prefix}] 轮询失败 (HTTP {poll_resp.status})")
                        continue
                    poll_data = await poll_resp.json()
            except aiohttp.ClientError as e:
                print(f"[{log_prefix}] 轮询网络异常，稍后重试: {e}")
                continue

            status = _extract_task_status(poll_data)
            print(f"[{log_prefix}] 任务轮询状态: {status or 'pending'}")

            if status == "Success":
                images = _extract_output_images(poll_data)
                if not images:
                    raise Exception(f"任务状态 Success, 但 outputs.images 为空: {poll_data}")
                return images, request_id
            elif status == "Failed":
                raise Exception(f"任务执行失败 (status=Failed): {_extract_error_message(poll_data)}")
            elif status in ("Pending", "Running", "Saving") or not status:
                continue
            else:
                print(f"[{log_prefix}] 获得未知任务状态: {status}，继续轮询")

    raise Exception(
        f"任务轮询超时（超过 {max_wait}s）。请到 BizyAir 检查 request_id: {request_id}"
    )


async def decode_image_from_url_async(image_url: str) -> torch.Tensor:
    """异步下载图像；解码部分用线程池避免阻塞事件循环。"""
    if not HAS_AIOHTTP:
        return await asyncio.to_thread(decode_image_from_url, image_url)

    print(f"[BizyAir] 异步下载输出图: {image_url}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url, headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            content = await resp.read()

    def _decode(buf: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(buf))
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr[np.newaxis, ...])

    return await asyncio.to_thread(_decode, content)


async def upload_comfyui_images_to_oss_async(
    images_list: list, api_key: str, log_prefix: str
) -> List[str]:
    """OSS 三步上传含 STS 签名计算，整体走线程池即可释放事件循环。"""
    return await asyncio.to_thread(
        upload_comfyui_images_to_oss, images_list, api_key, log_prefix
    )


def upload_comfyui_images_to_oss(images_list: list, api_key: str, log_prefix: str) -> List[str]:
    """
    将 ComfyUI 内连入的一组 IMAGE Tensor 转换为 bytes 并并行/依次上传到 BizyAir OSS，返回 HTTPS URL 列表。
    """
    oss_urls = []
    
    for idx, img in enumerate(images_list):
        if img is None:
            continue
        try:
            # 转换为 uint8 的 PIL Image 并保存为 PNG
            img_np = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
            if img_np.dtype != np.uint8:
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            
            if len(img_np.shape) == 4:
                img_np = img_np[0]
                
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                pil_img = Image.fromarray(img_np, "RGB")
            else:
                raise ValueError(f"不支持的图像形状: {img_np.shape}")
                
            # 检查和自适应压缩，与原 nanobanana 文件一致
            max_bytes = 10 * 1024 * 1024
            target_raw_bytes = int(max_bytes * 0.7)
            min_dim = 512
            
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG", optimize=True)
            raw_size = buf.tell()
            image_format = "PNG"
            
            # 如果偏大，按最长边比例缩放
            resize_attempts = 0
            while raw_size > target_raw_bytes and (pil_img.width > min_dim or pil_img.height > min_dim) and resize_attempts < 5:
                scale = max((target_raw_bytes / raw_size) ** 0.5, 0.3)
                new_w = max(int(pil_img.width * scale), min_dim)
                new_h = max(int(pil_img.height * scale), min_dim)
                if new_w == pil_img.width and new_h == pil_img.height:
                    new_w = max(int(pil_img.width * 0.75), min_dim)
                    new_h = max(int(pil_img.height * 0.75), min_dim)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                resize_attempts += 1
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG", optimize=True)
                raw_size = buf.tell()

            # 实在不行使用 JPEG 压缩画质
            if raw_size > target_raw_bytes:
                quality = 90
                while raw_size > target_raw_bytes and quality >= 40:
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
                    raw_size = buf.tell()
                    image_format = "JPEG"
                    quality -= 5
                    
            if raw_size > target_raw_bytes:
                raise ValueError(f"图片尺寸太大且经压缩仍超过阈值限制: {raw_size/1024/1024:.1f}MB")
                
            img_bytes = buf.getvalue()
            ext = "jpg" if image_format == "JPEG" else "png"
            
            # 调用 OSS 上传
            add_log = lambda t, m: print(f"[{log_prefix}][{t}] {m}")
            url = upload_image_to_bizyair(
                img_bytes,
                api_key,
                add_log_fn=add_log,
                file_name=f"comfyui_ref_{idx + 1}.{ext}"
            )
            oss_urls.append(url)
        except Exception as e:
            print(f"[{log_prefix}] 转换或上传参考图 {idx + 1} 失败: {e}")
            print(traceback.format_exc())
            raise e
            
    return oss_urls
