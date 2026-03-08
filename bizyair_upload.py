"""
BizyAir 图片上传工具模块
将图片上传到阿里云 OSS 并获取可访问的 URL，替代 base64 直传方式

上传流程（三步）：
1. GET  /x/v1/upload/token          → 获取 OSS 临时凭证 (STS)
2. PUT  到阿里云 OSS                → 使用临时凭证上传文件
3. POST /x/v1/input_resource/commit  → 提交资源获取最终 URL

与主插件 ZealmanAIforPS 的 bizyair-upload.ts 保持一致
"""

import base64
import hmac
import hashlib
import re
import time
from urllib.parse import quote
from typing import Callable, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def _add_log(add_log_fn: Optional[Callable[[str, str], None]], log_type: str, message: str) -> None:
    """输出日志，若 add_log_fn 为 None 则使用 print"""
    if add_log_fn:
        add_log_fn(log_type, message)
    else:
        print(f"[{log_type}] {message}")


def _oss_signature_with_x_date(
    secret: str,
    method: str,
    content_type: str,
    date_value: str,
    security_token: str,
    resource: str
) -> str:
    """
    计算 OSS V1 签名（与主插件 bizyair-upload.ts 的 ossSignatureWithXDate 一致）
    当使用 x-oss-date 替代 Date 头时，StringToSign 格式：
      VERB \\n
      Content-MD5 \\n
      Content-Type \\n
      Date (x-oss-date 的值) \\n
      x-oss-date:...\\n
      x-oss-security-token:...\\n
      CanonicalizedResource
    """
    sts = (
        f"{method}\n"
        "\n"
        f"{content_type}\n"
        f"{date_value}\n"
        f"x-oss-date:{date_value}\n"
        f"x-oss-security-token:{security_token}\n"
        f"{resource}"
    )
    sig_bytes = hmac.new(
        secret.encode('utf-8'),
        sts.encode('utf-8'),
        hashlib.sha1
    ).digest()
    return base64.b64encode(sig_bytes).decode('utf-8')


def _get_upload_token(
    file_name: str,
    api_key: str,
    add_log_fn: Optional[Callable[[str, str], None]] = None
) -> dict:
    """获取 OSS 上传凭证"""
    auth = api_key if api_key.startswith('Bearer ') else f"Bearer {api_key}"
    url = f"https://api.bizyair.cn/x/v1/upload/token?file_name={quote(file_name)}&file_type=inputs"

    _add_log(add_log_fn, "上传", f"[1/3] 获取上传凭证: {file_name}")
    resp = requests.get(url, headers={"Authorization": auth}, timeout=30)

    if not resp.ok:
        txt = resp.text if hasattr(resp, 'text') else ''
        msg = f"获取上传凭证失败 (HTTP {resp.status_code})"
        try:
            j = resp.json()
            msg = j.get('message') or j.get('error') or msg
        except Exception:
            pass
        raise RuntimeError(msg)

    body = resp.json()
    if not body.get("status") or not body.get("data"):
        raise RuntimeError(body.get("message", "获取上传凭证失败"))

    data = body["data"]
    obj_key = data.get("file", {}).get("object_key", "")
    _add_log(add_log_fn, "上传", f"凭证获取成功, object_key: {obj_key[:40]}...")
    return data


def _upload_to_oss(
    image_bytes: bytes,
    cred: dict,
    add_log_fn: Optional[Callable[[str, str], None]] = None,
    file_name: Optional[str] = None
) -> None:
    """将图片上传到阿里云 OSS"""
    file_data = cred.get("file", {})
    storage_data = cred.get("storage", {})

    object_key = file_data.get("object_key", "")
    access_key_id = file_data.get("access_key_id", "")
    access_key_secret = file_data.get("access_key_secret", "")
    security_token = file_data.get("security_token", "")

    endpoint = storage_data.get("endpoint", "")
    bucket = storage_data.get("bucket", "")
    region = storage_data.get("region", "")

    # 构建 OSS endpoint
    norm_region = region[4:] if region and region.startswith("oss-") else (region or "")
    base_host = endpoint or f"oss-{norm_region}.aliyuncs.com"
    base_host = base_host.replace("https://", "").replace("http://", "")
    if base_host.startswith(f"{bucket}."):
        base_host = base_host[len(bucket) + 1:]

    oss_url = f"https://{bucket}.{base_host}/{object_key}"

    # Content-Type
    content_type = "application/octet-stream"
    if file_name:
        ext = file_name.lower().split(".")[-1] if "." in file_name else ""
        if ext in ("jpg", "jpeg"):
            content_type = "image/jpeg"
        elif ext == "png":
            content_type = "image/png"
        elif ext == "gif":
            content_type = "image/gif"
        elif ext == "webp":
            content_type = "image/webp"

    # x-oss-date (RFC 2616 GMT)
    from email.utils import formatdate
    date_value = formatdate(usegmt=True)

    canon_resource = f"/{bucket}/{object_key}"
    sig = _oss_signature_with_x_date(
        access_key_secret, "PUT", content_type,
        date_value, security_token, canon_resource
    )

    _add_log(add_log_fn, "上传", f"[2/3] 上传到 OSS ({len(image_bytes) / 1024:.1f} KB)...")

    headers = {
        "Content-Type": content_type,
        "x-oss-date": date_value,
        "x-oss-security-token": security_token,
        "Authorization": f"OSS {access_key_id}:{sig}",
    }

    resp = requests.put(oss_url, headers=headers, data=image_bytes, timeout=60)

    if not resp.ok:
        txt = resp.text[:500] if resp.text else ""
        error_msg = f"OSS 上传失败 (HTTP {resp.status_code})"
        # 尝试解析 OSS XML 错误
        code_match = re.search(r"<Code>([^<]+)</Code>", txt)
        msg_match = re.search(r"<Message>([^<]+)</Message>", txt)
        if code_match and msg_match:
            error_msg = f"OSS 上传失败: {code_match.group(1)} - {msg_match.group(1)}"
        elif txt:
            error_msg = f"{error_msg}: {txt[:300]}"
        _add_log(add_log_fn, "错误", f"OSS 上传失败详情: URL={oss_url}, Status={resp.status_code}")
        raise RuntimeError(error_msg)

    _add_log(add_log_fn, "上传", "OSS 上传完成")


def _commit_resource(
    file_name: str,
    object_key: str,
    api_key: str,
    add_log_fn: Optional[Callable[[str, str], None]] = None
) -> str:
    """提交资源，获取最终可访问的 URL"""
    auth = api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"
    _add_log(add_log_fn, "上传", f"[3/3] 提交资源: {file_name}")

    resp = requests.post(
        "https://api.bizyair.cn/x/v1/input_resource/commit",
        headers={"Authorization": auth, "Content-Type": "application/json"},
        json={"name": file_name, "object_key": object_key},
        timeout=30
    )

    if not resp.ok:
        txt = resp.text if resp.text else ""
        msg = f"提交资源失败 (HTTP {resp.status_code})"
        try:
            j = resp.json()
            msg = j.get("message") or j.get("error") or msg
        except Exception:
            pass
        raise RuntimeError(msg)

    body = resp.json()
    if not body.get("status") or not body.get("data") or not body["data"].get("url"):
        raise RuntimeError(body.get("message", "提交资源失败，未返回 URL"))

    url = body["data"]["url"]
    _add_log(add_log_fn, "上传", f"资源 URL 获取成功: {url[:60]}...")
    return url


def upload_image_to_bizyair(
    image_bytes: bytes,
    api_key: str,
    add_log_fn: Optional[Callable[[str, str], None]] = None,
    file_name: Optional[str] = None
) -> str:
    """
    上传图片到 BizyAir OSS，返回可访问的 URL

    :param image_bytes: 图片二进制数据
    :param api_key: BizyAir API 密钥
    :param add_log_fn: 可选，日志回调 (log_type, message) -> None
    :param file_name: 可选，文件名（用于 Content-Type 和提交）
    :return: 可访问的 OSS URL
    """
    if not HAS_REQUESTS:
        raise RuntimeError("缺少 requests 库，请执行: pip install requests")

    name = file_name or f"comfyui_image_{int(time.time() * 1000)}.jpg"
    _add_log(add_log_fn, "上传", f"开始上传图片 ({len(image_bytes) / 1024:.1f} KB)...")

    cred = _get_upload_token(name, api_key, add_log_fn)
    _upload_to_oss(image_bytes, cred, add_log_fn, name)
    return _commit_resource(name, cred["file"]["object_key"], api_key, add_log_fn)
