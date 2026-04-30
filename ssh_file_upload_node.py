import json
import os
import posixpath
import re
import shlex
import tempfile
import time
import traceback
import uuid
import wave
from urllib.parse import quote

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False


class AnyType(str):
    """ComfyUI 通配输入类型，允许连接任意上游输出。"""

    def __ne__(self, other):
        return False


ANY_TYPE = AnyType("*")


class SSHFileUploadNode:
    """通过 SSH/SFTP 将任意本地文件上传到目标服务器。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anyting": (ANY_TYPE,),
                "ssh_command": (
                    "STRING",
                    {
                        "default": "ssh -p 21656 root@connect.bjb2.seetacloud.com",
                        "multiline": False,
                    },
                ),
                "server": ("STRING", {"default": "", "multiline": False}),
                "port": ("INT", {"default": 22, "min": 1, "max": 65535}),
                "username": ("STRING", {"default": "", "multiline": False}),
                "password": ("STRING", {"default": "", "multiline": False, "password": True}),
                "remote_dir": ("STRING", {"default": "/root", "multiline": False}),
                "remote_filename": (
                    "STRING",
                    {
                        "default": "{stem}_{timestamp_ms}{ext}",
                        "multiline": False,
                        "placeholder": "支持 {basename} {stem} {ext} {timestamp_ms} {uuid}",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("upload_info",)
    FUNCTION = "upload"
    CATEGORY = "🍎MYAPI"

    @classmethod
    def IS_CHANGED(
        cls,
        anyting,
        ssh_command,
        server,
        port,
        username,
        password,
        remote_dir,
        remote_filename,
    ):
        # 上传是副作用操作，每次执行工作流都应重新运行。
        return time.time()

    def _parse_ssh_command(self, command):
        result = {}
        if not command or not command.strip():
            return result

        command = command.strip()
        if re.fullmatch(r"[^@\s]+@[^@\s]+", command):
            user_host = command
            parts = []
        else:
            try:
                parts = shlex.split(command)
            except ValueError:
                parts = command.split()
            if parts and parts[0].lower() == "ssh":
                parts = parts[1:]
            user_host = ""

        index = 0
        while index < len(parts):
            part = parts[index]
            if part in ("-p", "-P") and index + 1 < len(parts):
                try:
                    result["port"] = int(parts[index + 1])
                except ValueError:
                    pass
                index += 2
                continue
            if part.startswith("-p") and len(part) > 2:
                try:
                    result["port"] = int(part[2:])
                except ValueError:
                    pass
                index += 1
                continue
            if not part.startswith("-"):
                user_host = part
            index += 1

        if user_host:
            if "@" in user_host:
                user, host = user_host.rsplit("@", 1)
                if user:
                    result["username"] = user
                if host:
                    result["server"] = host
            else:
                result["server"] = user_host

        return result

    def _resolve_connection(self, ssh_command, server, port, username):
        parsed = self._parse_ssh_command(ssh_command)
        resolved_server = (server or "").strip() or parsed.get("server", "")
        resolved_username = (username or "").strip() or parsed.get("username", "")
        resolved_port = int(port or parsed.get("port", 22))

        if parsed.get("port") and (not port or int(port) == 22):
            resolved_port = int(parsed["port"])

        if not resolved_server:
            raise ValueError("请填写 SSH 服务器地址，或在 ssh_command 中粘贴 ssh 登录命令。")
        if not resolved_username:
            raise ValueError("请填写 SSH 用户名，或使用 root@host 这类 SSH 命令格式。")

        return resolved_server, resolved_port, resolved_username

    def _extract_file_path(self, value):
        if value is None:
            return ""

        if isinstance(value, str):
            return value.strip().strip('"').strip("'")

        if isinstance(value, os.PathLike):
            return os.fspath(value)

        if isinstance(value, dict):
            for key in ("path", "file_path", "filename", "name"):
                candidate = value.get(key)
                extracted = self._extract_file_path(candidate)
                if extracted:
                    return extracted

        if isinstance(value, (list, tuple)) and value:
            for item in value:
                extracted = self._extract_file_path(item)
                if extracted:
                    return extracted

        for attr in ("path", "file_path", "filename", "name"):
            if hasattr(value, attr):
                extracted = self._extract_file_path(getattr(value, attr))
                if extracted:
                    return extracted

        return ""

    def _normalize_existing_path(self, path):
        if not path:
            return ""

        candidate = os.path.expanduser(os.path.expandvars(str(path).strip().strip('"').strip("'")))
        if not candidate:
            return ""
        if not os.path.isabs(candidate):
            candidate = os.path.abspath(candidate)
        return candidate if os.path.isfile(candidate) or os.path.isdir(candidate) else ""

    def _temp_dir(self):
        temp_dir = os.path.join(tempfile.gettempdir(), "comfyui_myapi_ssh_upload")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _write_temp_file(self, suffix, data, mode="wb"):
        temp_path = os.path.join(
            self._temp_dir(),
            f"upload_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{suffix}",
        )
        open_kwargs = {"encoding": "utf-8"} if "b" not in mode else {}
        with open(temp_path, mode, **open_kwargs) as file_obj:
            file_obj.write(data)
        return temp_path

    def _to_numpy(self, value):
        if HAS_TORCH and isinstance(value, torch.Tensor):
            if value.is_cuda:
                value = value.cpu()
            return value.detach().numpy()
        if HAS_NUMPY and isinstance(value, np.ndarray):
            return value
        return None

    def _save_image_to_temp_file(self, value):
        if not HAS_PIL or not HAS_NUMPY:
            raise ImportError("上传图像张量需要 Pillow 和 numpy 依赖。")

        if HAS_PIL and isinstance(value, Image.Image):
            temp_path = self._write_temp_file(".png", b"")
            value.convert("RGB").save(temp_path, "PNG")
            return temp_path

        image = self._to_numpy(value)
        if image is None:
            return ""

        if image.ndim == 4:
            if image.shape[0] < 1:
                raise ValueError("图像批次为空，无法上传。")
            image = image[0]

        if image.ndim != 3:
            return ""

        if image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] != 3:
            return ""

        if image.dtype in (np.float32, np.float64) or image.max() <= 1.0:
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

        temp_path = self._write_temp_file(".png", b"")
        Image.fromarray(image, "RGB").save(temp_path, "PNG")
        return temp_path

    def _save_audio_to_temp_file(self, value):
        if not isinstance(value, dict) or "waveform" not in value:
            return ""

        if not HAS_NUMPY:
            raise ImportError("上传音频张量需要 numpy 依赖。")

        sample_rate = int(value.get("sample_rate") or value.get("sampling_rate") or 44100)
        audio = self._to_numpy(value.get("waveform"))
        if audio is None:
            return ""

        if audio.ndim == 3:
            audio = audio[0]
        if audio.ndim == 1:
            audio = audio[:, None]
        elif audio.ndim == 2 and audio.shape[0] <= 8:
            audio = audio.T
        elif audio.ndim != 2:
            return ""

        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)
        temp_path = self._write_temp_file(".wav", b"")
        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(int(pcm.shape[1]))
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return temp_path

    def _save_content_to_temp_file(self, value):
        if isinstance(value, (bytes, bytearray, memoryview)):
            return self._write_temp_file(".bin", bytes(value))

        if isinstance(value, str):
            existing_path = self._normalize_existing_path(value)
            if existing_path:
                return existing_path
            return self._write_temp_file(".txt", value, mode="w")

        if isinstance(value, dict):
            audio_path = self._save_audio_to_temp_file(value)
            if audio_path:
                return audio_path
            for key in ("text", "content", "string"):
                if isinstance(value.get(key), str):
                    return self._write_temp_file(".txt", value[key], mode="w")
            return self._write_temp_file(
                ".json",
                json.dumps(value, ensure_ascii=False, default=str),
                mode="w",
            )

        if isinstance(value, (list, tuple)) and value:
            for item in value:
                try:
                    resolved = self._resolve_local_path(item)
                    if resolved:
                        return resolved
                except Exception:
                    continue

        image_path = self._save_image_to_temp_file(value)
        if image_path:
            return image_path

        return ""

    def _resolve_local_path(self, anyting):
        candidate = self._normalize_existing_path(self._extract_file_path(anyting))
        if not candidate:
            candidate = self._save_content_to_temp_file(anyting)
        if not candidate:
            raise ValueError("无法把 anyting 输入解析为可上传文件。请连接文件路径、图像、音频、文本或 bytes 类型输出。")

        if not os.path.isfile(candidate) and not os.path.isdir(candidate):
            raise FileNotFoundError(f"本地路径不存在或不是文件/目录: {candidate}")

        return candidate

    def _get_local_stats(self, local_path):
        if os.path.isfile(local_path):
            return os.path.getsize(local_path), 1

        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(local_path, followlinks=False):
            dirs[:] = [name for name in dirs if not os.path.islink(os.path.join(root, name))]
            for name in files:
                path = os.path.join(root, name)
                if os.path.islink(path) or not os.path.isfile(path):
                    continue
                total_size += os.path.getsize(path)
                file_count += 1
        return total_size, file_count

    def _format_remote_filename(self, remote_filename, local_path):
        basename = os.path.basename(local_path)
        stem, ext = os.path.splitext(basename)
        timestamp_ms = str(int(time.time() * 1000))
        template = (remote_filename or "").strip() or "{stem}_{timestamp_ms}{ext}"
        values = {
            "basename": basename,
            "stem": stem,
            "ext": ext,
            "timestamp_ms": timestamp_ms,
            "timestamp": timestamp_ms,
            "uuid": uuid.uuid4().hex,
        }

        try:
            return template.format(**values)
        except KeyError as exc:
            raise ValueError(f"remote_filename 包含不支持的占位符: {{{exc.args[0]}}}") from exc

    def _normalize_remote_path(self, remote_dir, remote_filename, local_path):
        safe_dir = (remote_dir or "").strip().replace("\\", "/") or "."
        safe_name = self._format_remote_filename(remote_filename, local_path).replace("\\", "/")
        if not safe_name:
            raise ValueError("无法确定远程文件名，请填写 remote_filename。")

        if safe_name.startswith("/") or "/" in safe_name:
            remote_path = safe_name.replace("\\", "/")
        else:
            remote_path = posixpath.join(safe_dir, safe_name)

        return posixpath.normpath(remote_path)

    def _mkdir_p(self, sftp, remote_dir):
        remote_dir = posixpath.normpath(remote_dir.replace("\\", "/"))
        if remote_dir in ("", ".", "/"):
            return

        current = "/" if remote_dir.startswith("/") else ""
        for part in [p for p in remote_dir.split("/") if p]:
            current = posixpath.join(current, part) if current else part
            try:
                sftp.stat(current)
            except IOError:
                sftp.mkdir(current)

    def _remote_exists(self, sftp, remote_path):
        try:
            sftp.stat(remote_path)
            return True
        except IOError:
            return False

    def _unique_remote_path(self, sftp, remote_path):
        if not self._remote_exists(sftp, remote_path):
            return remote_path

        remote_dir = posixpath.dirname(remote_path)
        basename = posixpath.basename(remote_path)
        stem, ext = posixpath.splitext(basename)
        for index in range(1, 10000):
            candidate = posixpath.join(remote_dir, f"{stem}_{index:03d}{ext}")
            if not self._remote_exists(sftp, candidate):
                return candidate

        raise FileExistsError(f"远程文件名冲突过多，无法生成唯一文件名: {remote_path}")

    def _upload_directory(self, sftp, local_dir, remote_dir):
        uploaded_size = 0
        uploaded_count = 0
        self._mkdir_p(sftp, remote_dir)

        for root, dirs, files in os.walk(local_dir, followlinks=False):
            dirs[:] = [name for name in dirs if not os.path.islink(os.path.join(root, name))]
            rel_dir = os.path.relpath(root, local_dir)
            current_remote_dir = remote_dir if rel_dir == "." else posixpath.join(
                remote_dir,
                rel_dir.replace("\\", "/"),
            )
            self._mkdir_p(sftp, current_remote_dir)

            for name in files:
                local_file = os.path.join(root, name)
                if os.path.islink(local_file) or not os.path.isfile(local_file):
                    continue
                remote_file = posixpath.join(current_remote_dir, name)
                sftp.put(local_file, remote_file)
                uploaded_size += sftp.stat(remote_file).st_size
                uploaded_count += 1

        return uploaded_size, uploaded_count

    def upload(
        self,
        anyting,
        ssh_command,
        server,
        port,
        username,
        password,
        remote_dir,
        remote_filename,
    ):
        local_path = ""
        remote_path = ""
        file_size = 0
        file_count = 0

        try:
            if not HAS_PARAMIKO:
                raise ImportError("缺少 paramiko 依赖，请执行: pip install paramiko")

            local_path = self._resolve_local_path(anyting)
            file_size, file_count = self._get_local_stats(local_path)
            host, resolved_port, resolved_username = self._resolve_connection(
                ssh_command,
                server,
                port,
                username,
            )
            if not password:
                raise ValueError("请填写 SSH 密码。")

            remote_path = self._normalize_remote_path(remote_dir, remote_filename, local_path)
            remote_parent = posixpath.dirname(remote_path)

            print(f"[SSHFileUpload] 上传 {local_path} -> {resolved_username}@{host}:{remote_path}")

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                ssh.connect(
                    hostname=host,
                    port=int(resolved_port),
                    username=resolved_username,
                    password=password,
                    timeout=30,
                    banner_timeout=30,
                    auth_timeout=30,
                    look_for_keys=False,
                    allow_agent=False,
                )
                sftp = ssh.open_sftp()
                try:
                    self._mkdir_p(sftp, remote_parent)
                    remote_path = self._unique_remote_path(sftp, remote_path)
                    if os.path.isdir(local_path):
                        uploaded_size, uploaded_count = self._upload_directory(
                            sftp,
                            local_path,
                            remote_path,
                        )
                    else:
                        sftp.put(local_path, remote_path)
                        uploaded_size = sftp.stat(remote_path).st_size
                        uploaded_count = 1
                finally:
                    sftp.close()
            finally:
                ssh.close()

            file_url = f"sftp://{quote(resolved_username)}@{host}:{resolved_port}{remote_path}"
            upload_info = {
                "success": True,
                "server": host,
                "port": int(resolved_port),
                "username": resolved_username,
                "local_path": local_path,
                "remote_path": remote_path,
                "file_url": file_url,
                "is_directory": os.path.isdir(local_path),
                "file_size": int(file_size),
                "file_count": int(file_count),
                "uploaded_size": int(uploaded_size),
                "uploaded_count": int(uploaded_count),
            }

            return (json.dumps(upload_info, ensure_ascii=False),)

        except Exception as exc:
            error_message = str(exc)
            print(f"[SSHFileUpload] 上传失败: {error_message}")
            print(traceback.format_exc())
            upload_info = {
                "success": False,
                "local_path": local_path,
                "remote_path": remote_path,
                "is_directory": os.path.isdir(local_path) if local_path else False,
                "file_size": int(file_size),
                "file_count": int(file_count),
                "error": error_message,
            }
            return (json.dumps(upload_info, ensure_ascii=False),)


NODE_CLASS_MAPPINGS = {
    "SSHFileUploadNode": SSHFileUploadNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SSHFileUploadNode": "📤SSH 文件上传",
}
