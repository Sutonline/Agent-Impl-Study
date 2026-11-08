from __future__ import annotations

"""
Step 1 核心组件：LLM 客户端封装 (harness/llm.py)

第一性原理视角：
LLM 本身只是一个“无状态的函数”：
    (messages, tools_schema) -> LLMResponse(content, tool_calls)
它不会自己执行任何工具，也不会记住上一轮对话（除非你把历史 messages 再传给它）。
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request


def load_env_file(dotenv_path: Path | None = None) -> None:
    """加载 .env 文件（优先使用 python-dotenv，若未安装则使用标准库解析）。"""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=dotenv_path)
        return
    except ImportError:
        pass

    candidates = [dotenv_path] if dotenv_path else [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]
    for path in candidates:
        if path and path.is_file():
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'").strip('"')
                os.environ.setdefault(key, val)
            break


@dataclass
class ToolCall:
    """LLM 发出的单次工具调用意图（注意：这只是意图，还没真正执行！）。"""
    id: str
    name: str
    arguments: str  # JSON 字符串，如 '{"expression": "6 * 7"}'


@dataclass
class LLMResponse:
    """统一封装 LLM 的单次响应结果。"""
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0


class LLMClient:
    """
    OpenAI Chat Completions 极简客户端。
    优先使用官方 `openai` SDK；若当前环境尚未 `pip install openai`，自动回退到标准库 `urllib.request` 直连 HTTP 接口。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        load_env_file()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        self._sdk_client: Any = None
        try:
            from openai import OpenAI  # type: ignore
            if self.api_key:
                self._sdk_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            self._sdk_client = None

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        """向 LLM 发送消息列表与可选工具定义，返回结构化的 LLMResponse。"""
        if not self.api_key:
            raise RuntimeError(
                "未检测到 OPENAI_API_KEY！请在环境变量或项目根目录 .env 文件中配置 OPENAI_API_KEY，"
                "或使用 `--demo` 参数运行内置模拟演示。"
            )

        if self._sdk_client is not None:
            return self._chat_via_sdk(messages, tools)
        return self._chat_via_urllib(messages, tools)

    def _chat_via_sdk(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        resp = self._sdk_client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        raw_tool_calls: list[dict[str, Any]] = []

        if choice.tool_calls:
            for tc in choice.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )
                raw_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        raw_msg: dict[str, Any] = {
            "role": "assistant",
            "content": choice.content,
        }
        if raw_tool_calls:
            raw_msg["tool_calls"] = raw_tool_calls

        total_tokens = resp.usage.total_tokens if getattr(resp, "usage", None) else 0
        return LLMResponse(
            content=choice.content,
            tool_calls=tool_calls,
            raw_message=raw_msg,
            total_tokens=total_tokens,
        )

    def _chat_via_urllib(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API HTTP {e.code}: {err_body}") from e

        msg = body["choices"][0]["message"]
        tool_calls: list[ToolCall] = []
        raw_tool_calls: list[dict[str, Any]] = []

        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            tool_calls.append(ToolCall(id=tc["id"], name=fn["name"], arguments=fn["arguments"]))
            raw_tool_calls.append(
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": fn["name"], "arguments": fn["arguments"]},
                }
            )

        raw_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.get("content"),
        }
        if raw_tool_calls:
            raw_msg["tool_calls"] = raw_tool_calls

        total_tokens = body.get("usage", {}).get("total_tokens", 0)
        return LLMResponse(
            content=msg.get("content"),
            tool_calls=tool_calls,
            raw_message=raw_msg,
            total_tokens=total_tokens,
        )
