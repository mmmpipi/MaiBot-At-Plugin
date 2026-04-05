import re
from typing import Any, Optional, Tuple

from maibot_sdk import EventHandler, MaiBotPlugin, Tool
from maibot_sdk.messages import MaiMessages, MessageSegment
from maibot_sdk.types import EventType, ToolParameterInfo, ToolParamType


class AtPlugin(MaiBotPlugin):
    """At 插件 — 为 LLM 注入 @at 格式支持，并将输出转换为消息段。"""

    # Class attribute for bot user ID filtering
    bot_user_id: Optional[str] = None

    async def on_load(self) -> None:
        try:
            bot_user_id = await self.ctx.config.get("bot.user_id")
            if bot_user_id:
                self.bot_user_id = str(bot_user_id)
                self.ctx.logger.info("Bot user_id loaded: %s", self.bot_user_id)
            else:
                self.ctx.logger.warning("未配置 bot.user_id，将无法过滤 bot 自身消息")
        except Exception as e:
            self.ctx.logger.warning("无法读取 bot.user_id 配置: %s", e)

    async def on_unload(self) -> None:
        return None

    async def on_config_update(
        self, scope: str, config_data: dict[str, object], version: str
    ) -> None:

        del scope
        del config_data
        del version

    def _extract_user_info_from_message(
        self, msg: Any
    ) -> Tuple[Optional[str], Optional[str]]:
        """从消息对象中提取用户 ID 和昵称。

        兼容字典和对象两种消息格式。

        Args:
            msg: 消息对象，可能是 dict 或具有 user_info 属性的对象。

        Returns:
            (user_id, user_nickname) 元组，提取失败时对应字段为 None。
        """
        if isinstance(msg, dict):
            user_info = msg.get("message_info", {}).get("user_info", {})
            return user_info.get("user_id"), user_info.get("user_nickname")
        else:
            try:
                user_info = getattr(msg, "user_info", None)
                if user_info:
                    uid = getattr(user_info, "user_id", None)
                    nick = getattr(user_info, "user_nickname", None)
                    return uid, nick
            except Exception:
                pass
        return None, None

    @staticmethod
    def get_at_and_replace_to_empty(message: str) -> Tuple[list[str], list[str]]:
        """从消息文本中提取 [@数字id] 模式并拆分。

        Args:
            message: 待处理的纯文本消息。

        Returns:
            (text_parts, user_ids) 元组，text_parts 为纯文本片段列表，
            user_ids 为提取出的用户 ID 列表。
        """
        pattern = r"(\[@\d+\])"
        parts = re.split(pattern, message)
        text_parts = []
        user_ids = []
        for part in parts:
            if part.startswith("[@") and part.endswith("]"):
                user_ids.append(part[2:-1])
            else:
                text_parts.append(part)
        return text_parts, user_ids

    @EventHandler(
        "llm_at_inject",
        event_type=EventType.POST_LLM,
        intercept_message=True,
        weight=100,
    )
    async def llm_at_inject(self, message: MaiMessages = None, **kwargs):
        """在 LLM 响应后注入 @at 格式提示。

        读取聊天历史中的用户列表，构造昵称→用户ID 映射表，
        追加到 LLM prompt 中使其能以 [@数字id] 格式输出 at。
        """
        if not message or not message.llm_prompt or not message.stream_id:
            return None

        chat_id = message.stream_id
        max_context_size = await self.ctx.config.get("chat.max_context_size", 20)
        try:
            max_context_size = int(max_context_size)
        except Exception:
            max_context_size = 20

        # Get recent messages for user mapping
        recent_msgs = await self.ctx.message.get_recent(
            chat_id=chat_id, limit=max_context_size
        )
        user_id_map: list[Tuple[str, str]] = []
        already_add: list[str] = []
        self.ctx.logger.info(f"注入历史{len(recent_msgs)}条消息的id")
        for m in recent_msgs:
            user_id, user_nickname = self._extract_user_info_from_message(m)
            if not user_id:
                continue
            if self.bot_user_id and str(user_id) == self.bot_user_id:
                continue
            if user_id in already_add:
                continue
            user_id_map.append((user_nickname or str(user_id), user_id))
            already_add.append(user_id)

        at_prompt = """
你可以使用[@唯一id]的形式在消息中插入at
例如：
[@12132323]
直接输出标签，不要带有其他修饰
以下是昵称和唯一id的映射表:\n"""
        for nick, uid in user_id_map:
            at_prompt += f"{nick}:{uid}\n"
        at_prompt += "\n你*可以*在提及某个人时使用at来强调\n"
        message.modify_prompt(at_prompt + (message.llm_prompt or ""))

        return {"modified_message": message}

    @EventHandler(
        "post_at_replace",
        event_type=EventType.POST_SEND_PRE_PROCESS,
        intercept_message=True,
        weight=100,
    )
    async def post_at_replace(self, message: MaiMessages = None, **kwargs):
        """将 LLM 输出的 [@数字id] 文本转换为消息段。

        从消息段中提取纯文本，匹配 [@数字id] 模式，
        重建消息段列表（文本段 + at 段），保留 reply 段。
        """
        if not message:
            return None

        # Count text segments
        text_num = sum(1 for seg in message.message_segments if seg.type == "text")
        if text_num == 0:
            self.ctx.logger.info("非文本消息，跳过处理")
            return None

        # Build raw text from text segments
        raw_text = ""
        for seg in message.message_segments:
            if seg.type != "text":
                continue
            # seg.data is a dict
            raw_text += seg.data.get("text", "")

        texts, ats = self.get_at_and_replace_to_empty(raw_text)
        if len(ats) == 0:
            self.ctx.logger.info("未发现at")
            return None
        index = 0
        message_seg: list[MessageSegment] = []
        if len(texts) - len(ats) != 1:
            self.ctx.logger.error("字段数量不符，跳过处理")
            return None
        for _ in ats:
            text = texts[index]
            if text:
                message_seg.append(MessageSegment(type="text", data={"text": text}))
            message_seg.append(MessageSegment(type="at", data={"qq": ats[index]}))
            index += 1
            text = texts[index]
            if text:
                if text.startswith(" "):
                    message_seg.append(MessageSegment(type="text", data={"text": text}))
                else:
                    message_seg.append(
                        MessageSegment(type="text", data={"text": " " + text})
                    )

        for seg in message.message_segments:
            if seg.type == "text":
                continue
            if seg.type == "reply":
                message_seg.insert(0, seg)
                continue
            message_seg.append(seg)

        message.message_segments = message_seg
        return {"modified_message": message}

    @Tool(
        "get_member_info_by_name",
        brief_description="获取群成员信息",
        detailed_description=("参数说明：\n- name：string，必填。要查找的群成员名称。"),
        parameters=[
            ToolParameterInfo(
                name="name",
                param_type=ToolParamType.STRING,
                description="要查找的群成员名称",
                required=True,
            ),
        ],
    )
    async def get_member_info_by_name(self, name: str, **kwargs):
        """根据成员名称查询 QQ 号。

        通过 ctx.person 接口先获取 person_id，再读取 user_id 字段。

        Args:
            name: 群成员名称（Bot 取的名字，对应 PersonInfo.person_name）。

        Returns:
            包含 content 字段的字典，成功时返回 "用户{name}的id为{user_id}"，
            未找到时返回 "未找到用户"。
        """
        # Resolve member id by name using ctx.person
        person_id = await self.ctx.person.get_id_by_name(name)
        if not person_id:
            self.ctx.logger.warning("未找到用户: %s", name)
            return {"content": "未找到用户"}
        user_id = await self.ctx.person.get_value(person_id, "user_id")
        if not user_id:
            return {"content": "未找到用户"}
        return {"content": f"用户{name}的id为{user_id}"}


def create_plugin() -> "AtPlugin":
    return AtPlugin()
