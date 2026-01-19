import re
import time
from typing import List, Optional, Tuple, Type

from maim_message import Seg

from src.chat.utils.chat_message_builder import get_raw_msg_before_timestamp_with_chat
from src.chat.utils.utils import is_bot_self
from src.person_info.person_info import Person, get_person_id_by_person_name
from src.plugin_system import (
    BasePlugin,
    ComponentInfo,
    ConfigField,
    register_plugin,
    get_logger,
)
from src.plugin_system.base.base_events_handler import BaseEventHandler
from src.plugin_system.base.base_tool import BaseTool
from src.plugin_system.base.component_types import CustomEventHandlerResult, EventType, MaiMessages, ToolParamType
from src.config.config import global_config


logger = get_logger("at_tool")


class LLMAtHandler(BaseEventHandler):
    event_type = EventType.POST_LLM
    handler_name = "llm_at_handler"
    handler_description = "为llm注入使用at的格式"
    weight = 100
    intercept_message = True

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], Optional[CustomEventHandlerResult], Optional[MaiMessages]]:
        if not message or not message.llm_prompt or not message.stream_id:
            return True, True, None, None, None
        reply_time_point = time.time()
        chat_id = message.stream_id
        message_list_before_short = get_raw_msg_before_timestamp_with_chat(
            chat_id=chat_id,
            timestamp=reply_time_point,
            limit=global_config.chat.max_context_size,
            filter_intercept_message_level=1,
        )
        user_id_map = []
        already_add = []
        logger.info(f"注入历史{len(message_list_before_short)}条消息的id")
        for i in message_list_before_short:
            user_id = i.user_info.user_id
            user_name = i.user_info.user_nickname
            if is_bot_self(i.user_info.platform, user_id):
                continue
            if user_id in already_add:
                continue
            user_id_map.append((user_name, user_id))
            already_add.append(user_id)

        at_prompt = """
你可以使用<at:user_id>的形式在消息中插入at
例如：
<at:12132344>你怎么还不起床
直接输出标签，不要使用 Markdown 链接或 @昵称
以下是昵称和user_id的映射表:\n"""
        for i in user_id_map:
            at_prompt += f"{i[0]}:{i[1]}\n"
        at_prompt += "\n你*可以*在提及某个人时使用at来强调\n"
        message.modify_llm_prompt(at_prompt + message.llm_prompt, suppress_warning=True)

        return True, True, None, None, message


class PostAtHandler(BaseEventHandler):
    event_type = EventType.POST_SEND_PRE_PROCESS
    handler_name = "post_at_handler"
    handler_description = "将llm输出的at格式转换为消息段"
    weight = 100
    intercept_message = True

    @staticmethod
    def get_at_and_replace_to_empty(message: str) -> Tuple[List[str], List[str]]:
        pattern = r"(<at:\d+>)"
        parts = re.split(pattern, message)
        text_parts = []
        user_ids = []
        for part in parts:
            if part.startswith("<at:") and part.endswith(">"):
                # 提取 user_id
                user_id = part[4:-1]  # 去掉 <at: 和 >
                user_ids.append(user_id)
            else:
                text_parts.append(part)
        return text_parts, user_ids

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], Optional[CustomEventHandlerResult], Optional[MaiMessages]]:
        if not message:
            return True, True, None, None, None
        text_num = 0
        for i in message.message_segments:
            if i.type == "text":
                text_num += 1
        if text_num == 0:
            logger.info("非文本消息，跳过处理")
            return True, True, None, None, None
        raw_text = ""
        for i in message.message_segments:
            if not i.type == "text":
                continue
            raw_text += i.data
        texts, ats = self.get_at_and_replace_to_empty(raw_text)
        if len(ats) == 0:
            logger.info("未发现at")
            return True, True, None, None, None
        index = 0
        message_seg = []
        if len(texts) - len(ats) != 1:
            logger.error("字段数量不符，跳过处理")
            return True, True, None, None, None
        for _ in ats:
            text = texts[index]
            if text:
                message_seg.append(Seg(type="text", data=text))
            message_seg.append(Seg(type="at", data=ats[index]))
            index += 1
            text = texts[index]
            if text:
                message_seg.append(Seg(type="text", data=text))
        for i in message.message_segments:
            if i.type == "text":
                continue
            message_seg.append(i)
        message.modify_message_segments(message_seg, True)
        return True, True, None, None, message


class GetMemberInfoByNameTool(BaseTool):
    name = "get_member_info_by_name"
    description = "获取群成员信息"
    parameters = [
        ("name", ToolParamType.STRING, "要查找的群成员名称", True, None),
    ]
    available_for_llm = True

    async def execute(self, function_args):
        name = function_args.get("name", "")
        person_id = get_person_id_by_person_name(name)
        if not person_id:
            logger.warn(f"未找到id为{name}的用户")
            return {"content": "未找到用户"}
        person = Person(person_id=person_id)
        person.load_from_database()
        if not person.user_id:
            return {"content": "未找到用户"}
        return {"content": f"用户{name}的id为{person.user_id}"}


@register_plugin
class AtPlugin(BasePlugin):
    plugin_name: str = "at_plugin"
    enable_plugin: bool = True
    dependencies: list[str] = []
    python_dependencies: list[str] = []
    config_file_name: str = "config.toml"

    # 配置节描述
    config_section_descriptions = {
        "plugin": "插件基本信息",
    }

    # 配置Schema定义
    config_schema: dict = {
        "plugin": {
            "name": ConfigField(type=str, default="at_plugin", description="插件名称"),
            "version": ConfigField(type=str, default="1.0.0", description="插件版本"),
            "config_version": ConfigField(type=str, default="1.0.0", description="配置文件版本"),
            "enabled": ConfigField(type=bool, default=True, description="是否启用插件"),
        }
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        """获取插件包含的组件列表"""
        return [
            (LLMAtHandler.get_handler_info(), LLMAtHandler),
            (PostAtHandler.get_handler_info(), PostAtHandler),
            (GetMemberInfoByNameTool.get_tool_info(), GetMemberInfoByNameTool),
        ]
