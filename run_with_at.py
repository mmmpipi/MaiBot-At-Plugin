"""MaiBot 启动脚本 — 集成 At 功能 dowhen 插桩

在启动 MaiBot 主程序前注册 At 功能插桩，
确保字节码注入在目标函数被调用前生效。

用法:
    python run_with_at.py
"""

import asyncio
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

from dowhen import do, when

# ─── 路径配置 ──────────────────────────────────────────────────────

mai_bot_root = Path(__file__).resolve().parent / "MaiBot"
if str(mai_bot_root) not in sys.path:
    sys.path.insert(0, str(mai_bot_root))

logger = logging.getLogger("at_injector")

# ─── 插桩配置 ──────────────────────────────────────────────────────

BOT_USER_ID: Optional[str] = None
MAX_CONTEXT_SIZE: int = 20


# ─── 辅助函数 ──────────────────────────────────────────────────


def _get_user_id_map(
    chat_id: str, reply_time_point: float, limit: int
) -> list[tuple[str, str]]:
    """从聊天历史中提取用户昵称→用户ID 映射表。"""
    try:
        from src.services.message_service import get_messages_before_time_in_chat
    except ImportError:
        logger.warning("无法导入消息服务，跳过 at 注入")
        return []

    message_list = get_messages_before_time_in_chat(
        chat_id=chat_id,
        timestamp=reply_time_point,
        limit=limit,
        filter_intercept_message_level=1,
    )

    user_id_map: list[tuple[str, str]] = []
    already_add: list[str] = []

    for msg in message_list:
        try:
            if hasattr(msg, "message_info") and msg.message_info:
                user_info = msg.message_info.user_info
                if user_info:
                    user_id = getattr(user_info, "user_id", None)
                    user_nickname = getattr(user_info, "user_nickname", None)
                else:
                    continue
            elif hasattr(msg, "bot_user_info") and msg.bot_user_info:
                user_id = getattr(msg.bot_user_info, "user_id", None)
                user_nickname = getattr(msg.bot_user_info, "user_nickname", None)
            else:
                continue
        except Exception:
            continue

        if not user_id:
            continue
        if BOT_USER_ID and str(user_id) == BOT_USER_ID:
            continue
        if user_id in already_add:
            continue

        user_id_map.append((user_nickname or str(user_id), str(user_id)))
        already_add.append(user_id)

    return user_id_map


def _build_at_prompt(user_id_map: list[tuple[str, str]]) -> str:
    """构造 at 格式提示词。"""
    at_prompt = """
你可以使用[@唯一id]的形式在消息中插入at
例如：
[@12132323]
直接输出标签，不要带有其他修饰
以下是昵称和唯一id的映射表:\n"""
    for nick, uid in user_id_map:
        at_prompt += f"{nick}:{uid}\n"
    at_prompt += "\n你*可以*在提及某个人时使用at来强调\n"
    return at_prompt


def _inject_at_prompt(prompt: str, stream_id: Optional[str]) -> str:
    """在 LLM 调用前注入 at 格式提示。"""
    if not prompt or not stream_id:
        return prompt

    reply_time_point = time.time()
    user_id_map = _get_user_id_map(
        chat_id=stream_id,
        reply_time_point=reply_time_point,
        limit=MAX_CONTEXT_SIZE,
    )

    if not user_id_map:
        logger.debug(f"未找到历史用户，跳过注入 (stream_id={stream_id})")
        return prompt

    at_prompt = _build_at_prompt(user_id_map)
    logger.info(f"注入历史 {len(user_id_map)} 条消息的id (stream_id={stream_id})")
    return at_prompt + prompt


def _inject_at_prompt_multimodal(messages, stream_id: Optional[str]) -> list:
    """在多模态消息列表中注入 at 格式提示到系统提示。"""
    if not messages or not stream_id:
        return messages

    reply_time_point = time.time()
    user_id_map = _get_user_id_map(
        chat_id=stream_id,
        reply_time_point=reply_time_point,
        limit=MAX_CONTEXT_SIZE,
    )

    if not user_id_map:
        logger.debug(f"多模态：未找到历史用户，跳过注入 (stream_id={stream_id})")
        return messages

    at_prompt = _build_at_prompt(user_id_map)
    logger.info(
        f"多模态：注入历史 {len(user_id_map)} 条消息的id (stream_id={stream_id})"
    )

    if messages and hasattr(messages[0], "parts"):
        for part in messages[0].parts:
            if hasattr(part, "text"):
                part.text = at_prompt + part.text
                break

    return messages


def _extract_at_and_replace(message: str) -> tuple[list[str], list[str]]:
    """从消息文本中提取 [@数字id] 模式并拆分。"""
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


# ─── 新链注入：send_service ─────────────────────────────────────


def _process_at_components_new_chain(message_obj) -> Any:
    """新链：将 SessionMessage.raw_message.components 中的 [@id] 文本转换为 AtComponent。"""
    if not message_obj or not hasattr(message_obj, "raw_message"):
        return message_obj

    raw_message = message_obj.raw_message
    if not raw_message or not hasattr(raw_message, "components"):
        return message_obj

    components = raw_message.components
    if not components:
        return message_obj

    try:
        from src.common.data_models.message_component_data_model import (
            AtComponent,
            ReplyComponent,
            TextComponent,
        )

        text_components = [c for c in components if isinstance(c, TextComponent)]
        if not text_components:
            return message_obj

        raw_text = "".join(c.text for c in text_components)
        texts, ats = _extract_at_and_replace(raw_text)
        if not ats:
            return message_obj

        if len(texts) - len(ats) != 1:
            logger.warning("新链：字段数量不符，跳过处理")
            return message_obj

        new_components = []
        index = 0
        for _ in ats:
            text = texts[index]
            if text:
                new_components.append(TextComponent(text=text))
            new_components.append(AtComponent(target_user_id=ats[index]))
            index += 1
            text = texts[index]
            if text:
                if text.startswith(" "):
                    new_components.append(TextComponent(text=text))
                else:
                    new_components.append(TextComponent(text=" " + text))

        for comp in components:
            if isinstance(comp, TextComponent):
                continue
            if isinstance(comp, ReplyComponent):
                new_components.insert(0, comp)
                continue
            new_components.append(comp)

        raw_message.components = new_components

        if hasattr(message_obj, "processed_plain_text"):
            message_obj.processed_plain_text = _build_processed_plain_text(
                new_components
            )

        logger.info(f"新链：转换了 {len(ats)} 个 at 标记")

    except Exception as e:
        logger.error(f"新链处理 at 组件失败: {e}", exc_info=True)

    return message_obj


def _build_processed_plain_text(components) -> str:
    """从组件列表构建 processed_plain_text。"""
    parts = []
    for comp in components:
        if hasattr(comp, "text"):
            parts.append(comp.text)
        elif hasattr(comp, "target_user_id"):
            target = (
                getattr(comp, "target_user_cardname", None)
                or getattr(comp, "target_user_nickname", None)
                or comp.target_user_id
            )
            parts.append(f"@{target}")
        elif hasattr(comp, "content") and comp.content:
            parts.append(comp.content)
        elif hasattr(comp, "format_name"):
            parts.append(f"[{comp.format_name}]")
    return " ".join(parts)


# ─── 旧链注入：uni_message_sender ───────────────────────────────


def _process_at_segments_old_chain(message_obj) -> Any:
    """旧链：将 SessionMessage.message_segment 中的 [@id] 文本转换为 Seg(type="at")。"""
    if not message_obj:
        return message_obj

    try:
        if hasattr(message_obj, "message_segment") and message_obj.message_segment:
            seg = message_obj.message_segment
            if seg.type == "seglist":
                segments = list(seg.data)
            elif seg.type == "text":
                segments = [seg]
            else:
                return message_obj
        else:
            return message_obj

        text_num = sum(1 for s in segments if s.type == "text")
        if text_num == 0:
            return message_obj

        raw_text = ""
        for s in segments:
            if s.type == "text":
                raw_text += s.data

        texts, ats = _extract_at_and_replace(raw_text)
        if not ats:
            return message_obj

        if len(texts) - len(ats) != 1:
            logger.warning("旧链：字段数量不符，跳过处理")
            return message_obj

        from maim_message import Seg

        message_seg = []
        index = 0
        for _ in ats:
            text = texts[index]
            if text:
                message_seg.append(Seg(type="text", data=text))
            message_seg.append(Seg(type="at", data=ats[index]))
            index += 1
            text = texts[index]
            if text:
                if text.startswith(" "):
                    message_seg.append(Seg(type="text", data=text))
                else:
                    message_seg.append(Seg(type="text", data=" " + text))

        for seg in segments:
            if seg.type == "text":
                continue
            if seg.type == "reply":
                message_seg.insert(0, seg)
                continue
            message_seg.append(seg)

        message_obj.message_segment = Seg(type="seglist", data=message_seg)
        logger.info(f"旧链：转换了 {len(ats)} 个 at 标记")

    except Exception as e:
        logger.error(f"旧链处理 at 消息段失败: {e}", exc_info=True)

    return message_obj


# ─── 插桩注册 ──────────────────────────────────────────────────


def _install_at_injector() -> None:
    """注册四个 dowhen handler。"""
    global BOT_USER_ID, MAX_CONTEXT_SIZE

    try:
        from src.config.config import global_config

        if hasattr(global_config, "bot") and hasattr(global_config.bot, "user_id"):
            bot_uid = global_config.bot.user_id
            if bot_uid:
                BOT_USER_ID = str(bot_uid)
                logger.info(f"从配置读取 bot_user_id: {BOT_USER_ID}")
        if hasattr(global_config, "chat") and hasattr(
            global_config.chat, "max_context_size"
        ):
            ctx_size = global_config.chat.max_context_size
            if ctx_size:
                MAX_CONTEXT_SIZE = int(ctx_size)
                logger.info(f"从配置读取 max_context_size: {MAX_CONTEXT_SIZE}")
    except Exception as e:
        logger.warning(f"无法从配置读取参数，使用默认值: {e}")

    # ─── 注入点 1: 非多模态 Maisaka ────────────────────────────
    try:
        from src.chat.replyer.maisaka_generator import (
            MaisakaReplyGenerator as NonMultiGenerator,
        )

        def _before_llm_non_multi(prompt, stream_id, _frame):
            new_prompt = _inject_at_prompt(prompt, stream_id)
            if new_prompt != prompt:
                return {"prompt": new_prompt}
            return None

        do(_before_llm_non_multi).when(
            NonMultiGenerator.generate_reply_with_context,
            "generation_result = await self.express_model.generate_response(prompt)",
        )
        logger.info("注入点 1 已注册: 非多模态 Maisaka LLM 调用前")
    except ImportError as e:
        logger.error(f"无法注册注入点 1（非多模态）: {e}")
    except Exception as e:
        logger.error(f"注入点 1 注册失败: {e}", exc_info=True)

    # ─── 注入点 2: 多模态 Maisaka ──────────────────────────────
    try:
        from src.chat.replyer.maisaka_generator_multi import (
            MaisakaReplyGenerator as MultiGenerator,
        )

        def _before_llm_multi(request_messages, stream_id, _frame):
            _inject_at_prompt_multimodal(request_messages, stream_id)
            return None

        do(_before_llm_multi).when(
            MultiGenerator.generate_reply_with_context,
            "generation_result = await self.express_model.generate_response_with_messages(",
        )
        logger.info("注入点 2 已注册: 多模态 Maisaka LLM 调用前")
    except ImportError as e:
        logger.error(f"无法注册注入点 2（多模态）: {e}")
    except Exception as e:
        logger.error(f"注入点 2 注册失败: {e}", exc_info=True)

    # ─── 注入点 3: 新链 send_service ───────────────────────────
    try:
        from src.services import send_service

        def _before_send_new_chain(message, _frame):
            _process_at_components_new_chain(message)
            return None

        do(_before_send_new_chain).when(
            send_service.send_session_message,
            "if not message.message_id:",
        )
        logger.info("注入点 3 已注册: 新链 send_session_message 入口")
    except ImportError as e:
        logger.error(f"无法注册注入点 3（新链）: {e}")
    except Exception as e:
        logger.error(f"注入点 3 注册失败: {e}", exc_info=True)

    # ─── 注入点 4: 旧链 uni_message_sender ─────────────────────
    try:
        from src.chat.message_receive.uni_message_sender import UniversalMessageSender

        def _before_send_old_chain(message, _frame):
            _process_at_segments_old_chain(message)
            return None

        do(_before_send_old_chain).when(
            UniversalMessageSender.send_message,
            "await message.process()",
        )
        logger.info("注入点 4 已注册: 旧链 message.process() 前")
    except ImportError as e:
        logger.error(f"无法注册注入点 4（旧链）: {e}")
    except Exception as e:
        logger.error(f"注入点 4 注册失败: {e}", exc_info=True)


def _uninstall_at_injector() -> None:
    """卸载所有 At 功能插桩。"""
    from dowhen import clear_all

    clear_all()
    logger.info("所有插桩已清除")


# ─── 入口 ──────────────────────────────────────────────────────


def main():
    _install_at_injector()

    bot_py = Path(__file__).resolve().parent / "bot.py"
    import runpy

    runpy.run_path(str(bot_py), run_name="__main__")


if __name__ == "__main__":
    main()
