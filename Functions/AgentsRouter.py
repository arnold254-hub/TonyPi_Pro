#!/usr/bin/python3
# coding=utf8
"""
与 TonyPi 主程序兼容的 router_agent 包装层。
"""
from typing import Any
import os
import sys
import traceback

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

EI_DIR = os.path.join(BASE_DIR, 'embodied_intelligence')
if EI_DIR not in sys.path:
    sys.path.append(EI_DIR)

os.environ.setdefault('ASR_LANGUAGE', 'Chinese')

from embodied_intelligence.agents_router import router_agent

LOG_FILE = '/home/pi/TonyPi/tonypi_service.log'


def log(message):
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as fp:
            fp.write(f"[AgentsRouter] {message}\n")
    except Exception:
        pass


def init() -> Any:
    """
    完成 router_agent 的初始化。
    """
    log('init called')
    try:
        return router_agent.init()
    except Exception as exc:
        log(f'init failed: {exc}\n{traceback.format_exc()}')
        raise


def start() -> Any:
    """
    启动 router_agent 主循环线程。
    """
    log('start called')
    try:
        return router_agent.start()
    except Exception as exc:
        log(f'start failed: {exc}\n{traceback.format_exc()}')
        raise


def stop() -> Any:
    """
    停止 router_agent，释放资源。
    """
    log('stop called')
    try:
        result = router_agent.stop()
        router_agent.clear_shared_frame()
        return result
    except Exception as exc:
        log(f'stop failed: {exc}\n{traceback.format_exc()}')
        raise


def exit() -> Any:
    """
    兼容接口，执行 stop。
    """
    log('exit called')
    try:
        result = router_agent.exit()
        router_agent.clear_shared_frame()
        return result
    except Exception as exc:
        log(f'exit failed: {exc}\n{traceback.format_exc()}')
        raise


def run(frame):
    """
    TonyPi 主循环会调用各功能模块的 run 函数。
    router_agent 自行控制摄像头，此处直接回传原始画面。
    """
    annotated = None
    try:
        router_agent.update_shared_frame(frame)
        annotated = router_agent.get_tracked_frame()
    except Exception as exc:
        log(f'update_shared_frame failed: {exc}')
    return annotated if annotated is not None else frame
