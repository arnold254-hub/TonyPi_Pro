#!/usr/bin/python3
# coding=utf8
"""
Shared helper utilities for large-model agents.
"""
import os
import shlex
import sys
from typing import Optional

# Ensure project root and embodied_intelligence package are importable
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
EI_DIR = os.path.join(ROOT_DIR, 'embodied_intelligence')

for path in (ROOT_DIR, EI_DIR):
    if path not in sys.path:
        sys.path.append(path)


def _load_env_from_rc():
    rc_path = os.path.expanduser('~/TonyPi/.typerc')
    if not os.path.exists(rc_path):
        return
    try:
        with open(rc_path, 'r', encoding='utf-8') as fp:
            for raw_line in fp:
                line = raw_line.strip()
                if not line or line.startswith('#') or not line.startswith('export '):
                    continue
                export_body = line[len('export '):]
                if '=' not in export_body:
                    continue
                key, value = export_body.split('=', 1)
                key = key.strip()
                if not key:
                    continue
                value = value.strip()
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                # Handle bash escape sequences minimally
                try:
                    parsed = shlex.split(f'"{value}"')[0]
                except Exception:
                    parsed = value
                os.environ.setdefault(key, parsed)
    except Exception as exc:
        print(f"[agent_adapter] Failed to load .typerc: {exc}")


_load_env_from_rc()
os.environ.setdefault('ASR_LANGUAGE', 'Chinese')

from embodied_intelligence.agents_router import router_agent

_AGENTS = {}


def get_agent(mode: str) -> router_agent.RouterAgent:
    agent = _AGENTS.get(mode)
    if agent is None:
        agent = router_agent.RouterAgent(mode=mode)
        _AGENTS[mode] = agent
    return agent


def init_agent(mode: str) -> router_agent.RouterAgent:
    agent = get_agent(mode)
    agent.init()
    return agent


def start_agent(mode: str) -> router_agent.RouterAgent:
    agent = get_agent(mode)
    agent.start()
    return agent


def stop_agent(mode: str):
    agent = _AGENTS.get(mode)
    if agent:
        agent.stop()


def exit_agent(mode: str):
    agent = _AGENTS.get(mode)
    if agent:
        agent.exit()


def run_frame(frame):
    router_agent.update_shared_frame(frame)
    tracked = router_agent.get_tracked_frame()
    return tracked if tracked is not None else frame

