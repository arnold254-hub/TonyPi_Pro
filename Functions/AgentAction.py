#!/usr/bin/python3
# coding=utf8
"""
Action agent entry points for Running.FUNCTIONS.
"""
from . import agent_adapter

MODE = 'action'


def init():
    agent_adapter.init_agent(MODE)


def start():
    agent_adapter.start_agent(MODE)


def stop():
    agent_adapter.stop_agent(MODE)


def exit():
    agent_adapter.exit_agent(MODE)


def run(frame):
    return agent_adapter.run_frame(frame)
