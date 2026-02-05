"""
Side effects for the voice agent.

This is the ONLY place where I/O happens.
All functions here interact with the outside world (WebSocket, player).
"""

import logging
from typing import Optional

from fastapi import WebSocket

from .types import Action, StartPlaybackAction, StopPlaybackAction
from .player import AudioPlayer

logger = logging.getLogger(__name__)


async def execute(
    action: Action,
    player: AudioPlayer,
) -> None:
    """
    Execute a side effect.
    
    This is the boundary between pure logic and the outside world.
    """
    if isinstance(action, StartPlaybackAction):
        await player.play(list(action.chunks))
    
    elif isinstance(action, StopPlaybackAction):
        await player.stop_and_clear()
