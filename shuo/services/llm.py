"""
LLM service with streaming (Groq, OpenAI-compatible).
"""

import os
import asyncio
from typing import Optional, Callable, Awaitable, List, Dict

from openai import AsyncOpenAI

from ..log import ServiceLogger

log = ServiceLogger("LLM")

SYSTEM_PROMPT = """You are a helpful voice assistant. Keep your responses concise and conversational, as they will be spoken aloud. Avoid using markdown, bullet points, or other formatting that doesn't work well in speech. Be friendly and natural."""


class LLMService:
    """
    OpenAI streaming LLM service.
    
    Manages conversation history and streams tokens via callback.
    """
    
    def __init__(
        self,
        on_token: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
    ):
        self._on_token = on_token
        self._on_done = on_done
        
        self._client = AsyncOpenAI(
            api_key=os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
        )
        self._task: Optional[asyncio.Task] = None
        self._running = False
        
        self._history: List[Dict[str, str]] = []
    
    @property
    def is_active(self) -> bool:
        return self._running and self._task is not None
    
    @property
    def history(self) -> List[Dict[str, str]]:
        return self._history.copy()
    
    def clear_history(self) -> None:
        self._history = []
    
    async def start(self, user_message: str) -> None:
        """Start generating a response."""
        if self._running:
            await self.cancel()
        
        self._history.append({"role": "user", "content": user_message})
        
        self._running = True
        self._task = asyncio.create_task(self._generate())
        log.connected()
    
    async def cancel(self) -> None:
        """Cancel ongoing generation."""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        log.cancelled()
    
    async def _generate(self) -> None:
        """Generate response and stream tokens."""
        assistant_response = ""
        
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ] + self._history
            
            stream = await self._client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                messages=messages,
                stream=True,
                max_tokens=500,
                temperature=0.7,
            )
            
            async for chunk in stream:
                if not self._running:
                    break
                
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    token = delta.content
                    assistant_response += token
                    await self._on_token(token)
            
            if self._running and assistant_response:
                self._history.append({"role": "assistant", "content": assistant_response})
                await self._on_done()
        
        except asyncio.CancelledError:
            if assistant_response:
                self._history.append({"role": "assistant", "content": assistant_response + "..."})
            raise
        
        except Exception as e:
            log.error("Generation failed", e)
            await self._on_done()
        
        finally:
            self._running = False
            self._task = None
