"""
OpenAI LLM service.

Streaming API for real-time token generation.
"""

import os
import asyncio
import logging
from typing import Optional, Callable, Awaitable, List, Dict

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# System prompt for the voice agent
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
        """
        Args:
            on_token: Callback for each streamed token
            on_done: Callback when generation completes
        """
        self._on_token = on_token
        self._on_done = on_done
        
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        self._task: Optional[asyncio.Task] = None
        self._running = False
        
        # Conversation history
        self._history: List[Dict[str, str]] = []
    
    @property
    def is_active(self) -> bool:
        """Whether LLM is currently generating."""
        return self._running and self._task is not None
    
    @property
    def history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self._history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history = []
        logger.info("Conversation history cleared")
    
    async def start(self, user_message: str) -> None:
        """
        Start generating a response.
        
        Args:
            user_message: The user's transcribed message
        """
        if self._running:
            logger.warning("LLM already running, cancelling previous")
            await self.cancel()
        
        # Add user message to history
        self._history.append({"role": "user", "content": user_message})
        
        self._running = True
        self._task = asyncio.create_task(self._generate())
    
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
        
        logger.info("LLM generation cancelled")
    
    async def _generate(self) -> None:
        """Generate response and stream tokens."""
        assistant_response = ""
        
        try:
            # Build messages with system prompt and history
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ] + self._history
            
            logger.info(f"Starting LLM generation with {len(self._history)} history messages")
            
            stream = await self._client.chat.completions.create(
                model="gpt-4o-mini",
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
            
            # Add assistant response to history if completed
            # @TODO: I think this means we always append the full response to the history, even if it wasn't fully uttered! This might add extra stuff to history on the next message if we got interrupted.
            if self._running and assistant_response:
                self._history.append({"role": "assistant", "content": assistant_response})
                logger.info(f"LLM generation complete: {len(assistant_response)} chars")
                await self._on_done()
        
        except asyncio.CancelledError:
            # If we have partial response, still add it to history
            if assistant_response:
                self._history.append({"role": "assistant", "content": assistant_response + "..."})
            raise
        
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            await self._on_done()  # Signal done even on error
        
        finally:
            self._running = False
            self._task = None
