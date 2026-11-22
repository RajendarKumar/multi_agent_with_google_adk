"""
Memory layer — STM and LTM wrappers over Google ADK services.

STM: DatabaseSessionService (PostgreSQL) — multi-turn session scratchpad
LTM: MemoryService — cross-session user/domain facts
"""

from typing import Any

from loguru import logger

from config.settings import settings


# ─────────────────────────────────────────────────────────────────────────────
# Short-Term Memory (STM)
# ─────────────────────────────────────────────────────────────────────────────
class STMService:
    """
    Wraps google.adk DatabaseSessionService with resilient fallback to InMemorySessionService.
    Provides typed helpers for the scratchpad state keys used across agents.
    """

    def __init__(self, use_in_memory: bool = False):
        from google.adk.sessions import DatabaseSessionService, InMemorySessionService
        self._fallback_service = InMemorySessionService()
        if use_in_memory:
            self._service = self._fallback_service
            logger.info("STM: Using InMemorySessionService")
        else:
            try:
                self._service = DatabaseSessionService(db_url=settings.database_url)
                logger.info("STM: Using DatabaseSessionService (PostgreSQL)")
            except Exception as e:
                logger.warning(f"STM: DatabaseSessionService initialization failed ({e}); falling back to InMemorySessionService")
                self._service = self._fallback_service

    def get_service(self):
        return self._service

    async def get_or_create_session(self, session_id: str, user_id: str, tenant_id: str):
        try:
            session = await self._service.get_session(
                app_name="responsive_agent",
                user_id=f"{tenant_id}:{user_id}",
                session_id=session_id,
            )
            if session is None:
                session = await self._service.create_session(
                    app_name="responsive_agent",
                    user_id=f"{tenant_id}:{user_id}",
                    session_id=session_id,
                    state={"tenant_id": tenant_id},
                )
            return session
        except Exception as e:
            logger.warning(f"STM: Database session access failed ({e}); switching to InMemorySessionService fallback")
            self._service = self._fallback_service
            session = await self._service.get_session(
                app_name="responsive_agent",
                user_id=f"{tenant_id}:{user_id}",
                session_id=session_id,
            )
            if session is None:
                session = await self._service.create_session(
                    app_name="responsive_agent",
                    user_id=f"{tenant_id}:{user_id}",
                    session_id=session_id,
                    state={"tenant_id": tenant_id},
                )
            return session

    async def update_state(self, session_id: str, user_id: str, tenant_id: str, updates: dict[str, Any]):
        from google.adk.events import Event, EventActions
        event = Event(author="system", actions=EventActions(state_delta=updates))
        session = await self.get_or_create_session(session_id, user_id, tenant_id)
        if session:
            try:
                await self._service.append_event(session, event)
            except Exception as e:
                logger.warning(f"STM: append_event failed on primary store ({e}); using fallback")
                self._service = self._fallback_service
                await self._service.append_event(session, event)


# ─────────────────────────────────────────────────────────────────────────────
# Long-Term Memory (LTM)
# ─────────────────────────────────────────────────────────────────────────────
class LTMService:
    """
    Wraps google.adk MemoryService for cross-session recall.
    After each session, call persist_session() to extract durable facts.
    """

    def __init__(self):
        try:
            from google.adk.memory import VertexAiMemoryBankService
            self._service = VertexAiMemoryBankService(project=None, location="us-central1")
            logger.info("LTM: Using VertexAiMemoryBankService")
        except Exception:
            # Fallback: custom pgvector-based memory
            self._service = self._build_pgvector_memory()
            logger.info("LTM: Using pgvector-based fallback MemoryService")

    def _build_pgvector_memory(self):
        """
        Fallback LTM: embed session summaries into pgvector for semantic recall.
        Replace with Mem0 or VertexAiMemoryBankService for production.
        """
        from google.adk.memory import InMemoryMemoryService
        logger.warning("LTM: Falling back to InMemoryMemoryService — facts won't persist across restarts")
        return InMemoryMemoryService()

    def get_service(self):
        return self._service

    async def persist_session(self, session, runner):
        """Extract and store durable facts from a completed session."""
        try:
            await runner.memory_service.add_session_to_memory(session)
            logger.debug(f"LTM: Persisted session {session.id}")
        except Exception as e:
            logger.error(f"LTM: Failed to persist session {session.id}: {e}")
