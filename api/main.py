"""FastAPI application factory for the showcase demo."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_faithfulness_verifier, get_llm, get_retriever, get_settings
from api.routes.chat import router as chat_router
from api.routes.health import router as health_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_retriever()
    get_llm()
    get_faithfulness_verifier()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="AI Research Papers Showcase", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.FRONTEND_ORIGIN],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(chat_router)
    app.include_router(health_router)
    return app


app = create_app()
