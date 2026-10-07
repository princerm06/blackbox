from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from blackbox.sessions import SessionStore, target_to_dict


class RegisterTargetRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    prompt: str | None = Field(default=None, max_length=120)


class SetTargetEnabledRequest(BaseModel):
    enabled: bool


def create_app(store: SessionStore | None = None) -> FastAPI:
    sessions = store or SessionStore()
    app = FastAPI(title="BlackBox", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/sessions")
    def create_session() -> dict[str, str]:
        session = sessions.create()
        return {"session_id": session.session_id}

    @app.get("/sessions/{session_id}/targets")
    def list_targets(session_id: str) -> dict[str, object]:
        try:
            session = sessions.get(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        return {
            "targets": [
                target_to_dict(target)
                for target in session.targets.targets()
            ]
        }

    @app.post("/sessions/{session_id}/targets", status_code=201)
    def register_target(
        session_id: str,
        request: RegisterTargetRequest,
    ) -> dict[str, object]:
        try:
            session = sessions.get(session_id)
            target = session.targets.register(
                display_name=request.display_name,
                prompt=request.prompt,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

        return target_to_dict(target)

    @app.patch("/sessions/{session_id}/targets/{target_id}")
    def set_target_enabled(
        session_id: str,
        target_id: str,
        request: SetTargetEnabledRequest,
    ) -> dict[str, object]:
        try:
            session = sessions.get(session_id)
            target = session.targets.set_enabled(
                target_id,
                request.enabled,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        return target_to_dict(target)

    @app.delete(
        "/sessions/{session_id}/targets/{target_id}",
        status_code=204,
    )
    def remove_target(session_id: str, target_id: str) -> None:
        try:
            session = sessions.get(session_id)
            session.targets.unregister(target_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return app


app = create_app()
