from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from blackbox.app_processing import (
    RunStore,
    SessionVideoProcessor,
    processing_summary_to_dict,
)
from blackbox.reasoning.query import BeliefQueryEngine
from blackbox.reasoning.reobserve import ReobservationPlanner
from blackbox.sessions import SessionStore, target_to_dict


class RegisterTargetRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    prompt: str | None = Field(default=None, max_length=120)


class SetTargetEnabledRequest(BaseModel):
    enabled: bool


def _answer_to_dict(answer) -> dict[str, object]:
    return {
        "subject_object_id": answer.subject_object_id,
        "answer": answer.answer,
        "confidence": answer.confidence,
        "provenance": answer.provenance,
        "primary_hypothesis": answer.primary_hypothesis,
        "alternatives": [
            {"hypothesis": value, "weight": weight}
            for value, weight in answer.alternatives
        ],
        "evidence": list(answer.evidence),
        "caveats": list(answer.caveats),
    }


def _reobservation_to_dict(plan) -> dict[str, object]:
    return {
        "subject_object_id": plan.subject_object_id,
        "should_reobserve": plan.should_reobserve,
        "uncertainty": plan.uncertainty,
        "rationale": plan.rationale,
        "targets": [
            {
                "target": target.target,
                "score": target.score,
                "rationale": target.rationale,
            }
            for target in plan.targets
        ],
    }



def _belief_value(beliefs, predicate: str):
    return next(
        (
            belief
            for belief in beliefs
            if belief.predicate == predicate
        ),
        None,
    )


def _object_summary(world_state, object_id: str) -> dict[str, object]:
    beliefs = world_state.beliefs_for(object_id)
    visibility = _belief_value(beliefs, "CURRENT_VISIBILITY")
    last_bbox = _belief_value(beliefs, "LAST_OBSERVED_BBOX")
    uncertainty = world_state.uncertainty_for(object_id)

    return {
        "object_id": object_id,
        "class_name": world_state.object_class(object_id),
        "visibility": visibility.value if visibility is not None else "unknown",
        "visibility_confidence": (
            visibility.confidence if visibility is not None else 0.0
        ),
        "last_observed_bbox": (
            last_bbox.value if last_bbox is not None else None
        ),
        "last_observed_frame": (
            last_bbox.updated_frame_index
            if last_bbox is not None
            else None
        ),
        "last_observed_timestamp_ms": (
            last_bbox.updated_timestamp_ms
            if last_bbox is not None
            else None
        ),
        "has_uncertainty": bool(uncertainty),
    }


def _event_to_dict(object_id, event) -> dict[str, object]:
    return {
        "object_id": object_id,
        "event_type": event.event_type.value,
        "frame_index": event.frame_index,
        "timestamp_ms": event.timestamp_ms,
        "class_name": event.class_name,
        "confidence": event.confidence,
        "provenance": event.provenance.value,
        "metadata": event.metadata,
        "related_track_ids": list(event.related_track_ids),
    }



def _run_snapshot(
    session_id: str,
    run_id: str,
    run,
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "run_id": run_id,
        "summary": processing_summary_to_dict(run.summary),
        "objects": [
            _object_summary(run.world_state, object_id)
            for object_id in run.world_state.all_object_ids()
        ],
        "events": [
            _event_to_dict(object_id, event)
            for object_id, event in run.world_state.all_events()
        ],
        "media": {
            "source": (
                f"/sessions/{session_id}/runs/{run_id}/media/source"
            ),
            "annotated": (
                f"/sessions/{session_id}/runs/{run_id}/media/annotated"
                if run.summary.annotated_video_path is not None
                else None
            ),
        },
    }


def create_app(
    store: SessionStore | None = None,
    processor: SessionVideoProcessor | None = None,
    run_store: RunStore | None = None,
    workspace_root: Path = Path("runs") / "app",
) -> FastAPI:
    sessions = store or SessionStore()
    video_processor = processor or SessionVideoProcessor()
    runs = run_store or RunStore()
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

    @app.post("/sessions/{session_id}/videos", status_code=201)
    async def process_video(
        session_id: str,
        video: UploadFile = File(...),
    ) -> dict[str, object]:
        try:
            session = sessions.get(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        if not session.targets.enabled_targets():
            raise HTTPException(
                status_code=409,
                detail="register at least one enabled tracking target first",
            )

        suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
        run_id = str(uuid4())
        run_dir = workspace_root / session_id / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        input_path = run_dir / f"input{suffix}"

        content = await video.read()
        if not content:
            raise HTTPException(status_code=400, detail="uploaded video is empty")
        input_path.write_bytes(content)

        try:
            processed = video_processor.process(
                registry=session.targets,
                input_path=input_path,
                output_dir=run_dir,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        runs.put(session_id, run_id, processed)

        return {
            "run_id": run_id,
            "session_id": session_id,
            "targets": [
                target_to_dict(target)
                for target in session.targets.enabled_targets()
            ],
            "summary": processing_summary_to_dict(processed.summary),
        }

    @app.get("/sessions/{session_id}/runs/{run_id}")
    def get_run(
        session_id: str,
        run_id: str,
    ) -> dict[str, object]:
        try:
            sessions.get(session_id)
            run = runs.get(session_id, run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="run not found") from error

        return _run_snapshot(session_id, run_id, run)

    @app.get("/sessions/{session_id}/runs/{run_id}/media/source")
    def get_source_video(
        session_id: str,
        run_id: str,
    ) -> FileResponse:
        try:
            sessions.get(session_id)
            run = runs.get(session_id, run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="run not found") from error

        if not run.source_video_path.exists():
            raise HTTPException(
                status_code=404,
                detail="source video is no longer available",
            )

        return FileResponse(
            run.source_video_path,
            media_type="video/mp4",
            filename=run.source_video_path.name,
        )

    @app.get("/sessions/{session_id}/runs/{run_id}/media/annotated")
    def get_annotated_video(
        session_id: str,
        run_id: str,
    ) -> FileResponse:
        try:
            sessions.get(session_id)
            run = runs.get(session_id, run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="run not found") from error

        path = run.summary.annotated_video_path
        if path is None or not path.exists():
            raise HTTPException(
                status_code=404,
                detail="annotated video is not available",
            )

        return FileResponse(
            path,
            media_type="video/mp4",
            filename=path.name,
        )

    @app.get("/sessions/{session_id}/runs/{run_id}/objects")
    def list_run_objects(
        session_id: str,
        run_id: str,
    ) -> dict[str, object]:
        try:
            sessions.get(session_id)
            run = runs.get(session_id, run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="run not found") from error

        return {
            "objects": [
                _object_summary(run.world_state, object_id)
                for object_id in run.world_state.all_object_ids()
            ]
        }

    @app.get("/sessions/{session_id}/runs/{run_id}/events")
    def list_run_events(
        session_id: str,
        run_id: str,
    ) -> dict[str, object]:
        try:
            sessions.get(session_id)
            run = runs.get(session_id, run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="run not found") from error

        return {
            "events": [
                _event_to_dict(object_id, event)
                for object_id, event in run.world_state.all_events()
            ]
        }

    @app.get(
        "/sessions/{session_id}/runs/{run_id}/targets/{target_id}/where"
    )
    def where_is_target(
        session_id: str,
        run_id: str,
        target_id: str,
    ) -> dict[str, object]:
        try:
            session = sessions.get(session_id)
            run = runs.get(session_id, run_id)
            target = next(
                target
                for target in session.targets.targets()
                if target.target_id == target_id
            )
        except (KeyError, StopIteration) as error:
            raise HTTPException(status_code=404, detail="target or run not found") from error

        object_ids = run.world_state.object_ids_for_class(target.prompt)
        if not object_ids:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no detected object matched target prompt "
                    f"{target.prompt!r} in this run"
                ),
            )

        object_id = object_ids[0]
        answer = BeliefQueryEngine(run.world_state).where_is(object_id)
        reobservation = ReobservationPlanner(
            run.world_state
        ).plan_for(object_id)

        return {
            "target": target_to_dict(target),
            "object_id": object_id,
            "answer": _answer_to_dict(answer),
            "reobservation": _reobservation_to_dict(reobservation),
            "matching_object_count": len(object_ids),
            "ambiguity": (
                "multiple physical objects matched this visual concept"
                if len(object_ids) > 1
                else None
            ),
        }

    return app


app = create_app()
