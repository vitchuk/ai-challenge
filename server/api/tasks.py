"""Маршруты протокола «Задачи» (строгая последовательность этапов).

Задача — это сессия ``kind=task`` со строгой машиной состояний:
``input`` (задаётся задача) → ``plan_review`` (план LLM, ждём подтверждения)
→ ``mode_select`` (выбор режима выполнения) → ``review`` (результат, ждём
проверки) → ``done`` (одобрено). Подтвердив план, пользователь выбирает
выполнить всё сразу (``run_all``) или по шагам с подтверждением каждого шага
(``start_steps``/``confirm_step``/``revise_step``, этап ``step_review``). На
этапах плана, проверки шага и результата доступна доработка: с плана и
результата — новый план, с проверки шага — переделка только текущего шага
(прогресс сохраняется). Каждый переход и содержимое (план, шаги, результаты)
сохраняются в БД, поэтому задачу можно продолжить с текущего этапа после
перерыва.
"""

from __future__ import annotations

import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..providers.base import ProviderError
from ..providers.client import StreamedCompletion
from ..schemas import TaskAdvanceRequest
from ..services import ChatService, SessionKind, sanitize_settings
from ..services.chat_service import (
    TASK_EXECUTOR_SYSTEM_PROMPT,
    TASK_PLANNER_SYSTEM_PROMPT,
    TASK_STEP_EXECUTOR_SYSTEM_PROMPT,
    TASK_STEP_REVISE_SYSTEM_PROMPT,
)
from ..services.registry import SessionRegistry
from .sessions import _provider_error_event, _resolve_spec, _sse

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

#: Допустимые действия протокола.
_ACTIONS = {
    "describe",
    "confirm",
    "revise",
    "run_all",
    "start_steps",
    "confirm_step",
    "revise_step",
    "approve",
}

#: Какие действия разрешены на каждом этапе.
_ALLOWED_ACTIONS: dict[str, set[str]] = {
    "input": {"describe"},
    "plan_review": {"confirm", "revise"},
    "mode_select": {"run_all", "start_steps"},
    "step_review": {"confirm_step", "revise_step"},
    "review": {"approve", "revise"},
    "done": set(),
}

#: Префиксы пунктов плана: «- », «1.», «1)», «Шаг 2:» и т.п.
_BULLET_RE = re.compile(r"^[-*\u2022]\s*")
_NUM_RE = re.compile(r"^(?:[Шш]аг\s*)?\d+\s*[.):\-\u2013\u2014]?\s*", re.IGNORECASE)


def _task_title(content: str) -> str:
    """Заголовок вкладки задачи из текста задачи."""
    title = " ".join(content.split()).strip()
    return title[:60] if title else "Задача"


def _task_description(task: ChatService) -> str:
    """Текст задачи — первое пользовательское сообщение истории."""
    for record in task.history:
        if record.role == "user":
            return record.content
    return ""


def _system_with_profile(system_prompt: str, profile_block: Optional[str]) -> str:
    """Системный промпт с добавленным блоком профиля (если он есть)."""
    if profile_block:
        return f"{profile_block}\n\n{system_prompt}"
    return system_prompt


def _parse_steps(plan: str) -> list[str]:
    """Разбирает план на шаги (по строкам, снимая нумерацию/маркеры).

    Пустые строки пропускаются; если после очистки шагов не осталось —
    весь план считается одним шагом.

    Args:
        plan: текст плана от планировщика.

    Returns:
        Список текстов шагов.
    """
    plan = (plan or "").strip()
    if not plan:
        return []
    steps: list[str] = []
    for raw in plan.splitlines():
        line = _BULLET_RE.sub("", raw.strip()).strip()
        if not line:
            continue
        cleaned = _NUM_RE.sub("", line).strip()
        steps.append(cleaned or line)
    return steps or [plan]


def _completed_steps_block(steps: list[str], results: list[str]) -> str:
    """Блок «Шаг N: …» с результатами выполненных шагов."""
    return "\n\n".join(
        f"Шаг {i + 1}: {steps[i]}\n{results[i]}"
        for i in range(min(len(steps), len(results)))
    )


def _combine_step_result(steps: list[str], results: list[str]) -> str:
    """Итог пошагового режима — склейка шагов и их результатов."""
    return _completed_steps_block(steps, results)


def _planner_messages(
    description: str,
    plan: Optional[str],
    result: Optional[str],
    feedback: str,
    profile_block: Optional[str],
    steps: Optional[list[str]] = None,
    step_results: Optional[list[str]] = None,
) -> list[dict]:
    """Сообщения запроса планировщика (шаг 2 — новый/исправленный план)."""
    parts = [f"Задача:\n{description}"]
    if plan:
        parts.append(f"Прежний план:\n{plan}")
    if result:
        parts.append(f"Прежний результат:\n{result}")
    if steps and step_results:
        parts.append(
            f"Результаты выполненных шагов:\n{_completed_steps_block(steps, step_results)}"
        )
    parts.append(f"Замечания пользователя:\n{feedback}")
    return [
        {
            "role": "system",
            "content": _system_with_profile(TASK_PLANNER_SYSTEM_PROMPT, profile_block),
        },
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _executor_messages(
    description: str, plan: str, profile_block: Optional[str]
) -> list[dict]:
    """Сообщения запроса исполнителя (выполнение всего плана)."""
    return [
        {
            "role": "system",
            "content": _system_with_profile(TASK_EXECUTOR_SYSTEM_PROMPT, profile_block),
        },
        {"role": "user", "content": f"Задача:\n{description}\n\nПодтверждённый план:\n{plan}"},
    ]


def _step_executor_messages(
    description: str,
    plan: str,
    steps: list[str],
    results: list[str],
    index: int,
    profile_block: Optional[str],
) -> list[dict]:
    """Сообщения запроса исполнителя одного шага (пошаговый режим)."""
    parts = [f"Задача:\n{description}", f"Полный план:\n{plan}"]
    if results:
        parts.append(f"Выполненные шаги:\n{_completed_steps_block(steps, results)}")
    else:
        parts.append("Выполненные шаги: (пока нет)")
    parts.append(f"Выполни шаг {index + 1}: {steps[index]}")
    return [
        {
            "role": "system",
            "content": _system_with_profile(
                TASK_STEP_EXECUTOR_SYSTEM_PROMPT, profile_block
            ),
        },
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _step_revise_messages(
    description: str,
    plan: str,
    steps: list[str],
    results: list[str],
    index: int,
    feedback: str,
    profile_block: Optional[str],
) -> list[dict]:
    """Сообщения запроса переделки одного шага (доработка на этапе шага)."""
    parts = [f"Задача:\n{description}", f"Полный план:\n{plan}"]
    if index > 0:
        parts.append(
            f"Выполненные шаги:\n{_completed_steps_block(steps[:index], results[:index])}"
        )
    parts.append(
        f"Прежний результат шага {index + 1}:\n{steps[index]}\n{results[index]}"
    )
    parts.append(f"Замечания пользователя:\n{feedback}")
    parts.append(f"Переделай шаг {index + 1}: {steps[index]}")
    return [
        {
            "role": "system",
            "content": _system_with_profile(
                TASK_STEP_REVISE_SYSTEM_PROMPT, profile_block
            ),
        },
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _step_marker(index: int) -> str:
    """Пользовательский маркер запуска шага (в историю)."""
    return f"Выполни шаг {index + 1}."


def _progress(task: ChatService) -> dict:
    """Прогресс задачи для JSON-ответов (без LLM)."""
    return {
        "stage": task.task_stage,
        "steps": list(task.task_steps),
        "step_results": list(task.task_step_results),
        "result": task.task_result,
    }


def _stage_event(task: ChatService) -> dict:
    """SSE-событие смены этапа и текущего шагового прогресса."""
    payload = {"type": "stage", **_progress(task)}
    return payload


def _task_state(task: ChatService) -> dict:
    """Состояние задачи для API (восстановление клиента)."""
    return {
        "id": task.id,
        "title": task.title,
        "stage": task.task_stage or "input",
        "plan": task.task_plan,
        "result": task.task_result,
        "steps": list(task.task_steps),
        "step_results": list(task.task_step_results),
        "model": task.model,
        "settings": task.settings.to_dict(),
        "last_active": task.last_active,
        "history": [m.to_dict() for m in task.history],
        "requests": [r.to_dict() for r in task.requests],
    }


@router.get("")
async def list_tasks(request: Request) -> dict:
    """Возвращает все задачи с состоянием (для восстановления клиента).

    Args:
        request: HTTP-запрос (реестр из состояния приложения).

    Returns:
        ``{"data": [{id, title, stage, plan, result, steps, step_results,
        model, settings, history, requests, last_active}]}``.
    """
    registry: SessionRegistry = request.app.state.registry
    data = [
        _task_state(session)
        for session in registry.list_sessions()
        if session.kind == SessionKind.TASK
    ]
    return {"data": data}


@router.post("/{task_id}/advance")
async def advance_task(task_id: str, body: TaskAdvanceRequest, request: Request):
    """Продвигает задачу по протоколу (действие зависит от текущего этапа).

    Args:
        task_id: идентификатор задачи.
        body: действие и его данные (``content``; для ``describe`` — ещё
            ``model``/``settings``).
        request: HTTP-запрос.

    Returns:
        SSE-поток для LLM-действий (``session``/``reasoning_*``/``request_log``/
        ``done``/``stage``/``error``) либо JSON с прогрессом (``confirm``,
        ``approve`` и финальный ``confirm_step``).

    Raises:
        HTTPException: 404 — задачи нет; 400 — не задача/неизвестное действие/
            пустой текст; 409 — задача занята.
    """
    registry: SessionRegistry = request.app.state.registry
    task = registry.get(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    if task.kind != SessionKind.TASK:
        raise HTTPException(400, "Not a task session")
    if task.busy:
        raise HTTPException(409, "Task is busy")

    action = body.action
    stage = task.task_stage or "input"
    if action not in _ACTIONS:
        raise HTTPException(400, f"Unknown action: {action}")
    if action not in _ALLOWED_ACTIONS.get(stage, set()):
        return JSONResponse(
            status_code=409,
            content={
                "error": "Действие недоступно на текущем этапе задачи",
                "stage": stage,
            },
        )

    content = (body.content or "").strip()
    if action in ("describe", "revise", "revise_step") and not content:
        raise HTTPException(400, "content is required")

    # Действия без обращения к LLM (профиль не нужен).
    if action == "confirm":
        task.task_stage = "mode_select"
        registry.persist_task_state(task)
        return _progress(task)
    if action == "approve":
        task.task_stage = "done"
        registry.persist_task_state(task)
        return _progress(task)

    remaining = len(task.task_steps) - len(task.task_step_results)
    if action == "confirm_step" and remaining <= 0:
        # Последний шаг уже выполнен — финализируем без LLM.
        task.task_result = _combine_step_result(task.task_steps, task.task_step_results)
        task.task_stage = "review"
        registry.persist_task_state(task)
        return _progress(task)

    # Далее — LLM-действия: профиль обязателен, как и в обычных чатах.
    profile_block = registry.active_profile_block()
    if profile_block is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Необходимо создать и установить профиль.",
                "code": "profile_required",
            },
        )

    opencode_session_id = request.app.state.opencode_session_id
    spec = _resolve_spec(registry, task, body.model, opencode_session_id)
    description = _task_description(task)

    if action == "describe":
        description = content
        if body.model:
            task.model = body.model
        if body.settings is not None:
            task.settings = sanitize_settings(body.settings)
        task.title = _task_title(content)
        task.add_user_message(content)
        messages = _planner_messages(content, None, None, "", profile_block)
    elif action == "revise":
        task.add_user_message(content)
        old_steps = list(task.task_steps)
        old_results = list(task.task_step_results)
        old_result = task.task_result
        # Новый план делает прежнее выполнение недействительным.
        task.task_steps = []
        task.task_step_results = []
        task.task_result = None
        messages = _planner_messages(
            description, task.task_plan, old_result, content, profile_block,
            old_steps, old_results,
        )
    elif action == "run_all":
        task.task_steps = []
        task.task_step_results = []
        task.add_user_message("Выполни весь план.")
        messages = _executor_messages(description, task.task_plan or "", profile_block)
    elif action == "start_steps":
        task.task_steps = _parse_steps(task.task_plan or "") or ["Выполнить задачу."]
        task.task_step_results = []
        task.add_user_message(_step_marker(0))
        messages = _step_executor_messages(
            description, task.task_plan or "", task.task_steps, [], 0, profile_block
        )
    elif action == "confirm_step":
        index = len(task.task_step_results)
        task.add_user_message(_step_marker(index))
        messages = _step_executor_messages(
            description,
            task.task_plan or "",
            task.task_steps,
            task.task_step_results,
            index,
            profile_block,
        )
    else:  # revise_step — переделываем текущий шаг (прогресс сохраняется)
        if not task.task_step_results:
            return JSONResponse(
                status_code=409,
                content={"error": "Нет шага для доработки", "stage": stage},
            )
        index = len(task.task_step_results) - 1
        task.add_user_message(content)
        messages = _step_revise_messages(
            description,
            task.task_plan or "",
            task.task_steps,
            task.task_step_results,
            index,
            content,
            profile_block,
        )

    gen_settings = task.settings
    runner = StreamedCompletion(client=request.app.state.http_client)

    async def event_stream():
        task.busy = True
        emitted = len(task.requests)

        def drain_request_logs() -> list[str]:
            nonlocal emitted
            frames = []
            while emitted < len(task.requests):
                record = task.requests[emitted]
                frames.append(_sse({"type": "request_log", "record": record.to_dict()}))
                emitted += 1
            return frames

        yield _sse({"type": "session", "id": task_id})
        try:
            async for event in task.stream_completion(
                runner, spec, gen_settings, messages=messages
            ):
                if event.get("type") == "done":
                    text = event.get("content") or ""
                    if action in ("describe", "revise"):
                        task.task_stage = "plan_review"
                        task.task_plan = text
                    elif action == "run_all":
                        task.task_stage = "review"
                        task.task_result = text
                    else:  # start_steps / confirm_step / revise_step
                        if action == "revise_step":
                            task.task_step_results[-1] = text
                        else:
                            task.task_step_results.append(text)
                        task.task_stage = "step_review"
                    for frame in drain_request_logs():
                        yield frame
                yield _sse(event)
            registry.remember_pair(task)
            registry.persist_task_state(task)
            registry.persist_new_requests(task)
            for frame in drain_request_logs():
                yield frame
            yield _sse(_stage_event(task))
        except ProviderError as exc:
            task.rollback_user_message()
            registry.persist_new_requests(task)
            yield _sse(_provider_error_event(exc, task))
        except Exception as exc:  # noqa: BLE001 - отдаём ошибку клиенту событием
            task.rollback_user_message()
            registry.persist_new_requests(task)
            yield _sse({"type": "error", "error": str(exc)})
        finally:
            task.busy = False

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
