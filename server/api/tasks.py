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
    TASK_PLANNER_SYSTEM_PROMPT,
    TASK_STEP_EXECUTOR_SYSTEM_PROMPT,
    TASK_STEP_REVISE_SYSTEM_PROMPT,
)
from ..services.registry import SessionRegistry
from .sessions import _collect_prompt_logs, _provider_error_event, _resolve_spec, _sse

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
    "edit_step",
    "approve",
}

#: Какие действия разрешены на каждом этапе.
_ALLOWED_ACTIONS: dict[str, set[str]] = {
    "input": {"describe"},
    "plan_review": {"confirm", "revise", "edit_step"},
    "mode_select": {"run_all", "start_steps", "edit_step"},
    "step_review": {"confirm_step", "revise_step", "edit_step"},
    "review": {"approve", "revise", "edit_step"},
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


def _system_with_profile(
    system_prompt: str,
    profile_block: Optional[str],
    rules_frame: Optional[str] = None,
) -> str:
    """Системный промпт с добавленными блоками правил и профиля.

    Порядок: правила → профиль → служебный промпт этапа (жёсткие ограничения
    идут первыми — у них приоритет над профилем и прочими указаниями).
    """
    parts = []
    if rules_frame:
        parts.append(rules_frame)
    if profile_block:
        parts.append(profile_block)
    parts.append(system_prompt)
    return "\n\n".join(parts)


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


def _plan_from_steps(steps: list[str]) -> str:
    """Пересобирает текст плана из отредактированных пользователем шагов."""
    return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))


def _same_meaning(a: str, b: str) -> bool:
    """Совпадают ли строки по смыслу (без учёта пробелов/переносов)."""
    return " ".join(a.split()) == " ".join(b.split())


#: Глаголы перехода по шагам — маршрутизация замечаний в ``revise_step``.
_STEP_TRANSITION_RE = re.compile(
    r"верн|назад|перейд|переход|повтор|заново|начни|отмени", re.IGNORECASE
)
_STEP_REF_DIGIT_RE = re.compile(
    r"(?:шаг|этап)[а-яё]*\s*№?\s*(\d{1,2})|(\d{1,2})\s*[-–]?\s*[йя]\s*(?:шаг|этап)",
    re.IGNORECASE,
)
_STEP_WORD_NUMBERS = {
    "перв": 1, "втор": 2, "трет": 3, "четв": 4, "пят": 5,
    "шест": 6, "седьм": 7, "восьм": 8, "девят": 9, "десят": 10,
}


def _parse_step_directive(content: str, current: int, total: int) -> Optional[int]:
    """Определяет явную просьбу перейти к шагу в замечаниях пользователя.

    Учитывается только при наличии глагола перехода (вернись/перейди/повтори/
    назад/…), иначе замечание считается обычной доработкой текущего шага.

    Args:
        content: текст замечаний.
        current: номер текущего шага (1-based).
        total: всего шагов в плане.

    Returns:
        Номер целевого шага (1-based) или ``None``, если перехода нет.
    """
    text = content.lower()
    if not _STEP_TRANSITION_RE.search(text):
        return None
    match = _STEP_REF_DIGIT_RE.search(text)
    if match:
        return int(match.group(1) or match.group(2))
    if "предыдущ" in text or "назад" in text:
        return current - 1
    if "следующ" in text or "дальше" in text or "далее" in text:
        return current + 1
    if "последн" in text:
        return total
    if "шаг" in text or "этап" in text:
        for stem, number in _STEP_WORD_NUMBERS.items():
            if stem in text:
                return number
    return None


def _parse_step_reference(content: str, total: int) -> Optional[int]:
    """Извлекает явную ссылку на шаг плана в замечаниях (без глагола).

    Используется на этапе валидации (``review``): замечание с указанием шага
    (номер, слово-число, «последний») трактуется как точечная доработка этого
    шага. Относительные фразы («предыдущий»/«следующий»/«назад») не
    распознаются — они неоднозначны вне пошагового режима.

    Args:
        content: текст замечаний.
        total: всего шагов в плане.

    Returns:
        Номер целевого шага (1-based) или ``None``, если ссылки нет.
    """
    text = content.lower()
    match = _STEP_REF_DIGIT_RE.search(text)
    if match:
        return int(match.group(1) or match.group(2))
    if "последн" in text and ("шаг" in text or "этап" in text):
        return total
    if "шаг" in text or "этап" in text:
        for stem, number in _STEP_WORD_NUMBERS.items():
            if stem in text:
                return number
    return None


def _planner_messages(
    description: str,
    plan: Optional[str],
    result: Optional[str],
    feedback: str,
    profile_block: Optional[str],
    steps: Optional[list[str]] = None,
    step_results: Optional[list[str]] = None,
    rules_frame: Optional[str] = None,
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
            "content": _system_with_profile(
                TASK_PLANNER_SYSTEM_PROMPT, profile_block, rules_frame
            ),
        },
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _step_executor_messages(
    description: str,
    plan: str,
    steps: list[str],
    results: list[str],
    index: int,
    profile_block: Optional[str],
    rules_frame: Optional[str] = None,
    directive: Optional[str] = None,
) -> list[dict]:
    """Сообщения запроса исполнителя одного шага (пошаговый режим)."""
    parts = [f"Задача:\n{description}", f"Полный план:\n{plan}"]
    if results:
        parts.append(f"Выполненные шаги:\n{_completed_steps_block(steps, results)}")
    else:
        parts.append("Выполненные шаги: (пока нет)")
    parts.append(f"Выполни шаг {index + 1}: {steps[index]}")
    if directive:
        parts.append(f"Указание пользователя: {directive}")
    return [
        {
            "role": "system",
            "content": _system_with_profile(
                TASK_STEP_EXECUTOR_SYSTEM_PROMPT, profile_block, rules_frame
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
    rules_frame: Optional[str] = None,
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
                TASK_STEP_REVISE_SYSTEM_PROMPT, profile_block, rules_frame
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
    if action in ("describe", "revise", "revise_step", "edit_step") and not content:
        raise HTTPException(400, "content is required")

    # Прямая правка шага плана пользователем. Без отката — действие без LLM;
    # если правится уже выполненный шаг — прогресс откатывается и шаг
    # запускается заново (см. ветку edit_step ниже).
    run_index: Optional[int] = None
    rollback_backup: Optional[tuple[list[str], Optional[str]]] = None
    rework_index: Optional[int] = None
    if action == "edit_step":
        index = body.index
        if index is None or not isinstance(index, int) or not (0 <= index < len(task.task_steps)):
            raise HTTPException(400, "index is out of range")
        if _same_meaning(content, task.task_steps[index]):
            # Изменились только пробелы — смысл тот же, ничего не делаем.
            return _progress(task)
        task.task_steps[index] = content
        task.task_plan = _plan_from_steps(task.task_steps)
        if index < len(task.task_step_results):
            # Правка выполненного шага: результаты с этого шага недействительны.
            rollback_backup = (list(task.task_step_results), task.task_result)
            task.task_step_results = task.task_step_results[:index]
            task.task_result = None
            run_index = index
        else:
            registry.persist_task_state(task)
            return _progress(task)

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
    rules_frame = registry.rules_frame()

    if action == "describe":
        description = content
        if body.model:
            task.model = body.model
        if body.settings is not None:
            task.settings = sanitize_settings(body.settings)
        task.title = _task_title(content)
        task.add_user_message(content)
        messages = _planner_messages(
            content, None, None, "", profile_block, rules_frame=rules_frame
        )
    elif action == "revise":
        # На валидации замечание с явной ссылкой на шаг правит только этот шаг
        # (остальные результаты сохраняются, возврат на review).
        if stage == "review" and len(task.task_step_results) == len(task.task_steps):
            target = _parse_step_reference(content, len(task.task_steps))
            if target is not None:
                if not (1 <= target <= len(task.task_steps)):
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": f"Шага {target} нет в плане (всего {len(task.task_steps)})."
                        },
                    )
                rework_index = target - 1
        if rework_index is not None:
            task.add_user_message(content)
            messages = _step_revise_messages(
                description,
                task.task_plan or "",
                task.task_steps,
                task.task_step_results,
                rework_index,
                content,
                profile_block,
                rules_frame,
            )
        else:
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
                old_steps, old_results, rules_frame,
            )
    elif action == "run_all":
        # Выполняем весь план пошагово (по одному запросу на шаг), без пауз.
        if not task.task_steps:
            task.task_steps = _parse_steps(task.task_plan or "") or ["Выполнить задачу."]
        task.task_step_results = []
        messages = None
    elif action == "start_steps":
        if not task.task_steps:
            task.task_steps = _parse_steps(task.task_plan or "") or ["Выполнить задачу."]
        task.task_step_results = []
        task.add_user_message(_step_marker(0))
        messages = _step_executor_messages(
            description, task.task_plan or "", task.task_steps, [], 0,
            profile_block, rules_frame,
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
            rules_frame,
        )
    elif action == "edit_step":
        # Откат прогресса до правленного шага + его повторное выполнение.
        index = run_index if run_index is not None else 0
        task.add_user_message(_step_marker(index))
        messages = _step_executor_messages(
            description,
            task.task_plan or "",
            task.task_steps,
            task.task_step_results,
            index,
            profile_block,
            rules_frame,
        )
    else:  # revise_step — доработка текущего шага или явный переход по замечаниям
        if not task.task_step_results:
            return JSONResponse(
                status_code=409,
                content={"error": "Нет шага для доработки", "stage": stage},
            )
        current = len(task.task_step_results)  # 1-based: шаг на подтверждении
        total = len(task.task_steps)
        target = _parse_step_directive(content, current, total)
        if target is not None and target != current:
            if target < 1:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Шага {target} нет в плане (всего {total})."},
                )
            if target > total:
                if target == current + 1:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": "Это последний шаг — подтвердите его для завершения.",
                            "code": "no_next_step",
                        },
                    )
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Шага {target} нет в плане (всего {total})."},
                )
            if target > current + 1:
                # Пропуск нескольких шагов вперёд запрещён.
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            f"Переход вперёд через несколько шагов запрещён: со шага "
                            f"{current} доступен только шаг {current + 1} или возврат "
                            f"к предыдущим (1–{current - 1})."
                        ),
                        "code": "step_skip_forbidden",
                    },
                )
            # Явный переход: следующий шаг (target == current + 1) или возврат назад.
            task.add_user_message(content)
            if target < current:
                rollback_backup = (list(task.task_step_results), task.task_result)
                task.task_step_results = task.task_step_results[: target - 1]
                task.task_result = None
            run_index = target - 1
            messages = _step_executor_messages(
                description,
                task.task_plan or "",
                task.task_steps,
                task.task_step_results,
                run_index,
                profile_block,
                rules_frame,
                directive=content,
            )
        else:
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
                rules_frame,
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
            if action == "run_all":
                # Весь план пошагово: по одному запросу исполнителя на шаг,
                # без пауз; после каждого — stage=step_review, в конце review.
                total = len(task.task_steps)
                for index in range(total):
                    step_messages = _step_executor_messages(
                        description,
                        task.task_plan or "",
                        task.task_steps,
                        task.task_step_results,
                        index,
                        profile_block,
                        rules_frame,
                    )
                    yield _sse({"type": "step_run", "index": index, "total": total})
                    async for event in task.stream_completion(
                        runner, spec, gen_settings, messages=step_messages,
                        prompt_kind="run_all",
                    ):
                        if event.get("type") == "done":
                            task.task_step_results.append(event.get("content") or "")
                            task.task_stage = "step_review"
                            for frame in drain_request_logs():
                                yield frame
                        yield _sse(event)
                    _collect_prompt_logs(registry, task, "Задача", task.title or "")
                    # Без пользовательских маркеров: шаги как самостоятельные
                    # ответы ассистента — персистим одиночное сообщение.
                    registry.remember_assistant(task)
                    registry.persist_task_state(task)
                    registry.persist_new_requests(task)
                    for frame in drain_request_logs():
                        yield frame
                    yield _sse(_stage_event(task))
                task.task_result = _combine_step_result(
                    task.task_steps, task.task_step_results
                )
                task.task_stage = "review"
                registry.persist_task_state(task)
                yield _sse(_stage_event(task))
                return

            async for event in task.stream_completion(
                runner, spec, gen_settings, messages=messages, prompt_kind=action
            ):
                if event.get("type") == "done":
                    text = event.get("content") or ""
                    if action in ("describe", "revise"):
                        if action == "revise" and rework_index is not None:
                            # Точечная доработка шага на валидации: результат
                            # заменяется на месте, итог пересобирается.
                            task.task_step_results[rework_index] = text
                            task.task_result = _combine_step_result(
                                task.task_steps, task.task_step_results
                            )
                            task.task_stage = "review"
                        else:
                            task.task_stage = "plan_review"
                            task.task_plan = text
                            # Шаги разбираем сразу — доступны для правок и рельсы.
                            task.task_steps = _parse_steps(text)
                            task.task_step_results = []
                    else:  # start_steps / confirm_step / revise_step / edit_step
                        if action == "revise_step" and run_index is None:
                            # Доработка текущего шага — результат заменяется.
                            task.task_step_results[-1] = text
                        else:
                            # Выполнение/возврат/следующий шаг — результат добавляется.
                            task.task_step_results.append(text)
                        task.task_stage = "step_review"
                    for frame in drain_request_logs():
                        yield frame
                yield _sse(event)
            _collect_prompt_logs(registry, task, "Задача", task.title or "")
            registry.remember_pair(task)
            registry.persist_task_state(task)
            registry.persist_new_requests(task)
            for frame in drain_request_logs():
                yield frame
            yield _sse(_stage_event(task))
        except ProviderError as exc:
            task.rollback_user_message()
            if rollback_backup is not None:
                # Сбой перезапуска правленого шага — возвращаем откаченный прогресс.
                task.task_step_results, task.task_result = rollback_backup
            registry.persist_new_requests(task)
            yield _sse(_provider_error_event(exc, task))
        except Exception as exc:  # noqa: BLE001 - отдаём ошибку клиенту событием
            task.rollback_user_message()
            if rollback_backup is not None:
                task.task_step_results, task.task_result = rollback_backup
            registry.persist_new_requests(task)
            yield _sse({"type": "error", "error": str(exc)})
        finally:
            task.busy = False

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
