@echo off
setlocal

rem Создаём виртуальное окружение, если его ещё нет.
if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    py -3.14 -m venv .venv
    if errorlevel 1 py -3 -m venv .venv
)

rem Ставим зависимости.
echo [setup] Installing dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 goto :error

rem Порт по умолчанию 3000; читаем PORT из .env, если задан.
set PORT=3000
if exist ".env" (
    for /f "tokens=1,2 delims==" %%a in (.env) do (
        if "%%a"=="PORT" set "PORT=%%b"
    )
)

rem Открываем браузер.
start http://localhost:%PORT%

rem Запускаем сервер (порт читается из .env внутри приложения).
".venv\Scripts\python.exe" -m server.main
goto :eof

:error
echo [error] Не удалось установить зависимости. Проверьте Python и доступ в интернет.
exit /b 1
