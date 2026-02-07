# BotIA (Ollama + Python) — OFFLINE + WEB

Dos bots en Python para Windows:
- **bot_offline.py**: IA local (sin Internet)
- **bot_web.py**: IA local + comandos para web (/web y /read)
Incluye lectura de archivos: TXT/LOG, PDF, DOCX.

## Requisitos
- Windows 10/11
- Ollama instalado
- Python (recomendado `py` en Windows)
- RAM recomendada: 16GB+

## 1) Instalar/descargar modelo en Ollama
```powershell
ollama pull llama3.2
ollama list
2) Instalar dependencias Python
py -m pip install -U -r requirements.txt
3) Ejecutar
OFFLINE
py bot_offline.py

WEB
py bot_web.py

Comandos principales (dentro del bot)

/mode helpdesk | /mode docente

/lang auto|es|it

/file <ruta> /pdf <ruta> /docx <ruta>

/filesum

/askfile <pregunta>

(solo WEB) /web <consulta> y /read <url>

