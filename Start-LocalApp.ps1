param(
    [string]$PythonPath = "$env:USERPROFILE\edge-rag\venv\Scripts\python.exe",
    [switch]$IndexOnly
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python environment not found: $PythonPath"
}
$env:AZURE_AI_CONTAINER_MODE = 'true'
$env:AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = 'http://127.0.0.1:5010'
$env:AZURE_DOCUMENT_INTELLIGENCE_API_VERSION = '2022-08-31'
$env:QDRANT_URL = 'http://127.0.0.1:6333'
$env:OLLAMA_HOST = 'http://127.0.0.1:11435'
$env:PYTHONIOENCODING = 'utf-8'

if ($IndexOnly) {
    & $PythonPath -u '.\src\enhanced_indexer.py' --english-only
} else {
    & $PythonPath -m streamlit run '.\src\enhanced_app.py' --server.port 8503 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
}
if ($LASTEXITCODE -ne 0) {
    throw "Application exited with code $LASTEXITCODE"
}
