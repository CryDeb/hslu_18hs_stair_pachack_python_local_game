if (Test-Path .\build) { Remove-Item .\build -Recurse; }
if (Test-Path .\dist) { Remove-Item .\dist -Recurse; }
if (Test-Path .\exe) {Remove-Item .\exe -Recurse;}
.\venv\Scripts\pyinstaller.exe --paths "C:\Windows\WinSxS\x86_microsoft-windows-m..namespace-downlevel_31bf3856ad364e35_10.0.17134.1_none_50c6cb8431e7428f" capture.py --onefile
if (Test-Path .\build) { Remove-Item .\build -Recurse; }
if (Test-Path .\dist) { 
    Copy-Item .\settings.ini .\dist;
    Rename-Item .\dist .\exe;
 }