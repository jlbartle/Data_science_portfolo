@ECHO OFF

SET SQLCMD = "C:\Program Files (x86)\Microsoft SQL Server\Client SDK\ODBC\170\Tools\Binn\SQLCMD.exe"

SET PATH="E:\OneDrive\DataScience\IST 659\Project\SQL"

SET SERVER="KS2O\SQLEXPRESS"

SET DB="LT"

SET LOGIN="UPDATER"

SET PASSWORD="jb"

SET OUTPUT="E:\OneDrive\DataScience\IST 659\Project\update.log"

CD /D %PATH%

ECHO %date% %time% >> %OUTPUT%
ECHO cd >> %OUTPUT%

%SQLCMD% -S %SERVER% -d %DB% -U %LOGIN% -P %PASSWORD% -i "DDLS\Lt_Table_Creates.sql" >> %OUTPUT%

for %%f in (*.sql) do (

%SQLCMD% -S %SERVER% -d %DB% -U %LOGIN% -P %PASSWORD% -i %%~f >> %OUTPUT%

)

%SQLCMD% -S %SERVER% -d %DB% -U %LOGIN% -P %PASSWORD% -i "DDLS\POST_Procedure_runs.sql" >> %OUTPUT%

cmd /k
