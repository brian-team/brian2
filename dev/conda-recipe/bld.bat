xcopy /I /Y /e "%RECIPE_DIR%\..\..\brian2" "%SRC_DIR%\brian2\"
xcopy "%RECIPE_DIR%\..\..\setup.py" "%SRC_DIR%"
"%PYTHON%" "%SRC_DIR%"\setup.py install --with-cython --fail-on-error --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1