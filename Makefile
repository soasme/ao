quick-compile:
	rpython targetao.py

opt-compile:
	rpython --opt=jit targetao.py

debug-example:
	PYPYLOG=jit-log-opt:logfile ./targetao-c examples/test.ao
