quick-compile:
	rpython targetao.py

opt-compile:
	rpython --opt=jit targetao.py

debug-example:
	PYPYLOG=jit-log-opt:logfile ./targetao-c examples/test.ao

run-py-tests:
	for i in tests/*/*.ao; do echo $$i; python targetao.py $$i && echo "$$i passed."; done

run-ao-tests:
	for i in tests/*/*.ao; do echo $$i; ./ao $$i && echo "$$i passed."; done
