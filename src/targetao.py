import sys
from ao.core.entry import run_script

def entry_point(argv):
    try:
        filename = argv[1]
    except IndexError:
        print "You must supply a filename"
        return 1

    return run_script(filename)

def target(*args):
    return entry_point, None

def jitpolicy(driver):
    from rpython.jit.codewriter.policy import JitPolicy
    return JitPolicy()

if __name__ == "__main__":
    entry_point(sys.argv)
