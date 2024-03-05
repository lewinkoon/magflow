import sys
from core.gui import gui
from core.export import export

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        export()
    else:
        gui()
