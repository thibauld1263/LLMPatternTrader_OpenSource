import os
import sys
import time
import multiprocessing

multiprocessing.freeze_support()

# PyInstaller path resolution
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)



_console_hidden = False

def hide_console():
    """Hide the console window using Win32 API (Windows only). Called once."""
    global _console_hidden
    if _console_hidden:
        return
    _console_hidden = True
    try:
        import ctypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        hWnd = kernel32.GetConsoleWindow()
        if hWnd:
            user32.ShowWindow(hWnd, 0)  # SW_HIDE = 0
    except Exception:
        pass

print("╔══════════════════════════════════════════╗")
print("║     LLM Pattern Trader — Loading...      ║")
print("╚══════════════════════════════════════════╝")
t0 = time.time()

print(f"  ▸ Loading PyQt5...", end=" ", flush=True)
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, pyqtSignal
print(f"OK ({time.time()-t0:.1f}s)")

print(f"  ▸ Loading pandas...", end=" ", flush=True)
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
print(f"OK ({time.time()-t0:.1f}s)")

print(f"  ▸ Loading async engine...", end=" ", flush=True)
import qasync
import asyncio
import logging
print(f"OK ({time.time()-t0:.1f}s)")

print(f"  ▸ Loading GUI...", end=" ", flush=True)
from gui.style import STYLESHEET
from gui.main_window import MainWindow
print(f"OK ({time.time()-t0:.1f}s)")

print(f"  ▸ Loading services...", end=" ", flush=True)
from services.config_manager import ConfigManager
from services.logger_config import setup_logging
print(f"OK ({time.time()-t0:.1f}s)")

print(f"  ▸ Loading trading engine...", end=" ", flush=True)
from engine import TradingEngine
print(f"OK ({time.time()-t0:.1f}s)")

print(f"  ✓ All modules loaded in {time.time()-t0:.1f}s")

class LogEmitter(QObject):
    log_signal = pyqtSignal(list)

def main():
    print(f"  ▸ Initializing application...", end=" ", flush=True)
    
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    log_emitter = LogEmitter()
    setup_logging(gui_signal=log_emitter.log_signal)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting LLMPatternTrader GUI...")
    print(f"OK ({time.time()-t0:.1f}s)")

    try:
        config_manager = ConfigManager()
        trading_engine = TradingEngine(config_manager)
        
        # Hide console when engine finishes first pattern scan
        trading_engine.console_ready.connect(hide_console)
        
        print(f"  ▸ Creating window...", end=" ", flush=True)
        window = MainWindow(trading_engine, config_manager, loop)
        log_emitter.log_signal.connect(window.append_log_message)
        window.show()
        print(f"OK ({time.time()-t0:.1f}s)")
        
        print(f"  ✓ Ready! Console will hide once patterns are loaded.")
        print(f"  ℹ  Press LAUNCH in the app to start pattern discovery.")

        logger.info("Main window displayed.")
        with loop:
            sys.exit(loop.run_forever())
            
    except Exception as e:
        print(f"\n  ✗ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        logging.getLogger(__name__).critical(f"A critical error occurred: {e}", exc_info=True)
        input("\nPress Enter to exit...")
        sys.exit(1)
        
if __name__ == "__main__":
    main()

