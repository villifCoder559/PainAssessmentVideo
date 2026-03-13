import builtins
import logging
import os
import sys
from typing import Optional

_logger: Optional[logging.Logger] = None


def setup_logger(output_dir: str, log_filename: str = 'run.log') -> logging.Logger:
  """
  Configure a dual-output logger (console + file) and override builtins.print.

  After calling this function, all print() calls in the process are redirected
  to the logger, which writes to both the console and a log file. If print()
  is called with an explicit file= argument other than stdout/stderr, the
  original behaviour is preserved.

  Args:
    output_dir:   Directory where the log file will be written. Must exist.
    log_filename: Name of the log file inside output_dir. Default: 'run.log'.

  Returns:
    Configured Logger instance.
  """
  global _logger

  log_path = os.path.join(output_dir, log_filename)

  _logger = logging.getLogger('pain_assessment')
  _logger.setLevel(logging.DEBUG)
  _logger.handlers.clear()
  _logger.propagate = False

  fmt = logging.Formatter('%(message)s')

  fh = logging.FileHandler(log_path, encoding='utf-8')
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(fmt)

  sh = logging.StreamHandler(sys.stdout)
  sh.setLevel(logging.DEBUG)
  sh.setFormatter(fmt)

  _logger.addHandler(fh)
  _logger.addHandler(sh)

  _original_print = builtins.print

  def _print_to_logger(*args, sep=' ', end='\n', file=None, flush=False):
    """
    Replacement for builtins.print that forwards output to the logger.

    Args:
      *args: Objects to print (converted to str and joined with sep).
      sep:   String inserted between values. Default: ' '.
      end:   String appended after the last value. Default: '\\n'.
      file:  If set to something other than stdout/stderr, use the original
             print() so file-targeted writes are unaffected.
      flush: Ignored (logging handles flushing internally).
    """
    if file is not None and file not in (sys.stdout, sys.stderr):
      _original_print(*args, sep=sep, end=end, file=file, flush=flush)
      return
    msg = sep.join(str(a) for a in args)
    # Strip the trailing newline that print() would add; logging adds its own.
    _logger.info(msg)

  builtins.print = _print_to_logger
  return _logger
