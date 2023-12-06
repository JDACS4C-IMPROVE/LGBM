import sys
import logging
import platform
# import psutil
from datetime import datetime


class Logger():
    """
    Examples:
        import classlogger.py
        logfilename = os.path.join(outdir, 'logfile.log')
        lg = classlogger.Logger(logfilename=logfilename)
        lg.logger.info(f'File path: {file_path}')
    """
    def __init__(self, logfilename='logfile.log'):
        """ Create logger. Output to file and console. """
        # Create file handler (output to file)
        # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        # "[%(asctime)s %(process)d] %(message)s"
        # fileFormatter = logging.Formatter("%(asctime)s : %(threadName)-12.12s : %(levelname)-5.5s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fileFormatter = logging.Formatter("%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fileHandler = logging.FileHandler(filename=logfilename)
        fileHandler.setFormatter(fileFormatter)
        fileHandler.setLevel(logging.INFO)
        self.fileHandler = fileHandler

        # Create console handler (output to console/terminal)
        # consoleFormatter = logging.Formatter("%(name)-12s : %(levelname)-8s : %(message)s")
        consoleFormatter = logging.Formatter("%(message)s")
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(consoleFormatter)
        consoleHandler.setLevel(logging.INFO)
        self.consoleHandler = consoleHandler

        # Create logger and add handlers
        # logger = logging.getLogger(__name__)
        logger = logging.getLogger('')
        logger.setLevel(logging.INFO)
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)
        self.logger = logger

        # from combo (when use candle)
        # for log in [logger, uno_data.logger]:
        #     log.setLevel(logging.DEBUG)
        #     log.addHandler(fh)
        #     log.addHandler(sh)

        self.logger.info('{}'.format('-' * 90))
        self.logger.info(datetime.now())
        # self.logger.info(f'Machine: {platform.node()} ({platform.system()}, {psutil.cpu_count()} CPUs)')
        #return logger


    def stop_logger(self):
        n_seps = 70
        self.logger.info('\nStop logger.')
        self.logger.info('{}\n'.format('-' * n_seps))
        self.logger.removeHandler(self.fileHandler)
        self.logger.removeHandler(self.consoleHandler)


def get_print_func(logger=None):
    """ Returns the python 'print' function if logger is None. Othersiwe, returns logger.info. """
    return print if logger is None else logger.info
