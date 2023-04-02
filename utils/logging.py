import logging, socket, os

def get_logger(name, level='info'):
    logger = logging.Logger(name)
    level = {
        'crititical': 50,
        'error': 40,
        'warning': 30,
        'info': 20,
        'debug': 10
    }[level]
    logger.setLevel(level)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(f'%(asctime)s [{socket.gethostname()}:%(process)d] [%(levelname)s] %(message)s') 
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger