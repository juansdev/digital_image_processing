import logging as log

log.basicConfig(
    level='INFO',
    format='%(asctime)s: %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        log.FileHandler('../log.txt'),
        log.StreamHandler()
    ]
)


if __name__ == '__main__':
    log.debug('Debug')
    log.info('Info')
    log.warning('Warning')
    log.error('Error')
    log.critical('Critical')
