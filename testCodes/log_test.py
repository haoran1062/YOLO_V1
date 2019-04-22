# encoding:utf-8
import os, logging, time


def create_logger(base_path, log_name):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    fhander = logging.FileHandler('%s/%s.log'%(base_path, log_name))
    fhander.setLevel(logging.INFO)

    shander = logging.StreamHandler()
    shander.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    fhander.setFormatter(formatter) 
    shander.setFormatter(formatter) 

    logger.addHandler(fhander)
    logger.addHandler(shander)

    return logger

if __name__ == "__main__":
    log_name = 'test'
    base_path = 'testCodes'

    logger = create_logger(base_path, log_name)
    
    logger.info('hello! ')
    logger.info(' world! ')

    for i in range(100):

        logger.info('epoch {}, cost time {:.5f}'.format(i, 0.01))