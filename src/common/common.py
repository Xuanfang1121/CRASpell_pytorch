# -*- coding: utf-8 -*-
import os
import logging.config

from config.global_conf import PROJECT_DIR

logging.config.fileConfig(os.path.join(PROJECT_DIR, 'config/logging.conf'))
logger = logging.getLogger()
