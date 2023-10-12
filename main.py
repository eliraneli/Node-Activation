import os
import shutil
import torch
from logger import Logger
from Trainer import Trainer
from config import Config
from Code import Code

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'output.log')
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'), exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = Config()

logger = Logger('main', LOG_FILE)

codewords = Code(CONFIG.H_filename, CONFIG.G_filename)

trainer = Trainer(codewords, CONFIG, LOG_FILE, DEVICE)

