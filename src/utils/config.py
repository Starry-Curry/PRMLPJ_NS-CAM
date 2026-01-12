import os
from dotenv import load_dotenv

# Load .env file
load_dotenv(override=True)


# OpenAI-compatible settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEYS') or os.getenv('OPENAI_API_KEY')

OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
OPENAI_MODEL = os.getenv('OPENAI_MODEL') or os.getenv('MODEL') or 'deepseek-v3'
OPENAI_EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL') or 'text-embedding-v2'

# Local embedding model path (user provided)
LOCAL_EMBEDDING_PATH = os.getenv('LOCAL_EMBEDDING_PATH') or r'D:\Classes\PRML\PJ\NS-CAM\model\bge-m3\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181'


# Runtime / environment
CONDA_ENV = os.getenv('CONDA_DEFAULT_ENV') or os.getenv('CONDA_PREFIX')

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_LOG_PATH = os.path.join(PROJECT_ROOT, 'logs', 'nscam.json')
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
