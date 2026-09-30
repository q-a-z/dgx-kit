import os
import tempfile

os.environ.setdefault("DGXKIT_CACHE_DIR", tempfile.mkdtemp())  # tests never touch the real ~/.cache
from tests.test_api import env  # noqa: F401  (shared fixture)
