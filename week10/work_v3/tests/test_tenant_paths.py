import os
import tempfile
import shutil
from pathlib import Path
import unittest

from work import config

class TenantPathsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name) / "tenants"
        os.environ["TENANTS_BASE_DIR"] = str(base)

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TENANTS_BASE_DIR", None)

    def test_dirs_resolve(self):
        t = "t1"
        kb_idx = config.get_kb_index_dir(t)
        kb_data = config.get_kb_data_dir(t)
        orders = config.get_orders_db_path(t)
        support = config.get_support_db_path(t)
        self.assertTrue(kb_idx.endswith(f"tenants\\{t}\\faiss_index") or kb_idx.endswith(f"tenants/{t}/faiss_index"))
        self.assertTrue(kb_data.endswith(f"tenants\\{t}\\datas") or kb_data.endswith(f"tenants/{t}/datas"))
        self.assertTrue((orders is None) or orders.endswith(f"tenants\\{t}\\db\\orders.sqlite") or orders.endswith(f"tenants/{t}/db/orders.sqlite"))
        self.assertTrue(support.endswith(f"tenants\\{t}\\support.db") or support.endswith(f"tenants/{t}/support.db"))

    def test_orders_path_exists_only_when_file_present(self):
        t = "t2"
        p = Path(config._tenant_dir(t)) / "db" / "orders.sqlite"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
        got = config.get_orders_db_path(t)
        self.assertTrue(isinstance(got, str) and got.endswith("orders.sqlite"))

if __name__ == "__main__":
    unittest.main()