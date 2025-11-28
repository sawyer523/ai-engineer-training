import os
from pathlib import Path
import unittest

from work import config

class CourseTenantMapTests(unittest.TestCase):
    def setUp(self):
        p = Path(__file__).resolve().parent.parent / "tenant_courses.json"
        os.environ["COURSE_TENANT_MAP"] = str(p)

    def tearDown(self):
        os.environ.pop("COURSE_TENANT_MAP", None)

    def test_get_tenant_for_course(self):
        t = config.get_tenant_for_course("AI工程化训练营")
        self.assertEqual(t, "t1")

    def test_get_paths_for_course(self):
        paths = config.get_paths_for_course("Python入门")
        self.assertIn("tenant_id", paths)
        self.assertEqual(paths.get("tenant_id"), "t2")
        self.assertIn("kb_index_dir", paths)

if __name__ == "__main__":
    unittest.main()