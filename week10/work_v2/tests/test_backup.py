import unittest
import zipfile
import tempfile
import shutil
import io
import pickle
from pathlib import Path

from work import backup


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "faiss_index"
        self.dest = self.root / "backup"
        self.src.mkdir(parents=True, exist_ok=True)
        self.dest.mkdir(parents=True, exist_ok=True)
        with (self.src / "index.faiss").open("wb") as f:
            f.write(b"FAISS\x00DATA\x01\x02\x03")
        with (self.src / "index.pkl").open("wb") as f:
            pickle.dump({"dim": 128, "count": 10}, f)

    def tearDown(self):
        self.tmp.cleanup()

    def test_version_increment_same_day(self):
        d = backup.today_str()
        p1 = backup.perform_backup(self.src, self.dest)
        p2 = backup.perform_backup(self.src, self.dest)
        self.assertTrue((self.dest / f"kb_{d}.zip").exists())
        self.assertTrue((self.dest / f"kb_{d}_v1.zip").exists())
        self.assertEqual(p1.name, f"kb_{d}.zip")
        self.assertEqual(p2.name, f"kb_{d}_v1.zip")

    def test_zip_integrity_and_restore(self):
        d = backup.today_str()
        p = backup.perform_backup(self.src, self.dest)
        with zipfile.ZipFile(str(p), "r") as zf:
            self.assertIsNone(zf.testzip())
            out = self.root / "restore"
            zf.extractall(out)
        self.assertTrue((out := self.root / "restore" / "faiss_index" / "index.faiss").exists())
        self.assertTrue((self.root / "restore" / "faiss_index" / "index.pkl").exists())
        with (self.src / "index.faiss").open("rb") as f1, out.open("rb") as f2:
            self.assertEqual(f1.read(), f2.read())

    def test_low_disk_space_handling(self):
        orig = backup.get_free_space_bytes
        try:
            backup.get_free_space_bytes = lambda _p: 0
            with self.assertRaises(backup.InsufficientSpaceError):
                backup.perform_backup(self.src, self.dest)
            log = self.dest / "backup.log"
            self.assertTrue(log.exists())
            txt = log.read_text(encoding="utf-8")
            self.assertIn("insufficient_space", txt)
        finally:
            backup.get_free_space_bytes = orig

    def test_log_records_time_version_status(self):
        p = backup.perform_backup(self.src, self.dest)
        log = self.dest / "backup.log"
        txt = log.read_text(encoding="utf-8")
        self.assertIn("backup success", txt)
        self.assertIn(f"name={p.name}", txt)
        self.assertIn("version=", txt)


if __name__ == "__main__":
    unittest.main()