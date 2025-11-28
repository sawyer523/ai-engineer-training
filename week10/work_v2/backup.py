"""知识库 FAISS 备份脚本

提供将知识库 FAISS 索引文件打包备份到
`work/backup` 的功能，包含：
- 按日期与版本号自动命名备份文件（kb_YYYYMMDD[_vN].zip）
- 在压缩前校验索引文件完整性（.faiss 非空、.pkl 可反序列化）
- 保留原始目录结构（以项目根的相对路径写入 zip）
- 记录备份日志（时间、文件名、版本、状态）并进行完善的错误处理
"""

import os
import sys
import datetime
import zipfile
import shutil
import logging
import pickle
from pathlib import Path


class IntegrityError(Exception):
    """完整性校验失败异常。

    在校验 FAISS 相关文件（如 `.faiss` 与 `index.pkl`）时，如果发现
    文件缺失、为空或反序列化失败，会抛出此异常。
    """
    pass


class InsufficientSpaceError(Exception):
    """磁盘空间不足异常。

    当估算需要的备份空间大于目标磁盘可用空间时，抛出此异常并记录日志。
    """
    pass


def today_str():
    """返回当前日期字符串，格式为 YYYYMMDD。"""
    return datetime.date.today().strftime("%Y%m%d")


def setup_logger(backup_dir: Path) -> logging.Logger:
    """初始化备份日志记录器。

    在 `backup_dir/backup.log` 写入日志，并同时输出到控制台。
    """
    backup_dir.mkdir(parents=True, exist_ok=True)  # 确保备份目录存在
    logger = logging.getLogger("kb_backup")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(str(backup_dir / "backup.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def parse_version(name: str) -> int:
    """解析文件名中的版本号后缀（_vN）。不存在则返回 0。"""
    base = Path(name).stem
    if "_v" in base:
        try:
            return int(base.split("_v")[-1])
        except ValueError:
            return 0
    return 0


def next_backup_name(backup_dir: Path, date_s: str) -> Path:
    """根据日期与已存在的文件计算下一个备份文件名。

    - 首次：`kb_YYYYMMDD.zip`
    - 后续：`kb_YYYYMMDD_vN.zip`（N 递增）
    """
    base = backup_dir / f"kb_{date_s}.zip"
    if not base.exists():
        return base
    existing = list(backup_dir.glob(f"kb_{date_s}_v*.zip"))
    if not existing:
        return backup_dir / f"kb_{date_s}_v1.zip"
    max_v = 0
    for p in existing:
        v = parse_version(p.name)
        if v > max_v:
            max_v = v
    return backup_dir / f"kb_{date_s}_v{max_v + 1}.zip"


def list_faiss_files(source_dir: Path) -> list[Path]:
    """列出源目录下所有需要被打包的文件。"""
    return [p for p in source_dir.rglob("*") if p.is_file()]


def validate_faiss(source_dir: Path):
    """校验 FAISS 索引文件完整性。

    要求：
    - 存在 `.faiss` 与 `.pkl` 文件；
    - 文件非空；
    - `index.pkl` 可被 `pickle.load` 成功反序列化。
    """
    faiss_bin = None
    pkl = None
    for p in list_faiss_files(source_dir):
        if p.name.endswith(".faiss"):
            faiss_bin = p
        if p.name.endswith(".pkl"):
            pkl = p
    if faiss_bin is None or pkl is None:
        raise IntegrityError("missing faiss index files")
    if faiss_bin.stat().st_size <= 0:
        raise IntegrityError("faiss index file empty")
    if pkl.stat().st_size <= 0:
        raise IntegrityError("pickle index file empty")
    with pkl.open("rb") as f:
        try:
            _ = pickle.load(f)
        except Exception as e:
            raise IntegrityError(f"pickle load failed: {e}")


def get_free_space_bytes(path: Path) -> int:
    """返回目标路径所在磁盘的可用空间（字节）。"""
    usage = shutil.disk_usage(str(path))
    return int(usage.free)


def estimate_backup_size(file_paths: list[Path]) -> int:
    """估算备份所需空间。

    以文件总大小为基础，加入一定压缩元数据与安全冗余（10% 或至少 1MB）。
    """
    total = 0
    for p in file_paths:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    overhead = max(1_000_000, int(total * 0.1))
    return total + overhead


def create_zip(backup_zip: Path, files: list[Path], arc_root: Path):
    """创建 zip 备份文件并进行压缩结果校验。

    使用 `arc_root` 作为相对路径根，以保留原始目录结构。
    压缩完成后重新打开 zip，使用 `testzip()` 进行一致性检查。
    """
    with zipfile.ZipFile(str(backup_zip), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            rel = p.relative_to(arc_root)
            zf.write(str(p), rel.as_posix())
    with zipfile.ZipFile(str(backup_zip), "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise IntegrityError(f"zip corrupt: {bad}")


def close_logger(logger: logging.Logger):
    """关闭并移除所有日志处理器。

    Windows 上如果不关闭文件句柄，临时目录或备份目录可能无法删除。
    """
    for h in list(logger.handlers):
        try:
            h.flush()
        except Exception:
            pass
        try:
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)


def perform_backup(source_dir: Path, backup_dir: Path) -> Path:
    """执行备份主流程。

    步骤：
    1. 初始化日志并检查源目录存在；
    2. 校验 FAISS 文件完整性；
    3. 估算备份空间并检查磁盘可用空间；
    4. 生成备份文件名并创建 zip；
    5. 记录成功日志并返回备份文件路径。
    """
    logger = setup_logger(backup_dir)
    try:
        if not source_dir.exists():  # 源目录不存在
            raise FileNotFoundError(str(source_dir))
        validate_faiss(source_dir)  # 完整性校验
        files = list_faiss_files(source_dir)  # 枚举待备份文件
        need = estimate_backup_size(files)  # 估算所需空间
        free = get_free_space_bytes(backup_dir)  # 查询可用空间
        if free < need:  # 空间不足
            raise InsufficientSpaceError(f"required={need} free={free}")
        name = next_backup_name(backup_dir, today_str())  # 计算文件名
        create_zip(name, files, source_dir.parent)  # 创建压缩包
        v = parse_version(name.name)
        logger.info(f"backup success | name={name.name} | version={v}")
        close_logger(logger)
        return name
    except InsufficientSpaceError as e:
        logger.error(f"backup failed | insufficient_space | {e}")
        close_logger(logger)
        raise
    except IntegrityError as e:
        logger.error(f"backup failed | integrity_error | {e}")
        close_logger(logger)
        raise
    except PermissionError as e:
        logger.error(f"backup failed | permission_error | {e}")
        close_logger(logger)
        raise
    except Exception as e:
        logger.error(f"backup failed | unexpected_error | {e}")
        close_logger(logger)
        raise


def main():
    """命令行入口：支持自定义源与目标目录以及低磁盘模拟。"""
    work_dir = Path(__file__).parent
    default_source = work_dir / "faiss_index"
    default_backup = work_dir / "backup"
    source = default_source
    dest = default_backup
    simulate_low = False
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--source" and i + 1 < len(args):
            source = Path(args[i + 1])
            i += 2
            continue
        if a == "--dest" and i + 1 < len(args):
            dest = Path(args[i + 1])
            i += 2
            continue
        if a == "--simulate-low-disk":
            simulate_low = True
            i += 1
            continue
        i += 1
    dest.mkdir(parents=True, exist_ok=True)  # 确保备份目录存在
    if simulate_low:
        global get_free_space_bytes
        def _fake_free(_path: Path) -> int:
            return 0
        get_free_space_bytes = _fake_free
    try:
        p = perform_backup(source, dest)
        print(str(p))
        sys.exit(0)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()