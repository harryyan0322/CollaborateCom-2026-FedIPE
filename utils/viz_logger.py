"""
可视化数据记录器
支持写入 JSON / CSV / NPZ，并自动创建 viz_data 目录。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


class VizLogger:
    def __init__(self, base_dir: str | Path = "viz_data") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, obj: Any, filename: str) -> Path:
        path = self.base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def save_csv(self, rows: List[Dict[str, Any]], filename: str) -> Path:
        path = self.base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        return path

    def save_npz(self, arrays: Dict[str, Any], filename: str) -> Path:
        path = self.base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        return path


__all__ = ["VizLogger"]
