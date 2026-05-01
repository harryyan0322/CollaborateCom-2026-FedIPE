"""
真实数据加载器
- 读取 data/proprecess/ 下预处理好的 nodes.csv / posts.csv / edges.csv / features.npz
- 输出与模拟器相同结构的 (users_df, posts_df, edges_df)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "proprecess"


class RealDataLoader:
    def __init__(self):
        self.nodes_path = DATA_DIR / "nodes.csv"
        self.posts_path = DATA_DIR / "posts.csv"
        self.edges_path = DATA_DIR / "edges.csv"
        self.features_path = DATA_DIR / "features.npz"

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        users_df = pd.read_csv(self.nodes_path)
        posts_df = pd.read_csv(self.posts_path)
        edges_df = pd.read_csv(self.edges_path)

        # 对齐字段
        users_df = users_df.rename(columns={"id": "user_id"})
        posts_df = posts_df.rename(columns={"id": "post_id", "account_id": "user_id", "send_ts": "timestamp"})

        # edges_df 添加 weight 和 edge_type 列（如果不存在，作为后备）
        # 注意：正常情况下这些列应该在 preprocess_realdata.py 生成 edges.csv 时就已经包含
        if "weight" not in edges_df.columns:
            edges_df["weight"] = 1.0
        if "edge_type" not in edges_df.columns:
            # 如果 edge_type 不存在，默认设为 "follow"（关注关系）
            edges_df["edge_type"] = "follow"

        # 时间转换
        if "timestamp" in posts_df.columns:
            posts_df["timestamp"] = pd.to_datetime(posts_df["timestamp"], unit="s", errors="coerce")

        # 平台字段（单平台场景补 "single"）
        if "platform" not in posts_df.columns:
            posts_df["platform"] = "single"

        # 情感占位（若缺失）
        if "sentiment_label" not in posts_df.columns:
            posts_df["sentiment_label"] = "neu"
        if "sentiment_score" not in posts_df.columns:
            posts_df["sentiment_score"] = 0.0

        # 交互计数若不存在则置 0
        for col in ["like_count", "repost_count", "comment_count", "view_count"]:
            if col not in posts_df.columns:
                posts_df[col] = 0

        # 主题/原帖标记
        if "topic" not in posts_df.columns:
            posts_df["topic"] = ""
        posts_df["is_original"] = posts_df["parent_id"].isna().astype(int)

        # 活跃度与验证标记（简单占位，可按需调整）
        if "activity_score" not in users_df.columns:
            users_df["activity_score"] = users_df["retweet_probability"].fillna(0)
        if "verified" not in users_df.columns:
            users_df["verified"] = 0

        # role 和 media_outlet：用于显式指标计算
        # 将 user_type 映射到 role（media -> media, 其他 -> public）
        if "user_type" in users_df.columns:
            users_df["role"] = users_df["user_type"].apply(
                lambda x: "media" if x == "media" else "public"
            )
        else:
            users_df["role"] = "public"

        # media_outlet：媒体用户使用 name 或基于 user_id 生成，非媒体用户设为 public_cluster
        if "media_outlet" not in users_df.columns:
            def get_media_outlet(row):
                if row["role"] == "media":
                    # 使用 name 字段，如果为空或 NaN 则使用 user_id 的前缀
                    name = row.get("name", "")
                    if pd.notna(name) and str(name).strip():
                        return str(name).strip()
                    else:
                        # 从 user_id 提取标识符（例如 twitter/123 -> twitter_123）
                        user_id = str(row.get("user_id", ""))
                        return user_id.replace("/", "_") if user_id else "unknown_media"
                else:
                    return "public_cluster"
            
            users_df["media_outlet"] = users_df.apply(get_media_outlet, axis=1)

        # base_embedding：从 features.npz 取出前 8 维
        base_emb = None
        if self.features_path.exists():
            f = np.load(self.features_path, allow_pickle=True)
            feats = f["features"]
            base_emb = feats[:, :8] if feats.shape[1] >= 8 else feats
        users_df["base_embedding"] = list(base_emb) if base_emb is not None else [np.zeros(8, dtype=float)] * len(users_df)

        return users_df, posts_df, edges_df


__all__ = ["RealDataLoader"]
