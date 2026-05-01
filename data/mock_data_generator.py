"""
模拟舆情数据生成器
---------------------------------
该生成器用于在没有真实数据时，构造符合论文设定的用户、帖文与关系数据。
生成的数据需支持后续显式指标（HHI、KAP 加权参与度）以及隐式图模型（GraphSAGE、GAT）。
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from metrics.explicit import ExplicitMetricsCalculator
from utils.viz_logger import VizLogger


class MockDataGenerator:
    """
    基于可重复随机种子的模拟数据生成器。
    - 用户侧：区分媒体用户与普通用户，并生成基础属性与嵌入。
    - 帖文侧：生成交互量（点赞/评论/转发），供 KAP 加权参与度使用。
    - 关系侧：生成用户间的关注/互动边，为后续图模型准备 edge_list。
    """

    def __init__(
        self,
        n_users: int = 400,
        n_posts: int = 1200,
        media_ratio: float = 0.18,
        seed: int = 42,
    ) -> None:
        self.n_users = n_users
        self.n_posts = n_posts
        self.media_ratio = media_ratio
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        self.platforms = ["weibo", "wechat", "douyin", "zhihu", "bilibili"]
        # 媒体机构列表（用于 HHI 统计中的 R_i）
        self.media_outlets = [
            "NationalNews",
            "LocalNews",
            "FinancialDaily",
            "TechReview",
            "PublicVoice",
        ]
        # 话题类别
        self.topics = [
            "politics",
            "economy",
            "society",
            "technology",
            "entertainment",
            "sports",
            "education",
            "health",
        ]

    # -------- 用户与帖文生成 --------
    def _sample_timestamp(self, idx: int) -> datetime:
        """在最近 21 天内生成时间戳，保持简单的递增趋势。"""
        start = datetime.now() - timedelta(days=21)
        return start + timedelta(seconds=random.randint(0, 21 * 24 * 3600) + idx)

    def generate_users(self) -> pd.DataFrame:
        """
        生成用户表。
        字段：
            user_id, role(媒体/公众), media_outlet, platform, follower_count,
            following_count, verified, activity_score, base_embedding(8维向量)
        """
        users = []
        n_media = max(1, int(self.n_users * self.media_ratio))
        media_ids = set(random.sample(range(self.n_users), n_media))

        for i in range(self.n_users):
            is_media = i in media_ids
            role = "media" if is_media else "public"
            media_outlet = (
                random.choice(self.media_outlets) if is_media else "public_cluster"
            )

            follower = int(np.random.lognormal(mean=5.2 if is_media else 4.0, sigma=1.2))
            following = int(np.random.lognormal(mean=3.2, sigma=1.0))
            activity = float(np.clip(np.random.normal(loc=0.6, scale=0.2), 0.05, 1.2))

            # 低维基础特征向量（供 GraphSAGE/对齐操作使用）
            base_embedding = np.random.normal(0, 1, size=8).astype(float)

            users.append(
                {
                    "user_id": f"user_{i:04d}",
                    "role": role,
                    "media_outlet": media_outlet,
                    "platform": random.choices(
                        self.platforms, weights=[0.3, 0.25, 0.2, 0.15, 0.1]
                    )[0],
                    "follower_count": follower,
                    "following_count": following,
                    "verified": int(np.random.choice([0, 1], p=[0.65, 0.35])),
                    "activity_score": activity,
                    "base_embedding": base_embedding,
                }
            )

        return pd.DataFrame(users)

    def _sample_interactions(self, is_media: bool) -> Tuple[int, int, int, int]:
        """
        生成单条帖文的交互计数：
        - 媒体用户通常拥有更高曝光与转发
        - 返回 (view_count, like_count, comment_count, repost_count)
        """
        view = int(np.random.lognormal(mean=10 if is_media else 8.5, sigma=0.6))
        like = int(np.random.lognormal(mean=5.5 if is_media else 4.5, sigma=0.9))
        comment = int(np.random.lognormal(mean=4.5 if is_media else 3.5, sigma=0.9))
        repost = int(np.random.lognormal(mean=5.0 if is_media else 3.8, sigma=0.95))
        return view, like, comment, repost

    def generate_posts(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """
        生成帖文数据表。
        字段：
            post_id, user_id, timestamp, topic, sentiment_score,
            view_count, like_count, comment_count, repost_count, is_original
        """
        posts = []
        user_ids = users_df["user_id"].tolist()

        # 按用户影响力分布采样发帖概率
        influence = users_df["follower_count"].to_numpy() + 10
        prob = influence / influence.sum()

        for i in range(self.n_posts):
            user_idx = np.random.choice(len(user_ids), p=prob)
            user_row = users_df.iloc[user_idx]
            is_media = user_row["role"] == "media"

            view, like, comment, repost = self._sample_interactions(is_media)

            posts.append(
                {
                    "post_id": f"post_{i:05d}",
                    "user_id": user_row["user_id"],
                    "timestamp": self._sample_timestamp(i),
                    "topic": random.choice(self.topics),
                    "sentiment_score": float(
                        np.clip(np.random.normal(loc=0.05, scale=0.6), -1.0, 1.0)
                    ),
                    "view_count": view,
                    "like_count": like,
                    "comment_count": comment,
                    "repost_count": repost,
                    "is_original": int(np.random.choice([0, 1], p=[0.35, 0.65])),
                }
            )

        return pd.DataFrame(posts)

    # -------- 关系生成（供图模型使用） --------
    def generate_relations(
        self, users_df: pd.DataFrame, posts_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        基于关注与互动生成边列表。
        字段：
            src, dst, weight, edge_type
        权重综合考虑：
            - 关注关系（基础权重 1.0）
            - 帖文交互强度（按点赞/评论/转发折算）
        """
        edges = {}
        user_ids = users_df["user_id"].tolist()
        n_follow_edges = max(self.n_users * 4, 300)

        # 随机生成关注边
        for _ in range(n_follow_edges):
            u, v = random.sample(user_ids, 2)
            key = tuple(sorted((u, v)))
            edges[key] = edges.get(key, 0.0) + 1.0

        # 基于帖文交互累积权重
        post_counts = posts_df.groupby("user_id")[["like_count", "comment_count", "repost_count"]].sum()
        for u in user_ids:
            if u not in post_counts.index:
                continue
            interactions = post_counts.loc[u]
            total_inter = (
                0.2 * interactions["like_count"]
                + 0.3 * interactions["comment_count"]
                + 0.5 * interactions["repost_count"]
            )
            # 与部分随机用户建立互动边，权重与交互强度正相关
            partner_num = max(5, int(math.log1p(total_inter)))
            partners = random.sample(user_ids, k=min(partner_num, len(user_ids)))
            for v in partners:
                if u == v:
                    continue
                key = tuple(sorted((u, v)))
                edges[key] = edges.get(key, 0.0) + total_inter / max(partner_num, 1) / 100.0

        rows = [
            {"src": u, "dst": v, "weight": w, "edge_type": "mixed"}
            for (u, v), w in edges.items()
        ]
        return pd.DataFrame(rows)

    # -------- 汇总入口 --------
    def generate_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """生成 users_df, posts_df, edges_df 三类数据。"""
        users_df = self.generate_users()
        posts_df = self.generate_posts(users_df)
        edges_df = self.generate_relations(users_df, posts_df)

        # 记录可视化数据
        try:
            viz = VizLogger()
            # 度分布
            deg_series = pd.concat([edges_df["src"], edges_df["dst"]]).value_counts()
            # 时间戳范围
            ts_min = pd.to_datetime(posts_df["timestamp"]).min()
            ts_max = pd.to_datetime(posts_df["timestamp"]).max()
            dataset_stats = {
                "degree_distribution": deg_series.to_dict(),
                "timestamp_min": ts_min.isoformat(),
                "timestamp_max": ts_max.isoformat(),
                "num_users": int(len(users_df)),
                "num_posts": int(len(posts_df)),
                "num_edges": int(len(edges_df)),
            }
            viz.save_json(dataset_stats, "dataset_stats.json")

            # 显式指标
            explicit_calc = ExplicitMetricsCalculator()
            metrics = explicit_calc.calculate_all(posts_df, users_df)
            viz.save_csv([metrics], "explicit_metrics.csv")
        except Exception:
            # 避免数据生成失败
            pass

        return users_df, posts_df, edges_df


if __name__ == "__main__":
    generator = MockDataGenerator()
    users, posts, edges = generator.generate_all()
    print(f"用户数: {len(users)}, 帖文数: {len(posts)}, 边数: {len(edges)}")
    print(users.head(3))
    print(posts.head(3))
    print(edges.head(3))
