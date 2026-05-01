"""
显式指标计算模块
----------------
按指标体系补充近 30 维显式指标（部分为占位/近似），分组：
- 平台：类型多样性、集中度、发帖频率、扩散速度/指数、用户覆盖/互动、发帖量
- 媒体：集中度、权威度、热度、报道多样性、态度分布、持续度
- 群众：知情度、参与度（KAP）、互动质量、情感分布、跨平台一致性
- 舆情发展：传播速度、覆盖范围（平台/媒体/用户）、热度指数
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class ExplicitMetricsCalculator:
    """显式指标计算器，输出结构化的指标字典。"""

    def __init__(
        self,
        kap_weights: Optional[Dict[str, float]] = None,
        media_col: str = "media_outlet",
    ) -> None:
        self.media_col = media_col
        self.kap_weights = kap_weights or {
            "like": 0.2,
            "comment": 0.3,
            "repost": 0.5,
        }

    # -------- 基础聚合 --------
    @staticmethod
    def _interactions(posts_df: pd.DataFrame) -> pd.Series:
        like = posts_df.get("like_count", 0)
        comment = posts_df.get("comment_count", 0)
        repost = posts_df.get("repost_count", 0)
        view = posts_df.get("view_count", 0)
        return like + comment + repost + view

    @staticmethod
    def _days_span(ts: pd.Series) -> int:
        ts = ts.dropna()
        if ts.empty:
            return 1
        return max((ts.max() - ts.min()).days + 1, 1)

    # -------- HHI 媒体集中度 --------
    def media_concentration_hhi(
        self, posts_df: pd.DataFrame, users_df: pd.DataFrame
    ) -> float:
        """
        计算媒体集中度 HHI。
        步骤：
            1) 将帖文表与用户表按 user_id 关联，获取每条帖文所属媒体。
            2) 仅统计媒体用户的帖文数量 R_i。
            3) 按公式 sum((R_i / R_total)^2) 求和。
        """
        if posts_df.empty:
            return 0.0

        merged = posts_df.merge(
            users_df[["user_id", "role", self.media_col]],
            on="user_id",
            how="left",
        )
        media_posts = merged[merged["role"] == "media"]
        if media_posts.empty:
            return 0.0

        counts = media_posts[self.media_col].value_counts().astype(float)
        total = counts.sum()
        hhi = float(((counts / total) ** 2).sum())
        return hhi

    # -------- KAP 群众参与度 --------
    def public_participation_kap(self, posts_df: pd.DataFrame) -> float:
        """
        基于点赞/评论/转发的 KAP 加权参与度。
        Score = w_like * Like + w_comment * Comment + w_repost * Repost
        返回值为所有帖文的加权总和（可视为总体参与量）。
        """
        required_cols = {"like_count", "comment_count", "repost_count"}
        if posts_df.empty or not required_cols.issubset(posts_df.columns):
            return 0.0

        w_like = self.kap_weights["like"]
        w_comment = self.kap_weights["comment"]
        w_repost = self.kap_weights["repost"]

        weighted = (
            w_like * posts_df["like_count"].to_numpy()
            + w_comment * posts_df["comment_count"].to_numpy()
            + w_repost * posts_df["repost_count"].to_numpy()
        )
        return float(weighted.sum())

    # -------- 平台指标 --------
    def platform_type_diversity(self, posts_df: pd.DataFrame) -> float:
        if posts_df.empty or "platform" not in posts_df.columns:
            return 0.0
        counts = posts_df["platform"].value_counts(normalize=True)
        return float(-(counts * np.log(counts + 1e-12)).sum())

    def platform_concentration_hhi(self, posts_df: pd.DataFrame) -> float:
        if posts_df.empty or "platform" not in posts_df.columns:
            return 0.0
        counts = posts_df["platform"].value_counts().astype(float)
        total = counts.sum()
        if total == 0:
            return 0.0
        return float(((counts / total) ** 2).sum())

    def platform_post_freq(self, posts_df: pd.DataFrame) -> float:
        if posts_df.empty or "timestamp" not in posts_df.columns:
            return 0.0
        ts = posts_df["timestamp"].dropna()
        if ts.empty:
            return 0.0
        days = self._days_span(ts)
        return float(len(posts_df) / days)

    def platform_spread_speed(self, posts_df: pd.DataFrame) -> float:
        if posts_df.empty or "timestamp" not in posts_df.columns:
            return 0.0
        ts = posts_df["timestamp"]
        inter = self._interactions(posts_df)
        df = pd.DataFrame({"ts": ts, "inter": inter}).dropna()
        if df.empty:
            return 0.0
        df = df.sort_values("ts")
        cum = df["inter"].cumsum()
        target = cum.iloc[-1] * 0.5
        idx = cum.searchsorted(target, side="left")
        if idx >= len(df):
            return 0.0
        t50 = df.iloc[idx]["ts"]
        span = (t50 - df.iloc[0]["ts"]).total_seconds()
        return float(span)

    def platform_spread_index(self, posts_df: pd.DataFrame) -> float:
        if posts_df.empty or "timestamp" not in posts_df.columns:
            return 0.0
        ts = posts_df["timestamp"].dropna()
        if ts.empty:
            return 0.0
        inter_sum = self._interactions(posts_df).sum()
        hours = max((ts.max() - ts.min()).total_seconds() / 3600.0, 1.0)
        return float(inter_sum / hours)

    def platform_user_coverage(self, posts_df: pd.DataFrame, users_df: pd.DataFrame) -> float:
        if posts_df.empty or users_df.empty or "user_id" not in posts_df.columns:
            return 0.0
        active_users = posts_df["user_id"].dropna().nunique()
        total_users = max(len(users_df), 1)
        return float(active_users / total_users)

    def platform_interaction_index(self, posts_df: pd.DataFrame) -> float:
        required_cols = {"like_count", "comment_count", "repost_count", "user_id"}
        if posts_df.empty or not required_cols.issubset(posts_df.columns):
            return 0.0
        interactions = self._interactions(posts_df).sum()
        active_users = max(posts_df["user_id"].dropna().nunique(), 1)
        return float(interactions / active_users)

    def platform_post_count(self, posts_df: pd.DataFrame) -> float:
        return float(len(posts_df))

    # -------- 媒体指标 --------
    def media_heat_index(self, posts_df: pd.DataFrame, users_df: pd.DataFrame) -> float:
        required_cols = {"like_count", "comment_count", "repost_count", "user_id"}
        if posts_df.empty or not required_cols.issubset(posts_df.columns):
            return 0.0
        merged = posts_df.merge(
            users_df[["user_id", "role", self.media_col]],
            on="user_id",
            how="left",
        )
        media_posts = merged[merged["role"] == "media"]
        if media_posts.empty:
            return 0.0
        interactions = self._interactions(media_posts)
        return float(interactions.mean())

    def media_persistence(self, posts_df: pd.DataFrame, users_df: pd.DataFrame) -> float:
        if posts_df.empty or "timestamp" not in posts_df.columns:
            return 0.0
        merged = posts_df.merge(
            users_df[["user_id", "role"]],
            on="user_id",
            how="left",
        )
        media_posts = merged[merged["role"] == "media"]
        if media_posts.empty:
            return 0.0
        ts = media_posts["timestamp"].dropna()
        if ts.empty:
            return 0.0
        active_days = ts.dt.date.nunique()
        total_days = (ts.max() - ts.min()).days + 1
        total_days = max(total_days, 1)
        return float(active_days / total_days)

    def media_authority(self, users_df: pd.DataFrame) -> float:
        if users_df.empty or "role" not in users_df.columns:
            return 0.0
        media_users = users_df[users_df["role"] == "media"]
        if media_users.empty:
            return 0.0
        # 使用粉丝数或 influence 作为权威度
        if "cnt_follower" in media_users.columns:
            return float(media_users["cnt_follower"].mean())
        return float(media_users.get("influence", 0).mean())

    def media_report_diversity(self, posts_df: pd.DataFrame, users_df: pd.DataFrame) -> float:
        if posts_df.empty:
            return 0.0
        merged = posts_df.merge(users_df[["user_id", "role"]], on="user_id", how="left")
        media_posts = merged[merged["role"] == "media"]
        if media_posts.empty or "topic" not in media_posts.columns:
            return 0.0
        counts = media_posts["topic"].value_counts(normalize=True)
        return float(-(counts * np.log(counts + 1e-12)).sum())

    def media_attitude_distribution(self, posts_df: pd.DataFrame, users_df: pd.DataFrame) -> Dict[str, float]:
        merged = posts_df.merge(users_df[["user_id", "role"]], on="user_id", how="left")
        media_posts = merged[merged["role"] == "media"]
        if media_posts.empty or "sentiment_label" not in media_posts.columns:
            return {"media_pos": 0.0, "media_neu": 1.0, "media_neg": 0.0}
        counts = media_posts["sentiment_label"].value_counts(normalize=True)
        return {
            "media_pos": float(counts.get("pos", 0.0)),
            "media_neu": float(counts.get("neu", 0.0)),
            "media_neg": float(counts.get("neg", 0.0)),
        }

    def sentiment_distribution(self, posts_df: pd.DataFrame) -> Dict[str, float]:
        if posts_df.empty or "sentiment_label" not in posts_df.columns:
            return {"pos": 0.0, "neu": 1.0, "neg": 0.0}
        counts = posts_df["sentiment_label"].value_counts(normalize=True)
        return {
            "pos": float(counts.get("pos", 0.0)),
            "neu": float(counts.get("neu", 0.0)),
            "neg": float(counts.get("neg", 0.0)),
        }

    # -------- 群众 / 舆情发展补充 --------
    def crowd_awareness(self, posts_df: pd.DataFrame, users_df: pd.DataFrame) -> float:
        if posts_df.empty:
            return 0.0
        # 用总浏览量或发帖数近似
        if "view_count" in posts_df.columns:
            return float(posts_df["view_count"].sum())
        return float(len(posts_df))

    def crowd_interaction_quality(self, posts_df: pd.DataFrame) -> float:
        if posts_df.empty:
            return 0.0
        comments = posts_df.get("comment_count", 0).sum()
        reposts = posts_df.get("repost_count", 0).sum()
        likes = posts_df.get("like_count", 0).sum()
        denom = likes + comments + reposts
        if denom == 0:
            return 0.0
        return float((comments + reposts) / denom)

    def cross_platform_sentiment_std(self, posts_df: pd.DataFrame) -> float:
        if posts_df.empty or "platform" not in posts_df.columns or "sentiment_score" not in posts_df.columns:
            return 0.0
        grouped = posts_df.groupby("platform")["sentiment_score"].mean()
        if len(grouped) <= 1:
            return 0.0
        return float(grouped.std())

    def propagation_speed(self, posts_df: pd.DataFrame) -> float:
        # 同平台_spread_speed，但不分平台
        return self.platform_spread_speed(posts_df)

    def coverage_scope(self, posts_df: pd.DataFrame, users_df: pd.DataFrame) -> Dict[str, float]:
        platforms = posts_df["platform"].nunique() if "platform" in posts_df.columns else 0
        media_count = users_df[users_df.get("role", "") == "media"].shape[0] if "role" in users_df.columns else 0
        active_users = posts_df["user_id"].nunique() if "user_id" in posts_df.columns else 0
        return {
            "coverage_platforms": float(platforms),
            "coverage_media": float(media_count),
            "coverage_users": float(active_users),
        }

    def hotness_index(self, posts_df: pd.DataFrame) -> float:
        if posts_df.empty or "timestamp" not in posts_df.columns:
            return 0.0
        df = posts_df.copy()
        df["date"] = df["timestamp"].dt.date
        inter = self._interactions(df)
        daily = inter.groupby(df["date"]).sum()
        if daily.empty:
            return 0.0
        return float(daily.max())

    # -------- 汇总接口 --------
    def calculate_all(self, posts_df: pd.DataFrame, users_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算并返回全部显式指标（约 30 维，部分为占位/近似）。
        """
        sentiment = self.sentiment_distribution(posts_df)
        media_sent = self.media_attitude_distribution(posts_df, users_df)
        coverage = self.coverage_scope(posts_df, users_df)

        metrics = {
            # 平台
            "platform_type_diversity": self.platform_type_diversity(posts_df),
            "platform_concentration_hhi": self.platform_concentration_hhi(posts_df),
            "platform_post_freq": self.platform_post_freq(posts_df),
            "platform_spread_speed": self.platform_spread_speed(posts_df),
            "platform_spread_index": self.platform_spread_index(posts_df),
            "platform_user_coverage": self.platform_user_coverage(posts_df, users_df),
            "platform_interaction_index": self.platform_interaction_index(posts_df),
            "platform_post_count": self.platform_post_count(posts_df),

            # 媒体
            "media_concentration_hhi": self.media_concentration_hhi(posts_df, users_df),
            "media_authority": self.media_authority(users_df),
            "media_heat_index": self.media_heat_index(posts_df, users_df),
            "media_report_diversity": self.media_report_diversity(posts_df, users_df),
            "media_persistence": self.media_persistence(posts_df, users_df),
            "media_attitude_pos": media_sent["media_pos"],
            "media_attitude_neu": media_sent["media_neu"],
            "media_attitude_neg": media_sent["media_neg"],

            # 群众
            "crowd_awareness": self.crowd_awareness(posts_df, users_df),
            "public_participation_kap": self.public_participation_kap(posts_df),
            "crowd_interaction_quality": self.crowd_interaction_quality(posts_df),
            "sentiment_pos": sentiment["pos"],
            "sentiment_neu": sentiment["neu"],
            "sentiment_neg": sentiment["neg"],
            "cross_platform_sentiment_std": self.cross_platform_sentiment_std(posts_df),

            # 舆情发展
            "propagation_speed": self.propagation_speed(posts_df),
            "hotness_index": self.hotness_index(posts_df),
            "coverage_platforms": coverage["coverage_platforms"],
            "coverage_media": coverage["coverage_media"],
            "coverage_users": coverage["coverage_users"],
        }
        return metrics
