"""
FedIPE multi-platform dataset construction.

This module turns either the real preprocessed corpus or mock data into the
three heterogeneous platforms used by the paper:
- Platform A: Twitter entertainment, medium-density directed graph
- Platform B: Twitter sports + animals, 40% edge-dropped sparse directed graph
- Platform C: Reddit-style politics proxy with user-subreddit bipartite topology
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import torch

from data.mock_data_generator import MockDataGenerator
from data.real_data_loader import RealDataLoader


CORE_SEMANTIC_ANCHORS: List[str] = [
    "reach_breadth",
    "engagement_depth",
    "reshare_amplification",
    "discussion_intensity",
    "temporal_velocity",
    "source_authority",
    "audience_diversity",
    "sentiment_polarization",
    "cross_community_penetration",
    "structural_persistence",
]


@dataclass
class PlatformGraphData:
    platform_id: str
    display_name: str
    topology: str
    users_df: pd.DataFrame
    posts_df: pd.DataFrame
    edges_df: pd.DataFrame
    node_ids: List[str]
    node_id_to_idx: Dict[str, int]
    edge_index: torch.Tensor
    node_features: torch.Tensor
    explicit_local: torch.Tensor
    core_targets: torch.Tensor
    labels: torch.Tensor
    target_node_indices: torch.Tensor
    target_node_ids: List[str]
    train_indices: torch.Tensor
    val_indices: torch.Tensor
    test_indices: torch.Tensor
    local_feature_names: List[str]
    core_feature_names: List[str]
    metadata: Dict[str, Any]
    nx_graph: nx.Graph

    @property
    def num_targets(self) -> int:
        return int(self.target_node_indices.numel())

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.size(0))


def _minmax_scale_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.astype(float).copy()
    for col in out.columns:
        col_min = out[col].min()
        col_max = out[col].max()
        if pd.isna(col_min) or pd.isna(col_max) or abs(col_max - col_min) < 1e-12:
            out[col] = 0.0
        else:
            out[col] = (out[col] - col_min) / (col_max - col_min)
    return out.fillna(0.0)


def _normalized_label(series: pd.Series) -> pd.Series:
    logged = np.log1p(series.astype(float).clip(lower=0.0))
    denom = logged.max() - logged.min()
    if denom < 1e-12:
        return pd.Series(np.zeros(len(logged), dtype=float), index=series.index)
    return (logged - logged.min()) / denom


def _entropy(values: pd.Series) -> float:
    values = values.dropna().astype(str)
    if values.empty:
        return 0.0
    probs = values.value_counts(normalize=True).to_numpy(dtype=float)
    return float(-(probs * np.log(probs + 1e-12)).sum())


def _ensure_post_columns(posts_df: pd.DataFrame) -> pd.DataFrame:
    posts = posts_df.copy()
    for col in ["like_count", "comment_count", "repost_count", "view_count", "url_count", "tag_count", "at_count"]:
        if col not in posts.columns:
            posts[col] = 0
    if "timestamp" not in posts.columns:
        posts["timestamp"] = pd.Timestamp("2024-01-01")
    posts["timestamp"] = pd.to_datetime(posts["timestamp"], errors="coerce")
    posts["timestamp"] = posts["timestamp"].fillna(pd.Timestamp("2024-01-01"))
    if "topic" not in posts.columns:
        posts["topic"] = "generic"
    if "sentiment_score" not in posts.columns:
        posts["sentiment_score"] = 0.0
    if "sentiment_label" not in posts.columns:
        posts["sentiment_label"] = "neu"
    if "is_original" not in posts.columns:
        posts["is_original"] = 1
    return posts


def _ensure_user_columns(users_df: pd.DataFrame) -> pd.DataFrame:
    users = users_df.copy()
    rename_map = {
        "cnt_follower": "follower_count",
        "cnt_following": "following_count",
    }
    users = users.rename(columns=rename_map)
    for col in ["follower_count", "following_count", "verified", "activity_score"]:
        if col not in users.columns:
            users[col] = 0
    if "base_embedding" not in users.columns:
        users["base_embedding"] = [np.zeros(8, dtype=float) for _ in range(len(users))]
    return users


def _select_active_users(
    posts_df: pd.DataFrame,
    max_users: int,
    seed: int,
    edges_df: pd.DataFrame | None = None,
) -> List[str]:
    if posts_df.empty:
        return []
    user_scores = (
        posts_df.groupby("user_id")[["like_count", "comment_count", "repost_count", "view_count"]]
        .sum()
        .sum(axis=1)
    )
    user_scores = np.log1p(user_scores)
    candidate_users = pd.Index(user_scores.index)
    if edges_df is not None and not edges_df.empty:
        edge_subset = edges_df[
            edges_df["src"].isin(candidate_users) & edges_df["dst"].isin(candidate_users)
        ]
        degree = pd.concat([edge_subset["src"], edge_subset["dst"]]).value_counts()
        combined = pd.DataFrame({"activity": user_scores}).join(degree.rename("degree")).fillna(0.0)
        combined["score"] = combined["activity"].rank(pct=True) + combined["degree"].rank(pct=True)
        ranked = combined["score"].sort_values(ascending=False)
    else:
        ranked = user_scores.sort_values(ascending=False)
    if len(ranked) <= max_users:
        return ranked.index.tolist()
    top_pool = ranked.head(max_users * 3).index.to_numpy()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(top_pool, size=max_users, replace=False)
    return sorted(chosen.tolist())


def _aggregate_user_statistics(
    users_df: pd.DataFrame,
    posts_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    community_map: Dict[str, List[str]] | None = None,
    edge_survival: pd.Series | None = None,
) -> pd.DataFrame:
    posts = _ensure_post_columns(posts_df)
    users = _ensure_user_columns(users_df)

    base = users[["user_id", "follower_count", "following_count", "verified", "activity_score"]].copy()
    grouped = posts.groupby("user_id")

    post_count = grouped.size().rename("post_count")
    sums = grouped[["like_count", "comment_count", "repost_count", "view_count", "url_count", "tag_count", "at_count"]].sum()
    means = grouped[["sentiment_score"]].mean().rename(columns={"sentiment_score": "sentiment_mean"})
    stds = grouped[["sentiment_score"]].std().fillna(0.0).rename(columns={"sentiment_score": "sentiment_std"})
    originality = grouped["is_original"].mean().rename("originality_ratio")
    topic_entropy = grouped["topic"].apply(_entropy).rename("topic_entropy")

    day_stats = posts.assign(day=posts["timestamp"].dt.date).groupby("user_id").agg(
        active_days=("day", "nunique"),
        first_ts=("timestamp", "min"),
        last_ts=("timestamp", "max"),
    )
    active_span = (day_stats["last_ts"] - day_stats["first_ts"]).dt.total_seconds().div(86400.0).add(1.0)
    day_stats["active_span_days"] = active_span.clip(lower=1.0)

    latency = (
        posts.sort_values(["user_id", "timestamp"])
        .groupby("user_id")["timestamp"]
        .apply(lambda s: float(s.diff().dt.total_seconds().dropna().mean() / 3600.0) if len(s) > 1 else 0.0)
        .rename("mean_gap_hours")
    )

    degree = pd.concat([edges_df["src"], edges_df["dst"]]).value_counts().rename("graph_degree")

    stats = (
        base.set_index("user_id")
        .join(post_count)
        .join(sums)
        .join(means)
        .join(stds)
        .join(originality)
        .join(topic_entropy)
        .join(day_stats[["active_days", "active_span_days"]])
        .join(latency)
        .join(degree)
        .fillna(0.0)
    )
    stats["post_freq"] = stats["post_count"] / stats["active_span_days"].clip(lower=1.0)
    stats["engagement_ratio"] = (
        stats["like_count"] + stats["comment_count"] + stats["repost_count"]
    ) / (stats["view_count"] + 1.0)

    if community_map is None:
        stats["community_count"] = 1.0
        stats["community_focus"] = 1.0
        stats["community_overlap"] = 0.0
    else:
        counts = {uid: len(comms) for uid, comms in community_map.items()}
        stats["community_count"] = pd.Series(counts).reindex(stats.index).fillna(1.0)
        stats["community_focus"] = (1.0 / stats["community_count"]).clip(lower=0.0, upper=1.0)
        stats["community_overlap"] = (stats["community_count"] - 1.0).clip(lower=0.0)

    if edge_survival is None:
        stats["edge_survival_ratio"] = 1.0
    else:
        stats["edge_survival_ratio"] = edge_survival.reindex(stats.index).fillna(1.0)

    return stats.reset_index().rename(columns={"index": "user_id"})


def _build_core_targets(stats: pd.DataFrame) -> pd.DataFrame:
    core = pd.DataFrame(index=stats["user_id"])
    core["reach_breadth"] = np.log1p(stats["view_count"] + 0.35 * stats["follower_count"])
    core["engagement_depth"] = np.log1p(stats["comment_count"] + 0.25 * stats["like_count"])
    core["reshare_amplification"] = np.log1p(stats["repost_count"]) / (np.log1p(stats["post_count"]) + 1.0)
    core["discussion_intensity"] = stats["comment_count"] / (stats["post_count"] + 1.0)
    core["temporal_velocity"] = (
        stats["like_count"] + stats["comment_count"] + stats["repost_count"]
    ) / stats["active_span_days"].clip(lower=1.0)
    core["source_authority"] = np.log1p(stats["follower_count"]) + 0.5 * stats["verified"] + 0.2 * stats["activity_score"]
    core["audience_diversity"] = stats["topic_entropy"] + 0.5 * stats["community_overlap"]
    core["sentiment_polarization"] = stats["sentiment_mean"].abs() + 0.5 * stats["sentiment_std"]
    core["cross_community_penetration"] = stats["community_count"] + 0.05 * stats["graph_degree"]
    core["structural_persistence"] = stats["active_days"] / stats["active_span_days"].clip(lower=1.0) + 0.02 * stats["graph_degree"]
    return _minmax_scale_frame(core).reset_index().rename(columns={"index": "user_id"})


def _build_local_explicit_features(stats: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, List[str]]:
    idx = stats["user_id"]
    features = pd.DataFrame(index=idx)
    if variant == "A":
        features["reach_proxy"] = np.log1p(stats["view_count"])
        features["like_proxy"] = np.log1p(stats["like_count"])
        features["comment_proxy"] = np.log1p(stats["comment_count"])
        features["repost_proxy"] = np.log1p(stats["repost_count"])
        features["post_freq"] = stats["post_freq"]
        features["originality_ratio"] = stats["originality_ratio"]
        features["sentiment_mean"] = stats["sentiment_mean"]
        features["sentiment_std"] = stats["sentiment_std"]
        features["hashtag_density"] = stats["tag_count"] / (stats["post_count"] + 1.0)
        features["mention_density"] = stats["at_count"] / (stats["post_count"] + 1.0)
        features["url_density"] = stats["url_count"] / (stats["post_count"] + 1.0)
        features["authority_proxy"] = np.log1p(stats["follower_count"])
        features["verified_flag"] = stats["verified"]
        features["graph_degree"] = stats["graph_degree"]
    elif variant == "B":
        features["reach_proxy"] = np.log1p(stats["view_count"])
        features["engagement_proxy"] = np.log1p(stats["like_count"] + stats["comment_count"])
        features["repost_proxy"] = np.log1p(stats["repost_count"])
        features["post_freq"] = stats["post_freq"]
        features["originality_ratio"] = stats["originality_ratio"]
        features["sentiment_mean"] = stats["sentiment_mean"]
        features["engagement_ratio"] = stats["engagement_ratio"]
        features["edge_survival_ratio"] = stats["edge_survival_ratio"]
        features["support_sparsity"] = 1.0 / (stats["graph_degree"] + 1.0)
        features["authority_proxy"] = np.log1p(stats["follower_count"])
        features["verified_flag"] = stats["verified"]
        features["graph_degree"] = stats["graph_degree"]
    else:
        features["score_proxy"] = np.log1p(stats["like_count"])
        features["agreement_ratio"] = (stats["like_count"] + 1.0) / (stats["comment_count"] + stats["repost_count"] + 2.0)
        features["discussion_depth_proxy"] = stats["comment_count"] / (stats["repost_count"] + 1.0)
        features["response_latency_inv"] = 1.0 / (stats["mean_gap_hours"] + 1.0)
        features["subreddit_focus"] = stats["community_focus"]
        features["thread_breadth"] = stats["community_count"]
        features["controversy"] = stats["sentiment_std"] + stats["comment_count"] / (stats["like_count"] + 1.0)
        features["post_freq"] = stats["post_freq"]
        features["originality_ratio"] = stats["originality_ratio"]
        features["sentiment_mean"] = stats["sentiment_mean"]
        features["authority_proxy"] = np.log1p(stats["follower_count"])
        features["verified_flag"] = stats["verified"]
        features["community_overlap"] = stats["community_overlap"]
    feature_names = list(features.columns)
    return _minmax_scale_frame(features).reset_index().rename(columns={"index": "user_id"}), feature_names


def _build_labels(stats: pd.DataFrame) -> pd.Series:
    raw = (
        0.45 * stats["repost_count"]
        + 0.30 * stats["comment_count"]
        + 0.20 * stats["like_count"]
        + 0.05 * stats["view_count"] / 50.0
    )
    return _normalized_label(pd.Series(raw, index=stats["user_id"]))


def _split_indices(n_items: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)
    n_train = max(1, int(0.6 * n_items))
    n_val = max(1, int(0.2 * n_items))
    train = perm[:n_train]
    val = perm[n_train:n_train + n_val]
    test = perm[n_train + n_val:]
    if len(test) == 0:
        test = perm[-max(1, n_items // 5):]
    return (
        torch.tensor(train, dtype=torch.long),
        torch.tensor(val, dtype=torch.long),
        torch.tensor(test, dtype=torch.long),
    )


def _node_feature_row(
    follower_count: float,
    following_count: float,
    verified: float,
    activity_score: float,
    post_freq: float,
    sentiment_mean: float,
    engagement_ratio: float,
    graph_degree: float,
    is_user: float,
    is_structural: float,
    topology_code: float,
) -> List[float]:
    return [
        float(np.log1p(follower_count)),
        float(np.log1p(following_count)),
        float(verified),
        float(activity_score),
        float(post_freq),
        float(sentiment_mean),
        float(engagement_ratio),
        float(np.log1p(graph_degree)),
        float(is_user),
        float(is_structural),
        float(topology_code),
        float(is_user * (0.5 + 0.5 * verified)),
    ]


class FederatedPlatformBuilder:
    def __init__(
        self,
        max_users_per_platform: int = 256,
        seed: int = 42,
    ) -> None:
        self.max_users_per_platform = max_users_per_platform
        self.seed = seed

    def build(self, use_real: bool = False) -> List[PlatformGraphData]:
        if use_real:
            users_df, posts_df, edges_df = RealDataLoader().load_all()
            return self._build_from_real(users_df, posts_df, edges_df)
        return self._build_from_mock()

    def _build_from_mock(self) -> List[PlatformGraphData]:
        gen_a = MockDataGenerator(n_users=max(220, self.max_users_per_platform + 40), n_posts=1800, seed=self.seed)
        gen_b = MockDataGenerator(n_users=max(260, self.max_users_per_platform + 60), n_posts=2200, seed=self.seed + 1)
        gen_c = MockDataGenerator(n_users=max(240, self.max_users_per_platform + 50), n_posts=2000, seed=self.seed + 2)

        users_a, posts_a, edges_a = gen_a.generate_all()
        posts_a = _ensure_post_columns(posts_a)
        posts_a["topic"] = "entertainment"
        posts_a["platform"] = "twitter"

        users_b, posts_b, edges_b = gen_b.generate_all()
        posts_b = _ensure_post_columns(posts_b)
        rng = np.random.default_rng(self.seed + 7)
        posts_b["topic"] = rng.choice(["sports", "animals"], size=len(posts_b), replace=True)
        posts_b["platform"] = "twitter"

        users_c, posts_c, edges_c = gen_c.generate_all()
        posts_c = _ensure_post_columns(posts_c)
        posts_c["topic"] = "politics"
        posts_c["platform"] = "reddit"

        platform_a = self._build_twitter_platform(
            platform_id="A",
            display_name="Platform A (Twitter Entertainment)",
            users_df=users_a,
            posts_df=posts_a,
            edges_df=edges_a,
            topics=["entertainment"],
            drop_edge_ratio=0.0,
            seed=self.seed,
        )
        platform_b = self._build_twitter_platform(
            platform_id="B",
            display_name="Platform B (Twitter Sports+Animals, 40% edge drop)",
            users_df=users_b,
            posts_df=posts_b,
            edges_df=edges_b,
            topics=["sports", "animals"],
            drop_edge_ratio=0.40,
            seed=self.seed + 11,
        )
        platform_c = self._build_reddit_proxy_platform(
            platform_id="C",
            display_name="Platform C (Reddit Politics Proxy)",
            users_df=users_c,
            posts_df=posts_c,
            edges_df=edges_c,
            seed=self.seed + 23,
        )
        return [platform_a, platform_b, platform_c]

    def _build_from_real(
        self,
        users_df: pd.DataFrame,
        posts_df: pd.DataFrame,
        edges_df: pd.DataFrame,
    ) -> List[PlatformGraphData]:
        platform_a = self._build_twitter_platform(
            platform_id="A",
            display_name="Platform A (Twitter Entertainment)",
            users_df=users_df,
            posts_df=posts_df,
            edges_df=edges_df,
            topics=["entertainment"],
            drop_edge_ratio=0.0,
            seed=self.seed,
        )
        platform_b = self._build_twitter_platform(
            platform_id="B",
            display_name="Platform B (Twitter Sports+Animals, 40% edge drop)",
            users_df=users_df,
            posts_df=posts_df,
            edges_df=edges_df,
            topics=["sports", "animals"],
            drop_edge_ratio=0.40,
            seed=self.seed + 11,
        )
        platform_c = self._build_reddit_proxy_platform(
            platform_id="C",
            display_name="Platform C (Reddit Politics Proxy)",
            users_df=users_df,
            posts_df=posts_df,
            edges_df=edges_df,
            seed=self.seed + 23,
        )
        return [platform_a, platform_b, platform_c]

    def _build_twitter_platform(
        self,
        platform_id: str,
        display_name: str,
        users_df: pd.DataFrame,
        posts_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        topics: Sequence[str],
        drop_edge_ratio: float,
        seed: int,
    ) -> PlatformGraphData:
        users = _ensure_user_columns(users_df)
        posts = _ensure_post_columns(posts_df)
        posts = posts[posts["topic"].isin(topics)].copy()
        candidate_edges = edges_df[
            edges_df["src"].isin(posts["user_id"]) & edges_df["dst"].isin(posts["user_id"])
        ].copy()
        chosen_users = _select_active_users(posts, self.max_users_per_platform, seed, edges_df=candidate_edges)
        posts = posts[posts["user_id"].isin(chosen_users)].copy()
        users = users[users["user_id"].isin(chosen_users)].drop_duplicates("user_id").copy()

        edges = edges_df[
            edges_df["src"].isin(chosen_users) & edges_df["dst"].isin(chosen_users)
        ].copy()
        if edges.empty and not candidate_edges.empty:
            graph = nx.Graph()
            graph.add_edges_from(candidate_edges[["src", "dst"]].itertuples(index=False, name=None))
            if graph.number_of_edges() > 0:
                largest_cc = max(nx.connected_components(graph), key=len)
                cc_posts = posts_df[
                    posts_df["topic"].isin(topics) & posts_df["user_id"].isin(largest_cc)
                ].copy()
                chosen_users = _select_active_users(cc_posts, self.max_users_per_platform, seed + 101, edges_df=candidate_edges)
                posts = cc_posts[cc_posts["user_id"].isin(chosen_users)].copy()
                users = _ensure_user_columns(users_df)
                users = users[users["user_id"].isin(chosen_users)].drop_duplicates("user_id").copy()
                edges = edges_df[
                    edges_df["src"].isin(chosen_users) & edges_df["dst"].isin(chosen_users)
                ].copy()
        edges["weight"] = edges.get("weight", 1.0).astype(float)
        original_degree = pd.concat([edges["src"], edges["dst"]]).value_counts()
        if drop_edge_ratio > 0 and not edges.empty:
            rng = np.random.default_rng(seed + 3)
            keep_mask = rng.random(len(edges)) > drop_edge_ratio
            edges = edges.loc[keep_mask].reset_index(drop=True)
        new_degree = pd.concat([edges["src"], edges["dst"]]).value_counts()
        edge_survival = (new_degree / original_degree.reindex(new_degree.index).clip(lower=1.0)).clip(upper=1.0)

        stats = _aggregate_user_statistics(users, posts, edges, edge_survival=edge_survival)
        core_targets = _build_core_targets(stats)
        local_features, local_feature_names = _build_local_explicit_features(stats, variant=platform_id)
        labels = _build_labels(stats)

        node_ids = users["user_id"].tolist()
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        if edges.empty:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_array = np.vstack(
                [
                    edges["src"].map(node_id_to_idx).to_numpy(dtype=np.int64),
                    edges["dst"].map(node_id_to_idx).to_numpy(dtype=np.int64),
                ]
            )
            edge_index = torch.from_numpy(edge_array)

        stats_map = stats.set_index("user_id")
        node_features = [
            _node_feature_row(
                follower_count=stats_map.loc[uid, "follower_count"],
                following_count=stats_map.loc[uid, "following_count"],
                verified=stats_map.loc[uid, "verified"],
                activity_score=stats_map.loc[uid, "activity_score"],
                post_freq=stats_map.loc[uid, "post_freq"],
                sentiment_mean=stats_map.loc[uid, "sentiment_mean"],
                engagement_ratio=stats_map.loc[uid, "engagement_ratio"],
                graph_degree=stats_map.loc[uid, "graph_degree"],
                is_user=1.0,
                is_structural=0.0,
                topology_code=0.0 if platform_id == "A" else 1.0,
            )
            for uid in node_ids
        ]
        node_features = torch.tensor(_minmax_scale_frame(pd.DataFrame(node_features)).to_numpy(), dtype=torch.float32)

        feature_map = local_features.set_index("user_id")
        core_map = core_targets.set_index("user_id")
        label_map = labels.reindex(node_ids).fillna(0.0)
        explicit_local = torch.tensor(feature_map.reindex(node_ids).fillna(0.0).to_numpy(), dtype=torch.float32)
        core_tensor = torch.tensor(core_map.reindex(node_ids).fillna(0.0).to_numpy(), dtype=torch.float32)
        label_tensor = torch.tensor(label_map.to_numpy(dtype=float), dtype=torch.float32)
        target_node_indices = torch.arange(len(node_ids), dtype=torch.long)
        train_idx, val_idx, test_idx = _split_indices(len(node_ids), seed)

        graph = nx.Graph()
        graph.add_nodes_from(node_ids)
        graph.add_edges_from(edges[["src", "dst"]].itertuples(index=False, name=None))

        metadata = {
            "topics": list(topics),
            "drop_edge_ratio": float(drop_edge_ratio),
            "num_users": int(len(users)),
            "num_posts": int(len(posts)),
            "num_edges": int(len(edges)),
            "explicit_dim": int(explicit_local.size(1)),
            "topology": "directed",
        }
        return PlatformGraphData(
            platform_id=platform_id,
            display_name=display_name,
            topology="directed",
            users_df=users.reset_index(drop=True),
            posts_df=posts.reset_index(drop=True),
            edges_df=edges.reset_index(drop=True),
            node_ids=node_ids,
            node_id_to_idx=node_id_to_idx,
            edge_index=edge_index,
            node_features=node_features,
            explicit_local=explicit_local,
            core_targets=core_tensor,
            labels=label_tensor,
            target_node_indices=target_node_indices,
            target_node_ids=node_ids,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            local_feature_names=local_feature_names,
            core_feature_names=CORE_SEMANTIC_ANCHORS,
            metadata=metadata,
            nx_graph=graph,
        )

    def _build_reddit_proxy_platform(
        self,
        platform_id: str,
        display_name: str,
        users_df: pd.DataFrame,
        posts_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        seed: int,
    ) -> PlatformGraphData:
        users = _ensure_user_columns(users_df)
        posts = _ensure_post_columns(posts_df)
        chosen_users = _select_active_users(posts, self.max_users_per_platform, seed, edges_df=edges_df)
        posts = posts[posts["user_id"].isin(chosen_users)].copy()
        users = users[users["user_id"].isin(chosen_users)].drop_duplicates("user_id").copy()
        posts["platform"] = "reddit"
        posts["topic"] = "politics"

        subreddit_names = ["politics", "elections", "policy", "worldnews", "governance"]
        rng = np.random.default_rng(seed)
        community_map: Dict[str, List[str]] = {}
        subreddit_edges: List[Dict[str, Any]] = []
        for uid, user_posts in posts.groupby("user_id"):
            comm_count = 1 + int(rng.random() > 0.72)
            communities = rng.choice(subreddit_names, size=comm_count, replace=False).tolist()
            community_map[uid] = communities
            weight_base = float(
                user_posts["comment_count"].sum()
                + user_posts["repost_count"].sum()
                + 0.25 * user_posts["like_count"].sum()
            )
            for community in communities:
                subreddit_edges.append(
                    {
                        "src": uid,
                        "dst": f"subreddit/{community}",
                        "weight": max(1.0, np.log1p(weight_base + 1.0)),
                        "edge_type": "participates_in",
                    }
                )

        social_edges = edges_df[
            edges_df["src"].isin(chosen_users) & edges_df["dst"].isin(chosen_users)
        ].copy()
        if not social_edges.empty:
            social_edges = social_edges.sample(
                n=min(len(social_edges), max(1, int(0.18 * len(social_edges)))),
                random_state=seed,
            )
            social_edges["edge_type"] = "reply_tie"

        subreddit_df = pd.DataFrame(subreddit_edges)
        edges = pd.concat([social_edges, subreddit_df], ignore_index=True)
        edges["weight"] = edges.get("weight", 1.0).astype(float)

        stats = _aggregate_user_statistics(users, posts, edges, community_map=community_map)
        core_targets = _build_core_targets(stats)
        local_features, local_feature_names = _build_local_explicit_features(stats, variant="C")
        labels = _build_labels(stats)

        user_node_ids = users["user_id"].tolist()
        structural_nodes = sorted({row["dst"] for row in subreddit_edges})
        node_ids = user_node_ids + structural_nodes
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        edge_index = (
            torch.from_numpy(
                np.vstack(
                    [
                        edges["src"].map(node_id_to_idx).to_numpy(dtype=np.int64),
                        edges["dst"].map(node_id_to_idx).to_numpy(dtype=np.int64),
                    ]
                )
            )
            if not edges.empty
            else torch.empty((2, 0), dtype=torch.long)
        )

        stats_map = stats.set_index("user_id")
        user_rows = [
            _node_feature_row(
                follower_count=stats_map.loc[uid, "follower_count"],
                following_count=stats_map.loc[uid, "following_count"],
                verified=stats_map.loc[uid, "verified"],
                activity_score=stats_map.loc[uid, "activity_score"],
                post_freq=stats_map.loc[uid, "post_freq"],
                sentiment_mean=stats_map.loc[uid, "sentiment_mean"],
                engagement_ratio=stats_map.loc[uid, "engagement_ratio"],
                graph_degree=stats_map.loc[uid, "graph_degree"],
                is_user=1.0,
                is_structural=0.0,
                topology_code=2.0,
            )
            for uid in user_node_ids
        ]
        community_strength = (
            pd.DataFrame(subreddit_edges).groupby("dst")["weight"].mean().reindex(structural_nodes).fillna(1.0)
            if structural_nodes else pd.Series(dtype=float)
        )
        structural_rows = [
            _node_feature_row(
                follower_count=0.0,
                following_count=0.0,
                verified=0.0,
                activity_score=community_strength.loc[node_id] / (community_strength.max() + 1e-6),
                post_freq=community_strength.loc[node_id] / (community_strength.max() + 1e-6),
                sentiment_mean=0.0,
                engagement_ratio=community_strength.loc[node_id] / (community_strength.max() + 1e-6),
                graph_degree=float((edges["src"].eq(node_id) | edges["dst"].eq(node_id)).sum()),
                is_user=0.0,
                is_structural=1.0,
                topology_code=2.0,
            )
            for node_id in structural_nodes
        ]
        all_rows = user_rows + structural_rows
        node_features = torch.tensor(_minmax_scale_frame(pd.DataFrame(all_rows)).to_numpy(), dtype=torch.float32)

        feature_map = local_features.set_index("user_id")
        core_map = core_targets.set_index("user_id")
        label_map = labels.reindex(user_node_ids).fillna(0.0)
        explicit_local = torch.tensor(feature_map.reindex(user_node_ids).fillna(0.0).to_numpy(), dtype=torch.float32)
        core_tensor = torch.tensor(core_map.reindex(user_node_ids).fillna(0.0).to_numpy(), dtype=torch.float32)
        label_tensor = torch.tensor(label_map.to_numpy(dtype=float), dtype=torch.float32)
        target_node_indices = torch.arange(len(user_node_ids), dtype=torch.long)
        train_idx, val_idx, test_idx = _split_indices(len(user_node_ids), seed)

        graph = nx.Graph()
        graph.add_nodes_from(node_ids)
        graph.add_edges_from(edges[["src", "dst"]].itertuples(index=False, name=None))

        metadata = {
            "topics": ["politics"],
            "drop_edge_ratio": 0.0,
            "num_users": int(len(users)),
            "num_posts": int(len(posts)),
            "num_edges": int(len(edges)),
            "num_structural_nodes": int(len(structural_nodes)),
            "explicit_dim": int(explicit_local.size(1)),
            "topology": "bipartite",
        }
        return PlatformGraphData(
            platform_id=platform_id,
            display_name=display_name,
            topology="bipartite",
            users_df=users.reset_index(drop=True),
            posts_df=posts.reset_index(drop=True),
            edges_df=edges.reset_index(drop=True),
            node_ids=node_ids,
            node_id_to_idx=node_id_to_idx,
            edge_index=edge_index,
            node_features=node_features,
            explicit_local=explicit_local,
            core_targets=core_tensor,
            labels=label_tensor,
            target_node_indices=target_node_indices,
            target_node_ids=user_node_ids,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            local_feature_names=local_feature_names,
            core_feature_names=CORE_SEMANTIC_ANCHORS,
            metadata=metadata,
            nx_graph=graph,
        )


__all__ = ["CORE_SEMANTIC_ANCHORS", "PlatformGraphData", "FederatedPlatformBuilder"]
