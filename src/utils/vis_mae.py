import torch
import matplotlib.pyplot as plt

import math
from typing import Final, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from av2.datasets.motion_forecasting.data_schema import ObjectType
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

_DRIVABLE_AREA_COLOR: Final[str] = "#7A7A7A"
_LANE_SEGMENT_COLOR: Final[str] = "#E0E0E0"

_FOCAL_AGENT_COLOR: Final[str] = "#ECA25B"

_CATEGORY_TO_COLOR: Final[dict] = {
    "vehicle": "#00a4ef",
    "pedestrian": "#d34836",
    "cyclist": "#007672",
    "unknown": "#000000",
    "truck": "#9500ff",
    "large_vehicle": "#0df2c8",
    "vehicular_trailer": "#D500F9",
    "school_bus": "#ffb300",
    "truck_cab": "#D500F9",
}

_PlotBounds = Tuple[float, float, float, float]


def plot_reconstruction(data, out, save_path="./test.pdf"):
    row, col = 3, 1
    fig, ax = plt.subplots(3, 1, figsize=(27 / 3, 24 / 3))
    fig.subplots_adjust(hspace=-0.08)

    fc = ["#EEEEEE", "#EEEEEE", "#EEEEEE"]
    for i in range(row * col):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].spines[["right", "top", "left", "bottom"]].set_visible(False)

    lane_key_padding_mask = data["lane_key_padding_mask"][0].cpu()
    lanes = data["lane_positions"][0].cpu()[~lane_key_padding_mask]
    lane_hat = out["lane_hat"][0].cpu().detach()
    lane_keep_ids = out["lane_keep_ids"][0].cpu()
    lane_hat += data["lane_centers"][0].cpu().unsqueeze(-2)
    lane_padding_mask = data["lane_padding_mask"][0].cpu()[~lane_key_padding_mask]

    lane_pred_mask = ~lane_key_padding_mask
    lane_pred_mask[lane_keep_ids] = False
    lane_keep_ids = out["lane_keep_ids"][0].cpu()
    lane_hat = lane_hat[lane_pred_mask]

    # agents history plot
    key_padding_mask = data["x_key_padding_mask"][0].cpu()
    x = data["x"][0].cpu()[~key_padding_mask]
    x_centers = data["x_centers"][0].cpu()[~key_padding_mask]
    x_padding_mask = data["x_padding_mask"][0].cpu()[~key_padding_mask]
    x_attr = data["x_attr"][0].cpu()[~key_padding_mask]

    # agent future plot
    y = data["y"][0]
    y_padding_mask = data["x_padding_mask"][0, :, 50:]
    y += x_centers[:, None, :]

    hist_keep_ids = out["hist_keep_ids"][0].cpu()

    HISTORY_COLOR = "dodgerblue"
    PRED_HISTORY = "royalblue"

    FUTURE_color = "orange"
    PRED_future = "coral"
    ####################
    idx = 0
    plot_lanes(ax[idx], lanes, lane_padding_mask, alpha=0.3)
    plot_history(
        ax[idx],
        x,
        x_centers,
        color=HISTORY_COLOR,
        x_padding_mask=x_padding_mask,
        linewidth=2,
    )
    plot_future(ax[idx], y, y_padding_mask, color=FUTURE_color, alpha=1.0, linewidth=2)
    plot_centers(ax[idx], x_centers, color="#82B0D2", alpha=1.0, s=30)

    ####################
    # Masked Agents
    ####################
    idx = 1
    plot_lanes(ax[1], lanes, lane_padding_mask, alpha=0.3, visible_index=lane_keep_ids)
    plot_centers(ax[idx], x_centers, color="#82B0D2", alpha=1.0, s=30)
    plot_history(
        ax[1],
        x,
        x_centers,
        color=HISTORY_COLOR,
        x_padding_mask=x_padding_mask,
        alpha=1.0,
        visible_index=hist_keep_ids,
        linewidth=2,
    )
    plot_future(
        ax[1],
        y,
        y_padding_mask,
        color=FUTURE_color,
        alpha=1.0,
        invisible_index=hist_keep_ids,
        linewidth=2,
    )

    hist_keep_ids = out["hist_keep_ids"][0].cpu()
    x_hats = out["x_hat"][0].cpu().detach()[~key_padding_mask]
    x_hats += x_centers[:, None, :]
    x_keep_mask = torch.zeros(len(x_hats), dtype=torch.bool)
    x_keep_mask[hist_keep_ids] = True
    x_hats = x_hats[~x_keep_mask]

    y_padding_mask = x_padding_mask[:, 50:]
    fut_keep_ids = out["fut_keep_ids"][0].cpu()
    y_hats = out["y_hat"][0, 0].cpu().detach()
    y_hats += x_centers[:, None, :]
    fut_keep_mask = torch.zeros(len(y_hats), dtype=torch.bool)
    fut_keep_mask[fut_keep_ids] = True
    y_hats = y_hats[(~fut_keep_mask)]

    ####################
    # reconstructed agents
    ####################
    idx = -1
    plot_history(
        ax[idx],
        x,
        x_centers,
        color=HISTORY_COLOR,
        x_padding_mask=x_padding_mask,
        alpha=1.0,
        visible_index=hist_keep_ids,
        linewidth=2,
    )
    for x_hat in x_hats:
        ax[idx].plot(
            x_hat[-25:, 0],
            x_hat[-25:, 1],
            color=PRED_HISTORY,
            alpha=1.0,
            linewidth=2,
            label="reconstructed history",
        )
    plot_future(
        ax[idx],
        y,
        y_padding_mask,
        color=FUTURE_color,
        alpha=1.0,
        invisible_index=hist_keep_ids,
        linewidth=2,
    )
    for y_hat in y_hats:
        ax[idx].plot(
            y_hat[:30, 0],
            y_hat[:30, 1],
            color=PRED_future,
            alpha=1.0,
            linewidth=2,
            label="reconstructed future",
        )
    plot_centers(ax[idx], x_centers, color="#82B0D2", alpha=1.0, s=30)
    plot_lanes(
        ax[idx],
        lanes,
        lane_padding_mask,
        visible_index=lane_keep_ids,
        alpha=0.2,
        label="lanes",
    )
    plot_lanes(
        ax[idx],
        lane_hat,
        color="darkseagreen",
        alpha=0.8,
        zorder=1,
        label="reconstructed lanes",
    )

    plt.rcParams["font.family"] = ["Serif"]
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    mark = ["(a)", "(b)", "(c)"]
    for i in range(row * col):
        ax[i].set_xlim(-120, 150)
        ax[i].set_ylim(-40, 40)
        ax[i].text(-115, 20, mark[i], fontsize=25)
        ax[i].set_aspect("equal")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        ncol=3,
        bbox_to_anchor=(0.97, 0),
        fontsize=16,
        frameon=False,
    )

    plt.tight_layout()


def plot_lanes(
    ax,
    lanes,
    padding_mask=None,
    linewidth=2,
    color="gray",
    alpha=1.0,
    visible_index=None,
    zorder=10,
    label="",
):
    for i, lane in enumerate(lanes):
        if visible_index is not None and i not in visible_index:
            continue
        if padding_mask is not None:
            mask = padding_mask[i]
            lane = lane[~mask]
        ax.plot(
            lane[:, 0],
            lane[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
            label=label,
        )


def plot_centers(ax, center, color="blue", alpha=1.0, linewidth=1.0, zorder=11, s=20):
    ax.scatter(center[:, 0], center[:, 1], color=color, s=s, alpha=alpha, zorder=zorder)


def plot_history(
    ax,
    x,
    x_centers,
    x_padding_mask,
    color="blue",
    alpha=1.0,
    linewidth=1.0,
    zorder=11,
    visible_index=None,
):
    for i, (actor, mask, ctr) in enumerate(zip(x, x_padding_mask, x_centers)):
        if visible_index is not None and i not in visible_index:
            continue
        valid_mask = ~mask[:50]
        xy = actor[valid_mask]
        xy = torch.cumsum(-torch.flip(xy, dims=[0]), dim=0) + ctr
        xy = torch.cat([ctr[None, :], xy], dim=0)
        xy = xy[:25, :]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
            label="history",
        )


def plot_future(
    ax,
    y,
    y_padding_mask,
    color="orange",
    alpha=1.0,
    linewidth=1.0,
    zorder=11,
    invisible_index=None,
):
    for i, (fut, mask) in enumerate(zip(y, y_padding_mask)):
        if invisible_index is not None and i in invisible_index:
            continue
        fut = fut[~mask]
        fut = fut[:30]
        ax.plot(
            fut[:, 0],
            fut[:, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
            label="future",
        )


def get_polyline_arc_length(xy: np.ndarray) -> np.ndarray:
    """Get the arc length of each point in a polyline"""
    diff = xy[1:] - xy[:-1]
    displacement = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    arc_length = np.cumsum(displacement)
    return np.concatenate((np.zeros(1), arc_length), axis=0)


def interpolate_lane(xy: np.ndarray, arc_length: np.ndarray, steps: np.ndarray):
    xy_inter = np.empty((steps.shape[0], 2), dtype=xy.dtype)
    xy_inter[:, 0] = np.interp(steps, xp=arc_length, fp=xy[:, 0])
    xy_inter[:, 1] = np.interp(steps, xp=arc_length, fp=xy[:, 1])
    return xy_inter


def interpolate_centerline(xy: np.ndarray, n_points: int):
    arc_length = get_polyline_arc_length(xy)
    steps = np.linspace(0, arc_length[-1], n_points)
    xy_inter = np.empty((steps.shape[0], 2), dtype=xy.dtype)
    xy_inter[:, 0] = np.interp(steps, xp=arc_length, fp=xy[:, 0])
    xy_inter[:, 1] = np.interp(steps, xp=arc_length, fp=xy[:, 1])
    return xy_inter
