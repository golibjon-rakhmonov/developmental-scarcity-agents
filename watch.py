"""Watch one agent live in a world.

    python watch.py runs/scarce_s0_after_childhood.pt                 the adult world
    python watch.py runs/scarce_s0_after_childhood.pt --world scarce  back in a scarce childhood world
    python watch.py runs/rich_s0_end.pt --speed 30
    python watch.py runs/rich_s0_end.pt --save rich_s0.gif --steps 600

Green squares are food (brighter means more), blue is water, the pale blue band
is cold ground, brown squares are shelters, red is the agent and the frame
around it is what it can see. The picture gets darker at night. On the right
are the agent's body and what it just did. The agent keeps the network it was
saved with; nothing is learned while you watch.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from agent import Net
from world import ADULT, CARRY_MAX, N_ACTIONS, OBS_DIM, R, RICH, SCARCE, T_COLLAPSE, T_SET, Worlds

WORLDS = {"adult": ADULT, "scarce": SCARCE, "rich": RICH}
ACTIONS = ["up", "down", "left", "right", "rest", "eat", "drink", "pick up"]


def picture(w):
    s = w.cfg.size
    img = np.ones((s, s, 3)) * [0.93, 0.89, 0.80]                 # warm ground
    img[w.cold[0]] = [0.80, 0.88, 0.96]                          # cold ground
    food = w.food[0][..., None]
    img = np.where(w.patch[0][..., None], (1 - food) * img + food * np.array([0.20, 0.65, 0.25]), img)
    img[w.shelter[0]] = [0.55, 0.40, 0.25]
    img[w.water[0]] = [0.15, 0.45, 0.85]
    if w.night()[0]:
        img = img * 0.55
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--world", choices=WORLDS, default="adult")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--speed", type=float, default=15, help="steps per second")
    p.add_argument("--save", help="write a gif instead of opening a window")
    p.add_argument("--steps", type=int, default=500, help="length of the gif")
    args = p.parse_args()

    net = Net(OBS_DIM, N_ACTIONS)
    net.load_state_dict(torch.load(args.model))
    net.eval()
    w = Worlds(WORLDS[args.world], 1, np.random.default_rng(args.seed))
    torch.manual_seed(args.seed)
    obs = w.observe()
    state = {"t": 0, "collapses": 0, "last": "", "meals": 0, "drinks": 0}

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.03, 0.05, 0.6, 0.9])
    side = fig.add_axes([0.68, 0.35, 0.28, 0.55])
    text_ax = fig.add_axes([0.68, 0.05, 0.28, 0.25]); text_ax.axis("off")
    im = ax.imshow(picture(w), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    agent_dot, = ax.plot([], [], "o", color="#d62728", markersize=14)
    view = Rectangle((0, 0), 2 * R + 1, 2 * R + 1, fill=False, edgecolor="#d62728", lw=1.5, ls="--")
    ax.add_patch(view)
    labels = ["energy", "water", "temperature", "food carried"]
    bars = side.barh(labels, [0] * 4, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#8c564b"])
    side.set_xlim(0, 1); side.invert_yaxis(); side.set_title("body")
    side.plot([T_SET, T_SET], [1.6, 2.4], color="black", lw=1, ls=":")   # the temperature the body aims for
    info = text_ax.text(0, 1, "", va="top", family="monospace", fontsize=10)

    def update(_):
        nonlocal obs
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(obs))
            a = torch.distributions.Categorical(logits=logits).sample().numpy()
        obs, _, collapsed, step_info = w.step(a)
        state["t"] += 1
        state["last"] = ACTIONS[int(a[0])]
        state["meals"] += int(step_info["ate"][0])
        state["drinks"] += int(step_info["drank"][0])
        if collapsed[0]:
            state["collapses"] += 1
            cause = "hunger" if step_info["starved"][0] else "thirst" if step_info["thirst"][0] else "cold"
            state["last"] = f"COLLAPSED ({cause})"
        im.set_data(picture(w))
        y, x = w.pos[0]
        agent_dot.set_data([x], [y])
        view.set_xy((x - R - 0.5, y - R - 0.5))
        values = [w.energy[0], w.water_level[0], w.temp[0], w.carry[0] / CARRY_MAX]
        for b, v in zip(bars, values):
            b.set_width(float(v))
        bars[2].set_color("#1f77b4" if w.temp[0] < T_COLLAPSE + 0.1 else "#ff7f0e")
        info.set_text(
            f"step        {state['t']}\n"
            f"time        {'night' if w.night()[0] else 'day'}\n"
            f"drought     {'yes' if w.drought()[0] else 'no'}\n"
            f"did         {state['last']}\n"
            f"meals       {state['meals']}\n"
            f"drinks      {state['drinks']}\n"
            f"collapses   {state['collapses']}")
        return im, agent_dot, view, info, *bars

    title = f"{args.model}  ·  {args.world} world"
    fig.suptitle(title, fontsize=10)
    if args.save:
        anim = FuncAnimation(fig, update, frames=args.steps, blit=False)
        anim.save(args.save, writer=PillowWriter(fps=args.speed))
        print("saved", args.save)
    else:
        anim = FuncAnimation(fig, update, interval=1000 / args.speed, blit=False, cache_frame_data=False)
        plt.show()


if __name__ == "__main__":
    main()
