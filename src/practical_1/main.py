from .task_1 import *
from .task_2 import *
from .task_3 import *


def main():
    print("Starting application")
    run_task_1()
    run_task_2()
    run_task_3()
    visualize()


def visualize():
    pitches = np.linspace(86, 94, 161)
    matrices = [
        Cz(np.deg2rad(30)) @ Cy(np.deg2rad(y)) @ Cx(np.deg2rad(10)) for y in pitches
    ]
    angles = []
    for C in matrices:
        try:
            angles.append(np.rad2deg(euler_zyx(C)))
        except ValueError:
            angles.append([np.nan] * 3)
    steps = [
        np.rad2deg(np.linalg.norm(orientation_error_world(A, B)))
        for A, B in zip(matrices[:-1], matrices[1:])
    ]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, name in enumerate(["yaw z", "pitch y", "roll x"]):
        axs[0].plot(pitches, np.array(angles)[:, i], label=name)
    axs[0].axvline(90, color="gray", ls=":")
    axs[0].set(
        xlabel="Input pitch [deg]",
        ylabel="Extracted angle [deg]",
        title="Euler branch at gimbal lock",
    )
    axs[0].legend()
    axs[1].plot(pitches[1:], steps)
    axs[1].set(
        xlabel="Input pitch [deg]",
        ylabel="Angle between adjacent poses [deg]",
        title="The orientation stays continuous",
        ylim=(0, 0.1),
    )
    q0 = quat_from_phi(np.deg2rad([0, 0, 170]))
    q1 = quat_from_phi(np.deg2rad([0, 0, -170]))
    ss = np.linspace(0, 1, 101)
    short = [C_from_quat(slerp(q0, q1, s)) @ [1, 0, 0] for s in ss]
    long = [Cz(np.deg2rad(170 * (1 - s) - 170 * s)) @ [1, 0, 0] for s in ss]
    axs[2].plot(
        np.array(long)[:, 0], np.array(long)[:, 1], "--", label="Linear yaw values"
    )
    axs[2].plot(
        np.array(short)[:, 0], np.array(short)[:, 1], lw=3, label="Shortest Slerp"
    )
    axs[2].set(
        xlabel="World x",
        ylabel="World y",
        title="Path of the rotated x-axis",
        aspect="equal",
    )
    axs[2].legend()
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("t1_rotations.png", dpi=150)
    plt.show()
    print("wrote t1_rotations.png")
    print("Largest adjacent orientation change:", max(steps), "deg")
    print(
        "World-axis correction from +170 to -170 deg:",
        np.rad2deg(orientation_error_world(C_from_quat(q0), C_from_quat(q1))),
    )


if __name__ == "__main__":
    main()
