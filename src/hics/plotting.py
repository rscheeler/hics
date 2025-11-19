import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import animation

from . import ureg
from .geo.geoplotting import *
from .hics import HCS

# For animations
plt.rcParams["animation.html"] = "jshtml"


def viewcs_pyvista(
    cs: HCS,
    reference_cs: HCS | None = None,
    plotter: pv.Plotter | None = None,
    vector_length: float = 1,
    units: str = "m",
    animate: bool = False,
) -> pv.Plotter:
    """
    View unit x, y, and z vectors in the defined coordinate system using PyVista.
    """
    # Create plotter if not specified
    if plotter is None:
        plotter = pv.Plotter()

    # Default is cs
    if reference_cs is None:
        reference_cs = cs

    # --- Calculation logic remains mostly the same ---
    # Get center position
    p0 = reference_cs.relative_position(cs)

    # Get X,Y,Z unit vectors (directions)
    xvect = (
        reference_cs.relative_position(HCS([vector_length, 0, 0] * ureg(units), reference=cs)) - p0
    )
    yvect = (
        reference_cs.relative_position(HCS([0, vector_length, 0] * ureg(units), reference=cs)) - p0
    )
    zvect = (
        reference_cs.relative_position(HCS([0, 0, vector_length] * ureg(units), reference=cs)) - p0
    )

    # Convert to specified units and extract magnitudes
    p0_mag = p0.data.to(units).magnitude
    xvect_mag = xvect.data.to(units).magnitude
    yvect_mag = yvect.data.to(units).magnitude
    zvect_mag = zvect.data.to(units).magnitude

    # Check for array of points (animated/multiple CS)
    is_multi_frame = len(p0.shape) > 1

    if is_multi_frame:
        p0r = p0_mag.reshape(-1, 3)
        xvectr = xvect_mag.reshape(-1, 3)
        yvectr = yvect_mag.reshape(-1, 3)
        zvectr = zvect_mag.reshape(-1, 3)
        num_frames = p0r.shape[0]

        if animate:
            # PyVista animation logic using update_scalars/callback

            # Initial Plotting: PyVista's add_arrows requires points and vectors
            # The structure for pyvista.add_arrows is:
            # 1. points: (N, 3) array of starting points for the vectors
            # 2. vectors: (N, 3) array of vector directions/magnitudes

            # PyVista add_arrows can plot all vectors at once, but for animation
            # where you want to *update* the position, it's often easier to plot
            # one frame and use a callback to update it, or use `add_mesh` with
            # the actor returned.

            # Plot the first frame and store actors/meshes for easy updating.
            # We'll plot one vector set per arrow actor for easier color control.

            # X-vector (Red)
            x_arrows = plotter.add_arrows(
                p0r[0, :].reshape(1, 3),  # Start point
                xvectr[0, :].reshape(1, 3),  # Direction vector
                color="red",
                line_width=3,
                name="x_arrows",  # Name the actor for easy reference/removal
            )
            # Y-vector (Green)
            y_arrows = plotter.add_arrows(
                p0r[0, :].reshape(1, 3),
                yvectr[0, :].reshape(1, 3),
                color="green",
                line_width=3,
                name="y_arrows",
            )
            # Z-vector (Blue)
            z_arrows = plotter.add_arrows(
                p0r[0, :].reshape(1, 3),
                zvectr[0, :].reshape(1, 3),
                color="blue",
                line_width=3,
                name="z_arrows",
            )

            # Set the camera
            plotter.view_isometric()

            def moving_cs_pyvista(frame_i):
                """Callback function to update the arrow positions for each frame."""
                # Fetch actors by name (or from the plotter.actors dictionary)
                x_actor = plotter.actors.get("x_arrows")
                y_actor = plotter.actors.get("y_arrows")
                z_actor = plotter.actors.get("z_arrows")

                if x_actor:
                    # Update X-vector
                    new_x_mesh = pv.Arrow(
                        start=p0r[frame_i, :],
                        direction=xvectr[frame_i, :],
                        tip_length=0.2,  # Default PyVista arrow attributes
                        tip_radius=0.1,
                        shaft_radius=0.05,
                    )
                    x_actor.SetMapper(new_x_mesh.mapper)

                if y_actor:
                    # Update Y-vector
                    new_y_mesh = pv.Arrow(
                        start=p0r[frame_i, :],
                        direction=yvectr[frame_i, :],
                        tip_length=0.2,
                        tip_radius=0.1,
                        shaft_radius=0.05,
                    )
                    y_actor.SetMapper(new_y_mesh.mapper)

                if z_actor:
                    # Update Z-vector
                    new_z_mesh = pv.Arrow(
                        start=p0r[frame_i, :],
                        direction=zvectr[frame_i, :],
                        tip_length=0.2,
                        tip_radius=0.1,
                        shaft_radius=0.05,
                    )
                    z_actor.SetMapper(new_z_mesh.mapper)

                # Update title
                plotter.add_text(
                    f"{p0.dims[0]}={p0.coords[p0.dims[0]][frame_i].data}",
                    name="frame_title",
                    position="upper_right",
                )

            # Start the animation loop
            # This is the PyVista equivalent of matplotlib's FuncAnimation
            plotter.add_callback(
                lambda frame: moving_cs_pyvista(frame % num_frames),
                interval=10,  # Interval in ms
            )
            plotter.open_gif("cs_animation.gif")  # Optional: save to GIF

        else:
            # Plot all coordinate systems at once (Non-Animated)

            # PyVista add_arrows takes arrays of points and vectors, so we can plot
            # all x, y, and z vectors for all time steps efficiently in one go.

            # X-vectors (Red)
            plotter.add_arrows(
                p0r,  # All start points
                xvectr,  # All x-directions
                color="red",
                line_width=3,
                label="X-axis",
            )
            # Y-vectors (Green)
            plotter.add_arrows(p0r, yvectr, color="green", line_width=3, label="Y-axis")
            # Z-vectors (Blue)
            plotter.add_arrows(p0r, zvectr, color="blue", line_width=3, label="Z-axis")

    else:
        # Single Coordinate System (Simple Plot)

        # X-vector (Red)
        plotter.add_arrows(
            p0_mag.reshape(1, 3),
            xvect_mag.reshape(1, 3),
            color="red",
            line_width=3,
        )
        # Y-vector (Green)
        plotter.add_arrows(
            p0_mag.reshape(1, 3),
            yvect_mag.reshape(1, 3),
            color="green",
            line_width=3,
        )
        # Z-vector (Blue)
        plotter.add_arrows(
            p0_mag.reshape(1, 3),
            zvect_mag.reshape(1, 3),
            color="blue",
            line_width=3,
        )

    # --- PyVista Plotter Setup (Replaces Matplotlib axis labels/extents) ---

    # Axis labels: PyVista uses a camera/scene system. We'll use a bounding box
    # and scalar bar annotations to label, or simply rely on the default axes
    # annotations (which are on by default).

    # Set the overall title
    plotter.add_title(f"Coordinate System View (Units: {units})")

    # Set the camera
    plotter.view_isometric()

    # PyVista returns the plotter object
    return plotter


def viewcs(
    cs: HCS,
    reference_cs: HCS | None = None,
    ax: plt.Axes | None = None,
    vector_length: float = 1,
    units: str = "m",
    animate=False,
) -> plt.Axes:
    """
    View unit x, y, and z vectors in the defined coordinate system
    """
    # Create axes if not specified
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        ax.set_box_aspect(aspect=(1, 1, 1))

    # Default is cs (though not very interesting)
    if reference_cs is None:
        reference_cs = cs

    # Get center position
    p0 = reference_cs.relative_position(cs)

    # Get X,Y,Z unit vectors
    xvect = (
        reference_cs.relative_position(HCS([vector_length, 0, 0] * ureg(units), reference=cs)) - p0
    )
    yvect = (
        reference_cs.relative_position(HCS([0, vector_length, 0] * ureg(units), reference=cs)) - p0
    )
    zvect = (
        reference_cs.relative_position(HCS([0, 0, vector_length] * ureg(units), reference=cs)) - p0
    )

    # Convert to specified units
    p0.data = p0.data.to(units)
    xvect.data = xvect.data.to(units)
    yvect.data = yvect.data.to(units)
    zvect.data = zvect.data.to(units)

    if len(p0.shape) > 1:
        p0r = p0.data.reshape(-1, 3)
        xvectr = xvect.data.reshape(-1, 3)
        yvectr = yvect.data.reshape(-1, 3)
        zvectr = zvect.data.reshape(-1, 3)

        if animate:
            # Plot first item so it can be deleted
            # Plot x-vector
            c1 = ax.quiver(
                *p0r[0, ...].magnitude,  # <-- starting point of vector
                *xvectr[0, ...].magnitude,  # <-- directions of vector
                color="red",
                alpha=0.8,
                lw=3,
            )

            # Plot y-vect
            c2 = ax.quiver(
                *p0r[0, ...].magnitude,  # <-- starting point of vector
                *yvectr[0, ...].magnitude,  # <-- directions of vector
                color="green",
                alpha=0.8,
                lw=3,
            )
            # Plot z-vect
            c3 = ax.quiver(
                *p0r[0, ...].magnitude,  # <-- starting point of vector
                *zvectr[0, ...].magnitude,  # <-- directions of vector
                color="blue",
                alpha=0.8,
                lw=3,
            )

            def moving_cs(i):
                # Get all quivers
                quivers = []
                for c in ax.get_children():
                    if isinstance(c, matplotlib.lines.mpl.collections.LineCollection):
                        quivers.append(c)
                # Remove last three
                for c in quivers[-3:]:
                    c.remove()
                # Plot x-vector
                c1 = ax.quiver(
                    *p0r[i, ...].magnitude,  # <-- starting point of vector
                    *xvectr[i, ...].magnitude,  # <-- directions of vector
                    color="red",
                    alpha=0.8,
                    lw=3,
                )

                # Plot y-vect
                c2 = ax.quiver(
                    *p0r[i, ...].magnitude,  # <-- starting point of vector
                    *yvectr[i, ...].magnitude,  # <-- directions of vector
                    color="green",
                    alpha=0.8,
                    lw=3,
                )
                # Plot z-vect
                c3 = ax.quiver(
                    *p0r[i, ...].magnitude,  # <-- starting point of vector
                    *zvectr[i, ...].magnitude,  # <-- directions of vector
                    color="blue",
                    alpha=0.8,
                    lw=3,
                )
                line = (c1, c2, c3)
                ax.set_title(f"{p0.dims[0]}={p0.coords[p0.dims[0]][i].data}")
                return line

        else:
            for i in range(p0r.shape[0]):
                # Plot x-vector
                ax.quiver(
                    *p0r[i, ...].magnitude,  # <-- starting point of vector
                    *xvectr[i, ...].magnitude,  # <-- directions of vector
                    color="red",
                    alpha=0.8,
                    lw=3,
                )

                # Plot y-vect
                ax.quiver(
                    *p0r[i, ...].magnitude,  # <-- starting point of vector
                    *yvectr[i, ...].magnitude,  # <-- directions of vector
                    color="green",
                    alpha=0.8,
                    lw=3,
                )
                # Plot z-vect
                ax.quiver(
                    *p0r[i, ...].magnitude,  # <-- starting point of vector
                    *zvectr[i, ...].magnitude,  # <-- directions of vector
                    color="blue",
                    alpha=0.8,
                    lw=3,
                )

    else:
        # Plot x-vector
        ax.quiver(
            *p0.data.magnitude,  # <-- starting point of vector
            *xvect.data.magnitude,  # <-- directions of vector
            color="red",
            alpha=0.8,
            lw=3,
        )

        # Plot y-vect
        ax.quiver(
            *p0.data.magnitude,  # <-- starting point of vector
            *yvect.data.magnitude,  # <-- directions of vector
            color="green",
            alpha=0.8,
            lw=3,
        )
        # Plot z-vect
        ax.quiver(
            *p0.data.magnitude,  # <-- starting point of vector
            *zvect.data.magnitude,  # <-- directions of vector
            color="blue",
            alpha=0.8,
            lw=3,
        )

    # Labels based on units
    ax.set_xlabel(f"X [{units}]")
    ax.set_ylabel(f"Y [{units}]")
    ax.set_zlabel(f"Z [{units}]")

    # Attempt at auto-extents
    xmin = xvect - p0
    xmax = p0 - xvect
    ymin = yvect - p0
    ymax = p0 - yvect
    zmin = zvect - p0
    zmax = p0 - zvect
    extents = (
        np.vstack(
            (
                xmin.data.magnitude,
                xmax.data.magnitude,
                ymin.data.magnitude,
                ymax.data.magnitude,
                zmin.data.magnitude,
                zmax.data.magnitude,
            )
        )
        * 1.2
    )
    extents = extents.reshape(-1, 3)
    x0 = ax.get_xlim()[0]
    x0 = min(x0, extents[:, 0].min())
    x1 = ax.get_xlim()[1]
    x1 = max(x1, extents[:, 0].max())
    ax.set_xlim(x0, x1)

    y0 = ax.get_ylim()[0]
    y0 = min(y0, extents[:, 1].min())
    y1 = ax.get_ylim()[1]
    y1 = max(y1, extents[:, 1].max())
    ax.set_ylim(y0, y1)

    z0 = ax.get_zlim()[0]
    z0 = min(z0, extents[:, 2].min())
    z1 = ax.get_zlim()[1]
    z1 = max(z1, extents[:, 2].max())
    ax.set_zlim(z0, z1)

    if animate:
        return animation.FuncAnimation(
            plt.gcf(), moving_cs, save_count=p0r.shape[0], interval=10, blit=True
        )
    else:
        return ax
