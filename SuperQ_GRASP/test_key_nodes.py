import open3d as o3d
import numpy as np
import json

object_name = "Chair_custom_fixed"
# Load the mesh file
mesh = o3d.io.read_triangle_mesh(f"./SuperQ_GRASP/object-models/{object_name}.obj")

# Load the json file containing the key nodes
with open(f"./SuperQ_GRASP/object-models/{object_name}_outermost_sq_locations.json", "r") as f:
    key_nodes = json.load(f)

# Visualize the mesh and key nodes
sq_outermost_centers = []
for key_node in key_nodes:
    sq_outermost_center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sq_outermost_center_sphere.translate(key_node)
    sq_outermost_center_sphere.paint_uniform_color((1, 0, 0))
    sq_outermost_centers.append(sq_outermost_center_sphere)

# Create the window to display everything
vis= o3d.visualization.Visualizer()
vis.create_window()


vis.add_geometry(mesh)
for sq_outermost_center in sq_outermost_centers:
    vis.add_geometry(sq_outermost_center)

vis.run()

# Close all windows
vis.destroy_window()