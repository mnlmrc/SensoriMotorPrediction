import motornet as mn
import matplotlib.pyplot as plt

# Define the joints
joints = ['root', 'branch1', 'branch2']

# Parent relationships
parents = {'branch1': 'root', 'branch2': 'root'}

# Degrees of freedom
# Let's say each joint rotates (1 DoF), you can adjust
dof = 2

# Space dimension (2D or 3D) — let's start with 2D for simplicity
space_dim = 2

# Now create the skeleton properly
skeleton = mn.skeleton.Skeleton(
    joints=joints,
    parents=parents,
    dof=dof,
    space_dim=space_dim
)

muscle = mn.muscle.ReluMuscle()

effector = mn.effector.Effector(
    skeleton=skeleton,
    muscle=muscle
)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# effector.plot(ax=ax)
# plt.show()

