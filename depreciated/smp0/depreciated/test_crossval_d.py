import numpy as np

def generate_G_crossval(U_partitioned):
    count = 0
    initial_shape = np.dot(U_partitioned[0], U_partitioned[1].T).shape
    sum_of_products = np.zeros(initial_shape)
    for i, partition_1 in enumerate(U_partitioned):
        for j, partition_2 in enumerate(U_partitioned):
            if i != j:
                partition_1, partition_2 = (partition_1,
                                            partition_2)
                product = np.dot(partition_1, partition_2.T)  # Transposing the second partition
                sum_of_products += product
                count += 1

    G_crossval = sum_of_products / count
    return G_crossval

def generate_D_squared(G_crossval):
    D_squared = np.zeros(G_crossval.shape)
    for i in range(D_squared.shape[0]):
        for j in range(D_squared.shape[0]):
            D_squared[i, j] = G_crossval[i, i] + G_crossval[j, j] - G_crossval[i, j] - G_crossval[j, i]
    return D_squared

# Example to generate a matrix that produces negative values in D_squared
# Create a set of partitioned matrices with specific patterns to induce negative values in D_squared

matrix_size = (5, 5)
num_partitions = 4

# Create an example matrix U
U = np.random.rand(matrix_size[0] * num_partitions, matrix_size[1])

# Partition U
U_partitioned = np.array_split(U, num_partitions)

# Generate G_crossval and D_squared using the above matrices
G_crossval = generate_G_crossval(U_partitioned)
D_squared = generate_D_squared(G_crossval)

D_squared

