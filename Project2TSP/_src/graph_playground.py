#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import jraph
import numpy as np
import jax.numpy as jnp
import jax
from jraph import GraphsTuple
from matrix_helper import calculate_distances

jax.config.update("jax_traceback_filtering", "off")

# In[48]:


def build_random_city_graph(num_cities: int) -> GraphsTuple:
    # Generate random position for each city.
    position = np.random.uniform(0, 1, [num_cities, 2])

    # Connect all particles to all particles.
    particle_indices = np.arange(num_cities)
    senders, receivers = np.meshgrid(particle_indices, particle_indices)
    senders, receivers = senders.flatten(), receivers.flatten()

    distances = calculate_distances(position)
    distances = distances.flatten()

    # Remove interactions of particles to themselves.
    mask = senders != receivers
    senders, receivers = senders[mask], receivers[mask]
    distances = distances[mask]
    num_interactions = receivers.shape[0]

    return jraph.GraphsTuple(
        n_node=np.asarray([num_cities]),
        n_edge=np.asarray([num_interactions]),
        nodes=position,
        edges=distances,
        globals=np.eye(num_cities),
        senders=senders,
        receivers=receivers,
    )


# In[49]:


graph = build_random_city_graph(10)


# In[50]:


import matplotlib.pyplot as plt

cities = graph.nodes

plt.scatter(cities[:, 0], cities[:, 1])


# In[18]:


from flax import linen as nn


# In[51]:


X = graph.globals

np.roll(X, 1, axis=0).T @ X


# In[36]:


# def update_edge_fn(edges, senders, receivers, globals_):
#     del globals_
#     distance = jnp.linalg.norm(senders["position"] - receivers["position"])
#     hookes_potential_per_edge = 0.5 * edges["spring_constant"] * distance**2
#     return {"hookes_potential": hookes_potential_per_edge}


# def update_node_fn(nodes, sent_edges, received_edges, globals_):
#     del sent_edges, received_edges, globals_
#     momentum_norm = jnp.linalg.norm(nodes["momentum"])
#     kinetic_energy_per_node = momentum_norm**2 / (2 * nodes["mass"])
#     return {"kinetic_energy": kinetic_energy_per_node}


# # @jraph.concatenated_args
# # def update_global_fn(concatenated_args):
# #     return concatenated_args


# net = jraph.GraphNetwork(
#     update_edge_fn=update_edge_fn,
#     update_node_fn=update_node_fn,
# )


# In[53]:


from typing import Sequence, Callable


def add_graphs_tuples(
    graphs: jraph.GraphsTuple, other_graphs: jraph.GraphsTuple
) -> jraph.GraphsTuple:
    """Adds the nodes, edges and global features from other_graphs to graphs."""
    return graphs._replace(
        nodes=graphs.nodes + other_graphs.nodes,
        edges=graphs.edges + other_graphs.edges,
        globals=graphs.globals + other_graphs.globals,
    )


class MLP(nn.Module):
    """A multi-layer perceptron."""

    feature_sizes: Sequence[int]
    dropout_rate: float = 0
    deterministic: bool = True
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for size in self.feature_sizes:
            x = nn.Dense(features=size)(x)
            x = self.activation(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(x)
        return x


class GraphNet(nn.Module):
    """A complete Graph Network model defined with Jraph."""

    latent_size: int
    num_mlp_layers: int
    message_passing_steps: int
    output_globals_size: int
    dropout_rate: float = 0
    skip_connections: bool = True
    use_edge_model: bool = True
    layer_norm: bool = True
    deterministic: bool = True

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # We will first linearly project the original features as 'embeddings'.
        embedder = jraph.GraphMapFeatures(
            embed_node_fn=nn.Dense(self.latent_size),
            embed_edge_fn=nn.Dense(self.latent_size),
            embed_global_fn=nn.Dense(self.latent_size),
        )
        processed_graphs = embedder(graphs)

        # Now, we will apply a Graph Network once for each message-passing round.
        mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
        for _ in range(self.message_passing_steps):
            if self.use_edge_model:
                update_edge_fn = jraph.concatenated_args(
                    MLP(
                        mlp_feature_sizes,
                        dropout_rate=self.dropout_rate,
                        deterministic=self.deterministic,
                    )
                )
            else:
                update_edge_fn = None

            update_node_fn = jraph.concatenated_args(
                MLP(
                    mlp_feature_sizes,
                    dropout_rate=self.dropout_rate,
                    deterministic=self.deterministic,
                )
            )
            update_global_fn = jraph.concatenated_args(
                MLP(
                    mlp_feature_sizes,
                    dropout_rate=self.dropout_rate,
                    deterministic=self.deterministic,
                )
            )

            graph_net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                update_global_fn=update_global_fn,
            )

            if self.skip_connections:
                processed_graphs = add_graphs_tuples(
                    graph_net(processed_graphs), processed_graphs
                )
            else:
                processed_graphs = graph_net(processed_graphs)

            if self.layer_norm:
                processed_graphs = processed_graphs._replace(
                    nodes=nn.LayerNorm()(processed_graphs.nodes),
                    edges=nn.LayerNorm()(processed_graphs.edges),
                    globals=nn.LayerNorm()(processed_graphs.globals),
                )

        # Since our graph-level predictions will be at globals, we will
        # decode to get the required output logits.
        decoder = jraph.GraphMapFeatures(
            embed_global_fn=nn.Dense(self.output_globals_size)
        )
        processed_graphs = decoder(processed_graphs)

        return processed_graphs


# In[54]:


latent_size = 64
num_mlp_layers = 2
message_passing_steps = 3
output_globals_size = 10 * 10
model = GraphNet(
    latent_size=latent_size,
    num_mlp_layers=num_mlp_layers,
    message_passing_steps=message_passing_steps,
    output_globals_size=output_globals_size,
    skip_connections=True,
    use_edge_model=True,
    layer_norm=True,
    deterministic=False,
)


# In[63]:


import jax

model.apply(graph)


# In[62]:


graph.globals.shape


# In[ ]:
