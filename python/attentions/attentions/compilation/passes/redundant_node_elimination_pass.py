#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import logging
import operator

import torch
import torch.fx as fx

from .gm_pass_base import GraphModulePass

logger = logging.getLogger(__name__)


def get_node_shape(node: torch.fx.node) -> torch.Size | None:
    """Retrieve the shape of the tensor represented by the node, if available."""
    if not hasattr(node, "meta") or "val" not in node.meta:
        return None
    if isinstance(node.meta["val"], torch.Tensor):
        return node.meta["val"].shape
    else:
        return None


class ReduandantNodeEliminationPass(GraphModulePass):
    def __call__(self, gm: fx.GraphModule) -> None:
        """
        Runs a graph pass to eliminate semantically no-op nodes.

        Rules:
        1. Eliminates torch.ops.aten.clone.default if memory_format is default (Preserve).
        2. Eliminates torch.ops.aten.split_with_sizes.default if it only has one split,
        and replaces its (getitem, 0) users.
        3. Eliminates no-op view/reshape ops.
        """
        graph = gm.graph
        modified = False

        # We iterate over a static list of nodes, as we will be modifying
        # the graph's node list during iteration.
        for node in list(graph.nodes):
            # Rule 1: torch.ops.aten.clone.default
            if node.target == torch.ops.aten.clone.default:
                # signature: clone(Tensor self, *, MemoryFormat? memory_format=None)
                # memory_format is a keyword-only argument.

                # Get the memory_format arg, defaulting to None if not present
                memory_format = node.kwargs.get("memory_format")

                # None (default) and torch.preserve_format are both no-ops
                is_noop_clone = (memory_format is None) or (
                    memory_format == torch.preserve_format
                )

                if is_noop_clone:
                    # The input to the clone is the first argument
                    input_node = node.args[0]

                    # Replace all uses of the clone node with its input
                    node.replace_all_uses_with(input_node)

                    # Erase the clone node itself
                    graph.erase_node(node)
                    modified = True

            # Rule 2: torch.ops.aten.split_with_sizes.default
            elif node.target == torch.ops.aten.split_with_sizes.default:
                # signature: split_with_sizes(Tensor self, int[] split_sizes, int dim=0)

                # The split_sizes list is the second argument
                split_sizes_arg = node.args[1]

                # We can only optimize if split_sizes is a constant list/tuple
                if (
                    isinstance(split_sizes_arg, (list, tuple))
                    and len(split_sizes_arg) == 1
                ):
                    # This split operation produces a list of 1 tensor,
                    # which is just the original input tensor.
                    input_tensor_node = node.args[0]

                    # We must check all users *before* modifying the graph.
                    # All users MUST be `operator.getitem` with index 0.
                    # If the list is used in any other way, we cannot eliminate this.

                    can_eliminate = True
                    users_to_replace = []

                    # Iterate over a static list of users
                    for user_node in list(node.users.keys()):
                        # Check if the user is `getitem(self, 0)`
                        if (
                            user_node.target == operator.getitem
                            and len(user_node.args) == 2
                            and user_node.args[1] == 0
                        ):
                            users_to_replace.append(user_node)
                        else:
                            # This node is used in a way we don't support
                            # (e.g., passed as a whole list)
                            can_eliminate = False
                            break

                    if can_eliminate and users_to_replace:
                        # If all users are valid, replace them
                        for user_node in users_to_replace:
                            user_node.replace_all_uses_with(input_tensor_node)
                            graph.erase_node(user_node)

                        # After all users are gone, the split node has no
                        # users and can be erased.
                        graph.erase_node(node)
                        modified = True

            elif (
                node.target == torch.ops.aten.view.default
                or node.target == torch.ops.aten.reshape.default
            ):
                # remove unused view nodes if the shape is the same as input
                # we rely on the shape info via shape propagation
                input_node_shape = get_node_shape(node.args[0])
                output_node_shape = get_node_shape(node)
                if (
                    input_node_shape is not None
                    and output_node_shape is not None
                    and input_node_shape == output_node_shape
                ):
                    input_node = node.args[0]
                    node.replace_all_uses_with(input_node)
                    graph.erase_node(node)
                    modified = True

        if modified:
            # Clean up any dangling nodes that might have resulted
            graph.eliminate_dead_code()
            # Re-compile the graph module with the changes
            gm.recompile()
