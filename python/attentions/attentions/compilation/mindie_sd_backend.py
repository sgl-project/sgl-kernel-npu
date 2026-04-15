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

import functools
import logging
from typing import Any, Callable, Optional, Sequence

import torch
import torch.fx as fx
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.freezing import freeze
from ._custom_decomposition import select_custom_decomp_table

try:
    from torch.fx.passes.graph_transform_observer import GraphTransformObserver
except ImportError:
    class GraphTransformObserver:
        def __init__(self, gm, passname, subsystem=None, log_url=None):
            self.gm = gm
            self.passname = passname
            self.subsystem = subsystem
            self.log_url = log_url
        
        def apply_gm_pass(self, pass_obj):
            pass_obj(self.gm)
        
        def apply_graph_pass(self, pass_func):
            pass_func(self.gm.graph)

from .compiliation_config import CompilationConfig

from .passes import activate_pattern_once
from .passes.register_pattern_to_pass import patterns
from .passes.redundant_node_elimination_pass import ReduandantNodeEliminationPass

logger = logging.getLogger(__name__)


def decompose_auto_functionalized(graph: fx.Graph):
    try:
        from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized as original_decompose
        return original_decompose(graph)
    except ImportError:
        for node in list(graph.nodes):
            if node.op == 'call_function' and 'auto_functionalized' in str(node.target):
                orig_name = node.target.__name__.replace('_auto_functionalized', '')
                if hasattr(torch, orig_name):
                    node.target = getattr(torch, orig_name)
        graph.eliminate_dead_code()
        graph.lint()
        return graph


class MindieSDBackend:
    """
    The compilation backend for 'torch.compile'.
    It is used to process the FX graph and perform custom operation fusing etc.
    """

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        """
        Process the FX graph and perform custom operation fusing.

        Args:
            graph (fx.Graph): The FX graph to be processed.
            example_inputs (optional): Example inputs for the graph.

        Returns:
            fx.Graph: The processed FX graph with custom operation fusing applied.
        """
        graph = self.compile(graph, example_inputs)
        return graph

    @classmethod
    def apply_redundant_node_elimination_pass(cls, graph: fx.GraphModule, inputs):
        GraphTransformObserver(
            gm=graph,
            passname="redundant_node_elimination_pass",
            subsystem="redundant_node_elimination_pass",
            log_url=CompilationConfig.graph_log_url,
        ).apply_gm_pass(ReduandantNodeEliminationPass())
        logger.debug("Graph after redundant node elimination pass:")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(graph.print_readable(print_output=False))

    @classmethod
    def apply_pattern_match_passes(cls, graph: fx.GraphModule, inputs):
        activate_pattern_once()
        GraphTransformObserver(
            gm=graph,
            passname="pattern_match_pass",
            subsystem="pattern_match_passes",
            log_url=CompilationConfig.graph_log_url,
        ).apply_gm_pass(patterns)
        logger.debug("Graph after pattern matching:")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(graph.print_readable(print_output=False))

    @classmethod
    def apply_decompose_auto_functionalized_pass(cls, graph: fx.GraphModule):
        GraphTransformObserver(
            gm=graph,
            passname="decompose_auto_functionalized",
            subsystem="decompose_auto_functionalized_pass",
            log_url=CompilationConfig.graph_log_url,
        ).apply_graph_pass(decompose_auto_functionalized)

    def compile(
        self,
        gm: fx.GraphModule,
        example_inputs,
        **kwargs,
    ) -> tuple[Callable, Optional[Any]]:
        def freezing_compile(compile_inner, aot_autograd_gm, example_inputs):
            # Freeze the graph first before passing to AOT Autograd.
            frozen_gm, preserved_arg_indices = freeze(
                gm, aot_autograd_gm, example_inputs
            )
            example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]
            optimized_function = compile_inner(frozen_gm, example_inputs)

            def wrapper(args: list[object]) -> Sequence[torch.Tensor]:
                args_new = [args[i] for i in preserved_arg_indices]
                args.clear()
                return optimized_function(*args_new)

            wrapper._boxed_call = True  # type: ignore[attr-defined]

            return wrapper

        def graph_rewrite_before_freezing(fx_graph, inputs):
            logger.debug("Graph before compiling:")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(fx_graph.print_readable(print_output=False))
            self.__class__.apply_redundant_node_elimination_pass(fx_graph, inputs)
            self.__class__.apply_pattern_match_passes(fx_graph, inputs)
            return fx_graph

        def graph_rewrite_after_freezing(fx_graph, inputs):
            self.__class__.apply_redundant_node_elimination_pass(fx_graph, inputs)
            # make sure we add freezing passes after constant folding
            self.__class__.apply_decompose_auto_functionalized_pass(fx_graph)
            logger.debug("Graph after compiling:")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(fx_graph.print_readable(print_output=False))
            return fx_graph

        def compile_inner(fx_graph, inputs):
            # we split the rewrite into two phases: before and after freezing
            # since freezing would do CSE which might break some assumptions in
            # the rewrite rules.
            graph_rewrite_before_freezing(fx_graph, inputs)
            if CompilationConfig.enable_freezing:
                return freezing_compile(graph_rewrite_after_freezing, fx_graph, inputs)
            else:
                return graph_rewrite_after_freezing(fx_graph, inputs)

        decompositions = select_custom_decomp_table()
        # Use AOT Autograd to handle the forward compilation.
        return aot_autograd(
            fw_compiler=compile_inner,
            decompositions=decompositions,
        )(gm, example_inputs)