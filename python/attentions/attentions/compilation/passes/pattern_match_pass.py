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
from typing import Any, Callable, Dict, List, Tuple, Optional, Sequence
import re
import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass

from .gm_pass_base import GraphModulePass
from .._custom_decomposition import select_custom_decomp_table

logger = logging.getLogger(__name__)

torch_version = re.match(r"(\d+\.\d+)", torch.__version__).group(1)
IS_TORCH_21 = torch_version == "2.1"
if IS_TORCH_21:
    def mindie_inference_graph(fn, args):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._subclasses.fake_tensor import FakeTensor

        decomp_table = select_custom_decomp_table()

        def safe_to_copy(x, dtype=None, layout=None, device=None, pin_memory=False, non_blocking=False):
            if isinstance(x, FakeTensor):
                return x
            return torch.ops.aten._to_copy.default(x, dtype, layout, device, pin_memory, non_blocking)

        decomp_table[torch.ops.aten._to_copy.default] = safe_to_copy
        gm = make_fx(fn, decomposition_table=decomp_table)(*args)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        return gm


class PatternMatchPass(GraphModulePass):
    def __init__(self):
        self.pattern_replacements: Dict[
            str, Tuple[Callable[..., Any], Callable[..., Any]]
        ] = {}
        try:
            self.pattern_pass: PatternMatcherPass = PatternMatcherPass(
                pass_name="pattern_match_pass"
            )
        except TypeError:
            self.pattern_pass: PatternMatcherPass = PatternMatcherPass()

    def __call__(self, graph: torch.fx.GraphModule) -> torch.fx.GraphModule:
        matched_cnt = 0
        while True:
            cnt = self.pattern_pass.apply(graph)
            if cnt == 0:
                break
            matched_cnt += cnt
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("PatternMatchPass replace %d patterns.", matched_cnt)
            pattern_idx = 0
            logger.debug("Patterns registered for replacement:")
            try:
                from torch._inductor.pattern_matcher import PatternPrettyPrinter
                for pattern_entry in self.pattern_pass.patterns.values():
                    for p in pattern_entry:
                        p_str = PatternPrettyPrinter.run(p.pattern)
                        logger.debug("Pattern %d: %s", pattern_idx, p_str)
                        pattern_idx += 1
            except ImportError:
                logger.debug("PatternPrettyPrinter not available, skipping pattern printing")
        return graph

    def uuid(self) -> Any:
        return super().uuid()

    def register_pattern(
        self,
        name: str,
        pattern: Callable[..., Any],
        replacement: Callable[..., Any],
        example_inputs: List[torch.Tensor],
    ):
        if name in self.pattern_replacements:
            raise ValueError(f"Pattern '{name}' is already registered.")

        self.pattern_replacements[name] = (pattern, replacement)
        logger.debug("Registering pattern: %s", name)

        if not hasattr(pm, "fwd_only"):
            if IS_TORCH_21:
                pm.fwd_only = mindie_inference_graph
            else:
                logger.warning("fwd_only not available in current torch version")

        def fwd_only_with_custom_decomp(
            fn: Callable[..., Any],
            args: Sequence[Any],
            *,
            run_functional_passes: bool = True,
            get_decomp_fn: Optional[Callable[..., Any]] = select_custom_decomp_table,
        ) -> torch.fx.GraphModule:
            if IS_TORCH_21:
                return pm.fwd_only(
                    fn=fn,
                    args=args
                )
            else:
                return pm.fwd_only(
                    fn=fn,
                    args=args,
                    run_functional_passes=run_functional_passes,
                    get_decomp_fn=get_decomp_fn
                )

        try:
            pm.register_replacement(
                pattern,
                replacement,
                example_inputs,
                fwd_only_with_custom_decomp,
                self.pattern_pass.patterns,
            )
            logger.debug("Successfully register pattern: %s", name)
        except RuntimeError as e:
            if "Duplicate pattern" in str(e):
                logger.warning(
                    "Pattern '%s' is already registered. Skipping duplicate registration.",
                    name,
                )
            else:
                raise e