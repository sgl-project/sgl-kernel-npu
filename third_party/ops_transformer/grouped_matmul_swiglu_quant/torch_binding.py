# SPDX-License-Identifier: Apache-2.0
from cann_ops_transformer.op_builder import OpBuilder


class SglGroupedMatmulSwigluQuantOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("sgl_grouped_matmul_swiglu_quant", category="gmm")

    def sources(self):
        return ["csrc/gmm/sgl_grouped_matmul_swiglu_quant.cpp"]

    def schema(self):
        return None

    def register_meta(self):
        pass
