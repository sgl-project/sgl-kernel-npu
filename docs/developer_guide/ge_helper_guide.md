# GE Helper Library

## Overview

The GE Helper library provides a set of C++ utilities for working with Huawei's Graph Engine (GE) framework. It simplifies the process of defining operations, managing tensor descriptions, and handling attribute definitions for custom NPU kernels.

## Key Components

### 1. Data Type Conversion Functions

```cpp
constexpr ge::DataType SCALAR_TYPE_TO_GE_DATATYPE(at::ScalarType scalarType)
constexpr uint32_t GE_DATATYPE_TO_KEY(ge::DataType geDatatype)
```

These constexpr functions provide compile-time conversion between PyTorch scalar types and GE data types, as well as mapping GE data types to integer keys for tiling purposes.

### 2. Parameter Type Definitions

```cpp
enum class ParamTypeCls : uint32_t {
    REQUIRED = 0,
    OPTIONAL,
};

using AttrTypeCls = ParamTypeCls;
constexpr auto REQUIRED = ParamTypeCls::REQUIRED;
constexpr auto OPTIONAL = ParamTypeCls::OPTIONAL;
```

Defines parameter type classification for input/output tensors and attributes.


### 3. TilingContext Class

Manages the context for operation tiling including tensor registration and attribute management.

**Key Methods:**
- `RegisterTensor()` - Register PyTorch tensor(inputs and outputs, should be called follow tensor defined order in definition file), parameter scalerType is true for input tensor, false for output tensor.
- `GetInputDesc()` - Get input description.
- `GetInputShape()` - Get input shape by index.
- `GetInputTensor()` - Get input tensor by index.
- `GetOutputDesc()` - Get output description by index.
- `GetOutputShape()` - Get output shape by index.
- `GetOutputTensor()` - Get output tensor by index.
- `GetOptionalInputDesc` - Get optional input description by index.
- `GetOptionalInputShape()` - Get optional input shape by index.
- `GetOptionalInputTensor()` - Get optional input tensor by index.
- `GetTilingTensor()` - Serialize tiling data into a persistent device cache and return a graph-stable tensor view.
Note: In CANN framework, the inputs and optional inputs are stored in the same container, we keep it so the index is not separated between inputs and optional inputs.

### 4. Device Tiling Cache

Directly launched kernels do not have an `aclOpExecutor` to own the device-side tiling buffer. `TilingContext::GetTilingTensor()` provides that ownership in the helper layer, so individual operators do not implement cache maps, capture checks, or H2D copies.

```cpp
SparseAttnSharedkvTiling tiling(context.get());
TORCH_CHECK(tiling.DoOpTiling(&info) == ge::GRAPH_SUCCESS, "tiling failed");

auto tilingTensor = context->GetTilingTensor(tiling.GetTilingData());
```

The cache has the following behavior:

- One cache per tiling data type and NPU device.
- At most 512 persistent tiling records in each cache.
- Eager execution inserts a missing record and copies it to the device.
- NPU graph capture only accepts an existing record. Run one eager call with the same tiling configuration before capture.
- Once the cache is full, eager execution uses a one-shot tiling tensor; graph capture rejects an uncached configuration.
- Lookup uses the complete serialized tiling data as its key. Tiling structures must be trivially copyable and deterministically initialized, including padding. Value-initialize them with `{}`.

### 5. OpDef Class

Defines operation signatures including inputs, outputs, and attributes.

**Key Methods:**
- `Input()` - Define input parameter
- `Output()` - Define output parameter
- `Attr()` - Define attribute parameter
- `SetAttrStr()` - Set string attribute value, in the definition file we set the default value and string, but if the user needs to change the value, they can use this method.
- `SetAttrAny()` - Set any-type attribute value.
- `SetToContext()` - **The most important method**, apply definition to tiling context so we can use it in the tiling process. It has a type parameter, which can choose type from types and formats the op supports.

## Usage Examples

### Defining a Custom Operation
This is the classic way to define a custom operation. Through the `OpDef` class, you can define an operation with its attributes(default value) and inputs/outputs(support types and formats). Unlike GE framework, **sglang-kernel-npu framework does not need to register the operation**, so similar code in the below should be deleted.
```cpp
#include "register/op_def_registry.h"

OP_ADD(LightningIndexer);
```

### Setting Up Tiling Context

```cpp
auto context = std::make_shared<TilingContext>("MyCustomOp");
at::ScalarType scalarType = at::ScalarType::Half;

// Apply operation definition to context
op.SetToContext(context, scalarType);

// Register tensors
context->RegisterTensor(input_tensor, true);   // input
context->RegisterTensor(output_tensor, false); // output
```

### Accessing Attributes And Tensors

```cpp
auto attrs = context->GetAttrs();
const int* scale_ptr = attrs->GetAttrPointer<int>(0);
int scale_value = *scale_ptr;

auto query_desc = context->GetInputDesc(0);
auto query_shape = context->GetInputShape(0);
auto key_desc = context->GetInputDesc(1);
auto key_shape = context->GetInputShape(1);
```

## Error Handling

The library uses `TORCH_CHECK` for error handling instead of exceptions to maintain compatibility with ACL graph execution environments:

- Input validation with descriptive error messages
- Bounds checking for array accesses
- Type checking for attribute access
- Initialization state validation

## Thread Safety

Individual context instances are designed to be used in single-threaded host tiling code. The shared device tiling cache protects lookup and insertion with an internal mutex.

## Dependencies

- PyTorch
- Huawei CANN framework
- C++17 standard library (for `std::any` support)

## Limitations

- Maximum tensor dimensionality limited to 4D
- Requires C++17 or later for `std::any` support
