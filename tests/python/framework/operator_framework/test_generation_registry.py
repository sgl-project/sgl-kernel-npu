"""Generation declarations and registry behavior on controlled hardware."""

from unittest.mock import Mock

import pytest


# Verify registration classifies support using hardware detected at registry import time.
@pytest.mark.parametrize("generation_name", ["ARCH22", "ARCH35"])
def test_support_is_classified_against_import_time_hardware(
    registry_for_generation, generation_name
):
    modules = registry_for_generation(generation_name)
    registry, generation = modules.registry, modules.generation
    operator, default_backend, limited_backend = Mock(), Mock(), Mock()
    registry.register_operator("test.op")(operator)
    registry.register_impl("test.op", "default")(default_backend)
    supported = {generation.Generation.ARCH22}
    registry.register_impl("test.op", "limited", supported_generations=supported)(
        limited_backend
    )

    assert registry.get_impl(operator, "default") is default_backend
    assert registry.list_impls(operator) == ["default", "limited"]
    if generation_name == "ARCH22":
        assert registry.get_impl(operator, "limited") is limited_backend
    else:
        with pytest.raises(registry.UnsupportedGenerationError) as caught:
            registry.get_impl(operator, "limited")
        error = caught.value
        assert error.op_name == "test.op"
        assert error.impl == "limited"
        assert error.hardware == modules.hardware
        assert error.supported_generations == supported
        for text in (
            "test.op",
            "limited",
            modules.hardware.generation.value,
            generation.Generation.ARCH22.value,
        ):
            assert text in str(error)
    default_backend.assert_not_called()
    limited_backend.assert_not_called()
    # Import, not lookup or enumeration, is responsible for detection.
    modules.detector.assert_called_once_with()


# Verify support and exclusion sets handle empty, duplicate, current, and other generations.
@pytest.mark.parametrize(
    "option, names, supported",
    [
        ("supported_generations", [], False),
        ("supported_generations", ["ARCH22", "ARCH22"], True),
        ("excluded_generations", [], True),
        ("excluded_generations", ["ARCH22"], False),
        ("excluded_generations", ["ARCH35"], True),
        ("excluded_generations", ["ARCH22", "ARCH35"], False),
    ],
    ids=[
        "support-none",
        "support-deduplicated",
        "exclude-none",
        "exclude-current",
        "exclude-other",
        "exclude-all",
    ],
)
def test_support_and_exclusion_sets(registry_for_generation, option, names, supported):
    modules = registry_for_generation("ARCH22")
    registry, generation = modules.registry, modules.generation
    backend = Mock()
    choices = [generation.Generation[name] for name in names]
    registry.register_impl("test.op", "backend", **{option: choices})(backend)

    if supported:
        assert registry.get_impl_by_name("test.op", "backend") is backend
    else:
        with pytest.raises(registry.UnsupportedGenerationError):
            registry.get_impl_by_name("test.op", "backend")
    assert registry.list_impls_by_name("test.op") == ["backend"]
    backend.assert_not_called()


# Verify support and exclusion options cannot both be supplied, even as empty sets.
def test_support_options_are_mutually_exclusive(registry):
    with pytest.raises(ValueError, match="only"):
        registry.register_impl(
            "test.op", "backend", supported_generations=[], excluded_generations=[]
        )
    assert registry.list_impls_by_name("test.op") == []


# Verify generation strings are rejected without creating an implementation declaration.
@pytest.mark.parametrize("option", ["supported_generations", "excluded_generations"])
def test_generation_strings_are_rejected_without_registration(registry, option):
    with pytest.raises(ValueError, match="Generation"):
        registry.register_impl("test.op", "backend", **{option: ["ARCH22"]})
    assert registry.list_impls_by_name("test.op") == []


# Verify disjoint entries sharing one impl select the current architecture in either registration order.
@pytest.mark.parametrize("architecture", ["ARCH22", "ARCH35"])
@pytest.mark.parametrize("reverse_order", [False, True], ids=["forward", "reverse"])
def test_disjoint_entries_select_current_architecture(
    registry_for_generation, architecture, reverse_order
):
    modules = registry_for_generation(architecture)
    registry, generation = modules.registry, modules.generation
    entries = {"ARCH22": Mock(), "ARCH35": Mock()}
    names = list(entries)
    if reverse_order:
        names.reverse()
    for name in names:
        registry.register_impl(
            "test.op", "backend", supported_generations={generation.Generation[name]}
        )(entries[name])
    assert registry.get_impl_by_name("test.op", "backend") is entries[architecture]
    assert registry.list_impls_by_name("test.op") == ["backend"]
    for backend in entries.values():
        backend.assert_not_called()


# Verify aliases may express the same effective support using inclusion or exclusion sets.
def test_equivalent_support_declarations_allow_aliases(registry_for_generation):
    modules = registry_for_generation("ARCH22")
    registry, generation = modules.registry, modules.generation
    backend = Mock()
    registry.register_impl(
        "test.op",
        "backend",
        supported_generations={generation.Generation.ARCH22},
    )(backend)
    registry.register_impl(
        "test.op", "auto", excluded_generations={generation.Generation.ARCH35}
    )(backend)

    assert registry.get_impl_by_name("test.op", "auto") is backend
    assert registry.get_impl_by_name("test.op", "backend") is backend
    assert registry.list_impls_by_name("test.op") == ["auto", "backend"]
    backend.assert_not_called()


# Verify different impl aliases may share a callable while declaring independent support ranges.
@pytest.mark.parametrize(
    "excluded_names", [None, ("ARCH35",)], ids=["default-all", "exclude-arch35"]
)
def test_aliases_have_independent_support_ranges(
    registry_for_generation, excluded_names
):
    modules = registry_for_generation("ARCH22")
    registry, generation = modules.registry, modules.generation
    backend = Mock()
    registry.register_impl(
        "test.op", "triton", supported_generations={generation.Generation.ARCH35}
    )(backend)

    alias_options = (
        {}
        if excluded_names is None
        else {
            "excluded_generations": {
                generation.Generation[name] for name in excluded_names
            }
        }
    )
    registry.register_impl("test.op", "auto", **alias_options)(backend)
    assert registry.list_impls_by_name("test.op") == ["auto", "triton"]
    assert registry.get_impl_by_name("test.op", "auto") is backend
    with pytest.raises(registry.UnsupportedGenerationError) as caught:
        registry.get_impl_by_name("test.op", "triton")
    assert caught.value.supported_generations == {generation.Generation.ARCH35}
    backend.assert_not_called()


# Verify a shared callable may declare different generation support for different operators.
def test_shared_callable_support_is_scoped_to_operator(registry_for_generation):
    modules = registry_for_generation("ARCH22")
    registry, generation = modules.registry, modules.generation
    shared_backend = Mock()
    registry.register_impl("test.first", "backend")(shared_backend)
    registry.register_impl(
        "test.second", "backend", supported_generations={generation.Generation.ARCH35}
    )(shared_backend)
    assert registry.get_impl_by_name("test.first", "backend") is shared_backend
    with pytest.raises(registry.UnsupportedGenerationError):
        registry.get_impl_by_name("test.second", "backend")


# Verify later changes to the input support list do not alter the captured declaration.
def test_support_declaration_is_a_snapshot(registry_for_generation):
    modules = registry_for_generation("ARCH22")
    registry, generation = modules.registry, modules.generation
    choices = [generation.Generation.ARCH22]
    backend = Mock()
    decorator = registry.register_impl(
        "test.op", "backend", supported_generations=choices
    )
    choices.clear()
    decorator(backend)
    assert registry.get_impl_by_name("test.op", "backend") is backend


# Verify overlapping support is rejected atomically even when the old entry is unavailable locally.
@pytest.mark.parametrize("original_architecture", ["ARCH22", "ARCH35"])
def test_overlapping_support_is_rejected_atomically(
    registry_for_generation, original_architecture
):
    modules = registry_for_generation("ARCH22")
    registry, generation = modules.registry, modules.generation
    original, replacement, complement = Mock(), Mock(), Mock()
    supported = {generation.Generation[original_architecture]}
    registry.register_impl("test.op", "backend", supported_generations=supported)(
        original
    )
    with pytest.raises(ValueError, match="overlapping generations") as caught:
        registry.register_impl("test.op", "backend")(replacement)
    assert generation.Generation[original_architecture].value in str(caught.value)
    assert registry.list_impls_by_name("test.op") == ["backend"]
    if original_architecture == "ARCH22":
        assert registry.get_impl_by_name("test.op", "backend") is original
    else:
        with pytest.raises(registry.UnsupportedGenerationError) as unsupported:
            registry.get_impl_by_name("test.op", "backend")
        assert unsupported.value.supported_generations == supported
    # Rejected overlap must not claim the previously unregistered architecture.
    other = "ARCH35" if original_architecture == "ARCH22" else "ARCH22"
    registry.register_impl(
        "test.op", "backend", supported_generations={generation.Generation[other]}
    )(complement)
    expected = original if original_architecture == "ARCH22" else complement
    assert registry.get_impl_by_name("test.op", "backend") is expected
    for backend in (original, replacement, complement):
        backend.assert_not_called()


# Verify an empty support declaration remains visible and can later receive an architecture entry.
def test_empty_support_declaration_can_be_extended(registry_for_generation):
    modules = registry_for_generation("ARCH22")
    registry, generation = modules.registry, modules.generation
    empty, available = Mock(), Mock()
    registry.register_impl("test.op", "backend", supported_generations=[])(empty)
    assert registry.list_impls_by_name("test.op") == ["backend"]
    with pytest.raises(registry.UnsupportedGenerationError) as caught:
        registry.get_impl_by_name("test.op", "backend")
    assert caught.value.supported_generations == set()
    registry.register_impl(
        "test.op", "backend", supported_generations={generation.Generation.ARCH22}
    )(available)
    assert registry.get_impl_by_name("test.op", "backend") is available
    assert registry.list_impls_by_name("test.op") == ["backend"]
    empty.assert_not_called()
    available.assert_not_called()
