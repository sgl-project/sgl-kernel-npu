"""Generation declarations and registry behavior on controlled hardware."""

from unittest.mock import Mock

import pytest


# Verify registration classifies support using hardware detected at registry import time.
@pytest.mark.parametrize("generation_name", ["A2", "A3", "A5"])
def test_support_is_classified_against_import_time_hardware(
    registry_for_generation, generation_name
):
    modules = registry_for_generation(generation_name)
    registry, generation = modules.registry, modules.generation
    operator, default_backend, limited_backend = Mock(), Mock(), Mock()
    registry.register_operator("test.op")(operator)
    registry.register_impl("test.op", "default")(default_backend)
    supported = {generation.Generation.A3}
    registry.register_impl("test.op", "limited", supported_generations=supported)(
        limited_backend
    )

    assert registry.get_impl(operator, "default") is default_backend
    assert registry.list_impls(operator) == ["default", "limited"]
    if generation_name == "A3":
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
            modules.hardware.soc_name,
            generation_name,
            "A3",
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
        ("supported_generations", ["A2", "A3", "A3"], True),
        ("excluded_generations", [], True),
        ("excluded_generations", ["A3"], False),
        ("excluded_generations", ["A5"], True),
        ("excluded_generations", ["A2", "A3", "A5"], False),
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
    modules = registry_for_generation("A3")
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
        registry.register_impl("test.op", "backend", **{option: ["A3"]})
    assert registry.list_impls_by_name("test.op") == []


# Verify an unsupported declaration reserves its key against replacement for another generation.
def test_unsupported_declaration_still_reserves_the_key(registry_for_generation):
    modules = registry_for_generation("A2")
    registry, generation = modules.registry, modules.generation
    original, replacement = Mock(), Mock()
    registry.register_impl(
        "test.op", "backend", supported_generations={generation.Generation.A3}
    )(original)
    with pytest.raises(ValueError, match="Duplicate impl"):
        registry.register_impl(
            "test.op", "backend", supported_generations={generation.Generation.A2}
        )(replacement)
    with pytest.raises(registry.UnsupportedGenerationError) as caught:
        registry.get_impl_by_name("test.op", "backend")
    assert caught.value.supported_generations == {generation.Generation.A3}
    assert registry.list_impls_by_name("test.op") == ["backend"]
    original.assert_not_called()
    replacement.assert_not_called()


# Verify aliases may express the same effective support using inclusion or exclusion sets.
def test_equivalent_support_declarations_allow_aliases(registry_for_generation):
    modules = registry_for_generation("A3")
    registry, generation = modules.registry, modules.generation
    backend = Mock()
    registry.register_impl(
        "test.op",
        "backend",
        supported_generations={generation.Generation.A2, generation.Generation.A3},
    )(backend)
    registry.register_impl(
        "test.op", "auto", excluded_generations={generation.Generation.A5}
    )(backend)

    assert registry.get_impl_by_name("test.op", "auto") is backend
    assert registry.get_impl_by_name("test.op", "backend") is backend
    assert registry.list_impls_by_name("test.op") == ["auto", "backend"]
    backend.assert_not_called()


# Verify default and exclusion-based aliases reject conflicting support without partial registration.
@pytest.mark.parametrize(
    "excluded_names", [None, ("A5",)], ids=["default-all", "exclude-a5"]
)
def test_conflicting_alias_support_is_rejected_atomically(
    registry_for_generation, excluded_names
):
    modules = registry_for_generation("A2")
    registry, generation = modules.registry, modules.generation
    backend = Mock()
    registry.register_impl(
        "test.op", "triton", supported_generations={generation.Generation.A3}
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
    with pytest.raises(ValueError, match="Aliases"):
        registry.register_impl("test.op", "auto", **alias_options)(backend)
    assert registry.list_impls_by_name("test.op") == ["triton"]
    with pytest.raises(registry.ImplementationNotFoundError):
        registry.get_impl_by_name("test.op", "auto")
    with pytest.raises(registry.UnsupportedGenerationError) as caught:
        registry.get_impl_by_name("test.op", "triton")
    assert caught.value.supported_generations == {generation.Generation.A3}
    backend.assert_not_called()


# Verify a shared callable may declare different generation support for different operators.
def test_alias_rule_is_scoped_to_operator(registry_for_generation):
    modules = registry_for_generation("A2")
    registry, generation = modules.registry, modules.generation
    shared_backend = Mock()
    registry.register_impl("test.first", "backend")(shared_backend)
    registry.register_impl(
        "test.second", "backend", supported_generations={generation.Generation.A3}
    )(shared_backend)
    assert registry.get_impl_by_name("test.first", "backend") is shared_backend
    with pytest.raises(registry.UnsupportedGenerationError):
        registry.get_impl_by_name("test.second", "backend")


# Verify later changes to the input support list do not alter the captured declaration.
def test_support_declaration_is_a_snapshot(registry_for_generation):
    modules = registry_for_generation("A3")
    registry, generation = modules.registry, modules.generation
    choices = [generation.Generation.A3]
    backend = Mock()
    decorator = registry.register_impl(
        "test.op", "backend", supported_generations=choices
    )
    choices.clear()
    decorator(backend)
    assert registry.get_impl_by_name("test.op", "backend") is backend
