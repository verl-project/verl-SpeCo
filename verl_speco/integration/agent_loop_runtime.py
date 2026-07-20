"""Runtime patches for upstream verl agent-loop integration."""

from __future__ import annotations

import contextvars
import inspect
import logging
from functools import wraps
from typing import Any

logger = logging.getLogger(__file__)

_PATCHED = False
_DEFAULT_EXTRA_KEYS = {
    "turn_scores",
    "tool_rewards",
    "min_global_steps",
    "max_global_steps",
    "extras",
    "drafter_sample",
}
_CURRENT_GLOBAL_STEPS = contextvars.ContextVar(
    "speco_current_global_steps", default=None
)
_CURRENT_VALIDATE = contextvars.ContextVar("speco_current_validate", default=False)
SPECO_AGENT_LOOP_MANAGER_CLASS = (
    "verl_speco.integration.agent_loop_runtime.SpecoAgentLoopManager"
)
_SPECO_WORKER_WRAPPER_METHOD_NAMES = {
    "_speco_worker_init",
    "_speco_worker_generate_sequences",
    "_speco_worker_run_agent_loop",
    "_speco_worker_agent_loop_postprocess",
    "_speco_worker_postprocess",
}


def _is_sglang_rollout_config(config: Any) -> bool:
    rollout_config = getattr(config, "actor_rollout_ref", None)
    rollout_config = getattr(rollout_config, "rollout", None)
    if rollout_config is None and hasattr(config, "get"):
        actor_rollout_ref = config.get("actor_rollout_ref", {})
        rollout_config = (
            actor_rollout_ref.get("rollout")
            if hasattr(actor_rollout_ref, "get")
            else None
        )
    if rollout_config is None:
        return False
    if hasattr(rollout_config, "get"):
        return rollout_config.get("name") == "sglang"
    return getattr(rollout_config, "name", None) == "sglang"


def _ensure_extra_field_defaults(result: Any) -> Any:
    non_tensor_batch = getattr(result, "non_tensor_batch", None)
    if not isinstance(non_tensor_batch, dict):
        return result
    batch_len = None
    for value in non_tensor_batch.values():
        if isinstance(value, dict):
            continue
        try:
            batch_len = len(value)
            break
        except TypeError:
            continue
    if batch_len is None:
        return result

    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return result

    for key in _DEFAULT_EXTRA_KEYS:
        if key not in non_tensor_batch:
            values = np.empty(batch_len, dtype=object)
            values[:] = [None for _ in range(batch_len)]
            non_tensor_batch[key] = values
    return result


def _sampling_params_with_speco_context(sampling_params: Any) -> Any:
    if not isinstance(sampling_params, dict):
        return sampling_params
    patched = dict(sampling_params)
    global_steps = _CURRENT_GLOBAL_STEPS.get()
    if global_steps is not None:
        patched.setdefault("_verl_global_steps", global_steps)
    if bool(_CURRENT_VALIDATE.get()):
        patched["_verl_skip_drafter_collection"] = True
    return patched


def _sampling_params_with_speco_step(
    sampling_params: Any, *, global_steps: Any, validate: bool
) -> Any:
    if not isinstance(sampling_params, dict):
        return sampling_params
    patched = dict(sampling_params)
    if global_steps is not None and global_steps != -1:
        patched["_verl_global_steps"] = global_steps
    if validate:
        patched["_verl_skip_drafter_collection"] = True
    return patched


def _speco_parent_method(instance: Any, method_name: str) -> Any:
    for cls in type(instance).__mro__[1:]:
        if getattr(cls, "_speco_explicit_worker_runtime", False):
            continue
        method = cls.__dict__.get(method_name)
        if (
            method is not None
            and getattr(method, "__name__", None)
            not in _SPECO_WORKER_WRAPPER_METHOD_NAMES
        ):
            return method
    return None


def _speco_default_agent_loop_extra_fields(output: Any) -> Any:
    extra_fields = getattr(output, "extra_fields", None)
    if isinstance(extra_fields, dict):
        for key in _DEFAULT_EXTRA_KEYS:
            extra_fields.setdefault(key, None)
    return output


def _speco_default_agent_loop_inputs(inputs: Any) -> Any:
    for input_item in inputs:
        _speco_default_agent_loop_extra_fields(input_item)
    return inputs


def _speco_context_from_batch(batch: Any) -> tuple[Any, Any]:
    meta_info = getattr(batch, "meta_info", None)
    meta_info = meta_info if isinstance(meta_info, dict) else {}
    global_steps_token = _CURRENT_GLOBAL_STEPS.set(meta_info.get("global_steps"))
    validate_token = _CURRENT_VALIDATE.set(bool(meta_info.get("validate", False)))
    return global_steps_token, validate_token


def _speco_reset_context(global_steps_token: Any, validate_token: Any) -> None:
    _CURRENT_VALIDATE.reset(validate_token)
    _CURRENT_GLOBAL_STEPS.reset(global_steps_token)


def _speco_worker_init(self, *args, **kwargs):
    install_agent_loop_runtime_patch()
    init = _speco_parent_method(self, "__init__")
    if callable(init):
        init(self, *args, **kwargs)


async def _speco_worker_generate_sequences(self, batch):
    global_steps_token, validate_token = _speco_context_from_batch(batch)
    try:
        generate_sequences = _speco_parent_method(self, "generate_sequences")
        if not callable(generate_sequences):
            raise AttributeError(
                "SPECO AgentLoop worker parent has no generate_sequences method"
            )
        result = generate_sequences(self, batch)
        if inspect.isawaitable(result):
            result = await result
        return _ensure_extra_field_defaults(result)
    finally:
        _speco_reset_context(global_steps_token, validate_token)


async def _speco_worker_run_agent_loop(
    self, sampling_params, trajectory, *args, **kwargs
):
    trajectory = trajectory if isinstance(trajectory, dict) else {}
    sampling_params = _sampling_params_with_speco_step(
        sampling_params,
        global_steps=trajectory.get("step"),
        validate=bool(trajectory.get("validate", False)),
    )
    run_agent_loop = _speco_parent_method(self, "_run_agent_loop")
    if not callable(run_agent_loop):
        raise AttributeError(
            "SPECO AgentLoop worker parent has no _run_agent_loop method"
        )
    result = run_agent_loop(self, sampling_params, trajectory, *args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


async def _speco_worker_agent_loop_postprocess(self, output, validate, **kwargs):
    _speco_default_agent_loop_extra_fields(output)
    agent_loop_postprocess = _speco_parent_method(self, "_agent_loop_postprocess")
    if not callable(agent_loop_postprocess):
        raise AttributeError(
            "SPECO AgentLoop worker parent has no _agent_loop_postprocess method"
        )
    result = agent_loop_postprocess(self, output, validate, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return _speco_default_agent_loop_extra_fields(result)


def _speco_worker_postprocess(
    self, inputs, input_non_tensor_batch=None, validate=False
):
    _speco_default_agent_loop_inputs(inputs)
    postprocess = _speco_parent_method(self, "_postprocess")
    if not callable(postprocess):
        raise AttributeError("SPECO AgentLoop worker parent has no _postprocess method")
    signature = inspect.signature(postprocess)
    kwargs = {}
    if "input_non_tensor_batch" in signature.parameters:
        kwargs["input_non_tensor_batch"] = input_non_tensor_batch
    if "validate" in signature.parameters:
        kwargs["validate"] = validate
    result = postprocess(self, inputs, **kwargs)
    return _ensure_extra_field_defaults(result)


def _load_agent_loop_module():
    try:
        from verl.experimental.agent_loop import agent_loop as agent_loop_module

        return agent_loop_module
    except Exception:  # noqa: BLE001
        pass

    try:
        import verl.experimental.agent_loop as agent_loop_module

        return agent_loop_module
    except Exception:
        raise


try:
    _AgentLoopModule = _load_agent_loop_module()
    _UpstreamAgentLoopManager = getattr(_AgentLoopModule, "AgentLoopManager", object)
except Exception:  # noqa: BLE001
    _AgentLoopModule = None
    _UpstreamAgentLoopManager = object


def _build_speco_agent_loop_worker_class(worker_cls):
    speco_worker_cls = getattr(worker_cls, "_speco_remote_worker_cls", None)
    if speco_worker_cls is not None and getattr(
        speco_worker_cls, "_speco_explicit_worker_runtime", False
    ):
        return speco_worker_cls

    generate_sequences = getattr(worker_cls, "generate_sequences", None)
    run_agent_loop = getattr(worker_cls, "_run_agent_loop", None)
    agent_loop_postprocess = getattr(worker_cls, "_agent_loop_postprocess", None)
    postprocess = getattr(worker_cls, "_postprocess", None)

    attrs = {
        "__module__": __name__,
        "__doc__": "Ray-serializable AgentLoopWorker subclass carrying SPECO runtime patches.",
        "__init__": _speco_worker_init,
        "_speco_explicit_worker_runtime": True,
    }
    if callable(generate_sequences):
        attrs["generate_sequences"] = _speco_worker_generate_sequences
    if callable(run_agent_loop):
        attrs["_run_agent_loop"] = _speco_worker_run_agent_loop
    if callable(agent_loop_postprocess):
        attrs["_agent_loop_postprocess"] = _speco_worker_agent_loop_postprocess
    if callable(postprocess):
        attrs["_postprocess"] = _speco_worker_postprocess

    speco_worker_cls = type("SpecoAgentLoopWorker", (worker_cls,), attrs)
    speco_worker_cls.__qualname__ = "SpecoAgentLoopWorker"
    worker_cls._speco_remote_worker_cls = speco_worker_cls
    return speco_worker_cls


def _configure_speco_agent_loop_manager_instance(manager: Any, worker_cls: Any) -> None:
    if not _is_sglang_rollout_config(getattr(manager, "config", None)):
        return
    if getattr(manager, "_speco_agent_loop_manager_configured", False):
        return

    import ray

    manager.agent_loop_workers_class = ray.remote(
        _build_speco_agent_loop_worker_class(worker_cls)
    )

    try:
        from verl.workers.rollout.sglang_rollout import async_sglang_server
        from verl_speco.integration.sglang_runtime import (
            _build_speco_replica_class,
            patch_sglang_server_adapter_update,
        )

        manager.rollout_replica_class = _build_speco_replica_class(async_sglang_server)
        patch_sglang_server_adapter_update()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Unable to bind SPECO SGLang replica bridge to AgentLoopManager: %s", exc
        )

    logger.warning(
        "SPECO AgentLoopManager bridge active for SGLang rollout: worker_class=%s rollout_replica_class=%s",
        getattr(
            _build_speco_agent_loop_worker_class(worker_cls),
            "__name__",
            type(worker_cls).__name__,
        ),
        getattr(getattr(manager, "rollout_replica_class", None), "__name__", None),
    )
    manager._speco_agent_loop_manager_configured = True


class SpecoAgentLoopManager(_UpstreamAgentLoopManager):
    """Explicit AgentLoopManager bridge used by upstream config loading."""

    def __init__(self, *args, **kwargs):
        install_agent_loop_runtime_patch()
        super().__init__(*args, **kwargs)
        agent_loop_module = _AgentLoopModule or _load_agent_loop_module()
        worker_cls = getattr(agent_loop_module, "AgentLoopWorker")
        _configure_speco_agent_loop_manager_instance(self, worker_cls)


def install_agent_loop_runtime_patch() -> bool:
    """Patch external verl agent-loop workers for SPECO side-channel fields."""

    global _PATCHED
    if _PATCHED:
        return True

    try:
        agent_loop_module = _load_agent_loop_module()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to install SPECO agent-loop runtime patch: %s", exc)
        return False

    worker_cls = getattr(agent_loop_module, "AgentLoopWorker", None)
    manager_cls = getattr(agent_loop_module, "AgentLoopManager", None)
    llm_server_manager_cls = getattr(agent_loop_module, "AsyncLLMServerManager", None)
    if worker_cls is None or manager_cls is None or llm_server_manager_cls is None:
        logger.warning(
            "Unable to install SPECO agent-loop runtime patch: missing upstream classes"
        )
        return False

    generate_sequences = getattr(worker_cls, "generate_sequences", None)
    run_agent_loop = getattr(worker_cls, "_run_agent_loop", None)
    agent_loop_postprocess = getattr(worker_cls, "_agent_loop_postprocess", None)
    postprocess = getattr(worker_cls, "_postprocess", None)
    manager_init = getattr(manager_cls, "__init__", None)
    manager_generate_sequences = getattr(manager_cls, "generate_sequences", None)
    llm_server_generate = getattr(llm_server_manager_cls, "generate", None)
    if (
        not callable(generate_sequences)
        or not callable(run_agent_loop)
        or not callable(postprocess)
        or not callable(manager_init)
        or not callable(manager_generate_sequences)
        or not callable(llm_server_generate)
    ):
        logger.warning(
            "Unable to install SPECO agent-loop runtime patch: missing upstream methods"
        )
        return False

    if not getattr(llm_server_manager_cls, "_speco_patched_generate", False):

        @wraps(llm_server_generate)
        async def speco_llm_server_generate(self, *args, **kwargs):
            if "sampling_params" in kwargs:
                kwargs["sampling_params"] = _sampling_params_with_speco_context(
                    kwargs["sampling_params"]
                )
            result = llm_server_generate(self, *args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result

        llm_server_manager_cls.generate = speco_llm_server_generate
        llm_server_manager_cls._speco_patched_generate = True

    if not getattr(worker_cls, "_speco_patched_generate_sequences", False):

        @wraps(generate_sequences)
        async def speco_generate_sequences(self, batch):
            meta_info = getattr(batch, "meta_info", None)
            meta_info = meta_info if isinstance(meta_info, dict) else {}
            global_steps_token = _CURRENT_GLOBAL_STEPS.set(
                meta_info.get("global_steps")
            )
            validate_token = _CURRENT_VALIDATE.set(
                bool(meta_info.get("validate", False))
            )
            try:
                result = generate_sequences(self, batch)
                if inspect.isawaitable(result):
                    result = await result
                return _ensure_extra_field_defaults(result)
            finally:
                _CURRENT_VALIDATE.reset(validate_token)
                _CURRENT_GLOBAL_STEPS.reset(global_steps_token)

        worker_cls.generate_sequences = speco_generate_sequences
        worker_cls._speco_patched_generate_sequences = True

    if not getattr(worker_cls, "_speco_patched_run_agent_loop", False):

        @wraps(run_agent_loop)
        async def speco_run_agent_loop(
            self, sampling_params, trajectory, *args, **kwargs
        ):
            trajectory = trajectory if isinstance(trajectory, dict) else {}
            sampling_params = _sampling_params_with_speco_step(
                sampling_params,
                global_steps=trajectory.get("step"),
                validate=bool(trajectory.get("validate", False)),
            )
            result = run_agent_loop(self, sampling_params, trajectory, *args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result

        worker_cls._run_agent_loop = speco_run_agent_loop
        worker_cls._speco_patched_run_agent_loop = True

    if callable(agent_loop_postprocess) and not getattr(
        worker_cls, "_speco_patched_agent_loop_postprocess", False
    ):

        @wraps(agent_loop_postprocess)
        async def speco_agent_loop_postprocess(self, output, validate, **kwargs):
            extra_fields = getattr(output, "extra_fields", None)
            if isinstance(extra_fields, dict):
                for key in _DEFAULT_EXTRA_KEYS:
                    extra_fields.setdefault(key, None)
            result = agent_loop_postprocess(self, output, validate, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            extra_fields = getattr(result, "extra_fields", None)
            if isinstance(extra_fields, dict):
                for key in _DEFAULT_EXTRA_KEYS:
                    extra_fields.setdefault(key, None)
            return result

        worker_cls._agent_loop_postprocess = speco_agent_loop_postprocess
        worker_cls._speco_patched_agent_loop_postprocess = True

    if not getattr(worker_cls, "_speco_patched_postprocess", False):

        @wraps(postprocess)
        def speco_postprocess(
            self, inputs, input_non_tensor_batch=None, validate=False
        ):
            for input_item in inputs:
                extra_fields = getattr(input_item, "extra_fields", None)
                if isinstance(extra_fields, dict):
                    for key in _DEFAULT_EXTRA_KEYS:
                        extra_fields.setdefault(key, None)
            signature = inspect.signature(postprocess)
            kwargs = {}
            if "input_non_tensor_batch" in signature.parameters:
                kwargs["input_non_tensor_batch"] = input_non_tensor_batch
            if "validate" in signature.parameters:
                kwargs["validate"] = validate
            result = postprocess(self, inputs, **kwargs)
            return _ensure_extra_field_defaults(result)

        worker_cls._postprocess = speco_postprocess
        worker_cls._speco_patched_postprocess = True

    if not getattr(manager_cls, "_speco_patched_init", False):

        @wraps(manager_init)
        def speco_manager_init(self, *args, **kwargs):
            manager_init(self, *args, **kwargs)
            _configure_speco_agent_loop_manager_instance(self, worker_cls)

        manager_cls.__init__ = speco_manager_init
        manager_cls._speco_patched_init = True

    if not getattr(manager_cls, "_speco_patched_generate_sequences", False):
        try:
            from verl.utils.ray_utils import auto_await
        except Exception:  # noqa: BLE001
            auto_await = None

        @wraps(manager_generate_sequences)
        async def speco_manager_generate_sequences(self, prompts):
            if _is_sglang_rollout_config(getattr(self, "config", None)):
                meta_info = getattr(prompts, "meta_info", None)
                if isinstance(meta_info, dict) and "global_steps" in meta_info:
                    global_steps = (
                        None
                        if meta_info.get("validate", False)
                        else meta_info.get("global_steps")
                    )
                    server_handles = getattr(self, "server_handles", None)
                    if server_handles:
                        import asyncio

                        set_step_refs = []
                        for handle in server_handles:
                            set_global_steps = getattr(handle, "set_global_steps", None)
                            remote = getattr(set_global_steps, "remote", None)
                            if callable(remote):
                                set_step_refs.append(remote(global_steps))
                        if set_step_refs:
                            await asyncio.gather(*set_step_refs)
            result = manager_generate_sequences(self, prompts)
            if hasattr(result, "__await__"):
                result = await result
            return _ensure_extra_field_defaults(result)

        manager_cls.generate_sequences = (
            auto_await(speco_manager_generate_sequences)
            if callable(auto_await)
            else speco_manager_generate_sequences
        )
        manager_cls._speco_patched_generate_sequences = True

    _PATCHED = True
    logger.warning("Installed SPECO agent-loop runtime patch.")
    return True
