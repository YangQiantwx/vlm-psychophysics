"""Enable the validation compatibility layer only for an explicit process opt-in."""
import os

if os.environ.get("CLASSIC_PRECISION_KV_COMPAT") == "1":
    try:
        from classic_precision_vllm_kv_compat import install
        install()
    except BaseException as error:
        # Python ignores ordinary sitecustomize exceptions. This process must stop.
        try:
            os.write(2, ("Classic precision KV compatibility refused: " + str(error) + "\n").encode())
        finally:
            os._exit(78)

if "CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB" in os.environ:
    try:
        from classic_precision_vllm_resource_compat import install as install_resources
        install_resources()
    except BaseException as error:
        try:
            os.write(2, ("Classic precision resource override refused: " + str(error) + "\n").encode())
        finally:
            os._exit(78)

if "CLASSIC_PRECISION_DEFER_MODELS" in os.environ:
    try:
        from classic_precision_queue_compat import install as install_queue
        install_queue()
    except BaseException as error:
        try:
            os.write(2, ("Classic precision queue override refused: " + str(error) + "\n").encode())
        finally:
            os._exit(78)
