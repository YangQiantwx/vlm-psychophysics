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
