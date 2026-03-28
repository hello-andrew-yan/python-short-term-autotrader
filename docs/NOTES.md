<h3 align="center">Developer Notes</h3>

This document is a personal study log. It tracks decisions I've made, concepts learned, and implementation notes. It is not intended as project documentation.

#### 28/03/2026

- `*` in a function signature marks everything after it as keyword-only. Prevents positional ambiguity when a function accepts both a callable and optional params.
- A `callable` is any object Python can invoke with `()`: functions, classes, objects implementing `__call__`. `callable()` returns `True` or `False`.
- A decorator accepting no args, empty params `()`, or keyword args requires dual-mode dispatch via `callable(func)` check to determine which form was used.

    ```python
    return decorator(func) if callable(func) else decorator
    ```

    - Bare `@decorator` has Python auto-pass the function as the first argument.
    - `@decorator()` receives `None`, returns the decorator for Python to apply.
    - `@decorator(param=x)` same as above but closes over custom params.

- `stack` is index-aware, pivots column levels into row `MultiIndex`. Used for reshaping multi-ticker `yfinance` output.

    ```python
    #            AAPL          GOOG
    #            Open  Close   Open  Close
    # 2024-01-01  150    152    140    141

    df.stack(level=0)

    #                      Close   Open
    # 2024-01-01  AAPL      152    150
    #             GOOG      141    140
    ```

- `melt` is flat, collapses named columns into variable/value pairs. No `MultiIndex` awareness.

    ```python
    #    name  math  english
    #   Alice    90       85

    df.melt(id_vars="name", var_name="subject", value_name="score")

    #    name  subject  score
    #   Alice     math     90
    #   Alice  english     85
    #     Bob     math     78
    #     Bob  english     92
    ```