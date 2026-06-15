"""
Shared direction-extraction helper.

`get_direction_pair` extracts both the harmfulness direction (`t_inst`) and
the refusal direction (`t_post_inst`) from the English PolyRefuse train
split in a single sweep, caches them to disk, and returns them. Both
pilot experiments and the validation attacks use this helper, so it lives
in its own module rather than under a specific experiment script.

Directions follow Arditi et al. (2024) and Zhao et al. (2025):
    d_harm    = mean_pos(act_harmful @ t_inst)      − mean_pos(act_harmless @ t_inst)
    d_refuse  = mean_pos(act_harmful @ t_post_inst) − mean_pos(act_harmless @ t_post_inst)
"""

from __future__ import annotations

import os
from typing import List

from bluedot.directions import (
    extract_t_inst_and_t_post_inst,
    load_direction,
    save_direction,
)

from . import _shared as S


# Direction extraction is anchored on English PolyRefuse for cross-lingual
# transfer, following Wang et al. (2025).
LANG = "en"


def get_direction_pair(
    suffix: str,
    harmful: List[str],
    harmless: List[str],
):
    """Extract both `t_inst` and `t_post_inst` in one sweep; cache per suffix.

    `suffix=""`   — raw harmful / harmless prompts (the default in every
                    public experiment).
    `suffix="*"`  — caller has wrapped the prompts in some prefix / suffix
                    template; tag the cache so different wrappings don't
                    collide.

    Returns
    -------
    (d_harm, d_refuse) : Tuple[Direction, Direction]
    """
    model = S.get_model()
    p_harm   = S.direction_path(LANG, "t_inst",      suffix=suffix)
    p_refuse = S.direction_path(LANG, "t_post_inst", suffix=suffix)

    if os.path.exists(p_harm + ".pt") and os.path.exists(p_refuse + ".pt"):
        return load_direction(p_harm), load_direction(p_refuse)

    dirs = extract_t_inst_and_t_post_inst(
        model, harmful, harmless,
        batch_size=S.CONFIG.batch_size, lang=LANG,
        source_pos_dataset=f"polyrefuse:harmful:{LANG}"  + (f":{suffix}" if suffix else ""),
        source_neg_dataset=f"polyrefuse:harmless:{LANG}" + (f":{suffix}" if suffix else ""),
    )
    save_direction(dirs["t_inst"],      p_harm)
    save_direction(dirs["t_post_inst"], p_refuse)
    return dirs["t_inst"], dirs["t_post_inst"]
