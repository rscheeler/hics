from __future__ import annotations

from typing import TYPE_CHECKING

import pint

if TYPE_CHECKING:
    from pint import UnitRegistry


def getreg() -> UnitRegistry:
    # 1. Create the registry with automatic OS-level caching
    # This can improve startup performance by 5x to 20x
    ureg = pint.UnitRegistry(cache_folder=":auto:")

    # 2. Set this instance as the global application registry
    pint.set_application_registry(ureg)

    # Now, subsequent calls anywhere in your app will use this cached registry
    app_ureg = pint.get_application_registry()
    
    # Settings
    app_ureg.autoconvert_offset_to_baseunit = True
    app_ureg.force_ndarray_like = True
    return app_ureg


ureg = getreg()
