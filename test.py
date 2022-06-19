from typing import Any

sample = "F67A33D65A13F67A33D65A13C629D3C629D4066A14369A44369A44167A24268A34268A34268A34066A14167A24066A13F65A04066A"
n=6
list: Any = [sample[i:i+n] for i in range(0, len(sample), n)]
print(list[0])