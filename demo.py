torch_model=
tune_model=
rank=
export_to_openvino=

import nncf

nncf.compress_weights(
    torch_model,
    ...,
)
ov_model = export_to_openvino(torch_model)
