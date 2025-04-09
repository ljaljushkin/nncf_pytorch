torch_model=
tune_model=
rank=
export_to_openvino=

import nncf

nncf.compress_weights(
    torch_model,
    ...,
    compression_format=nncf.CompressionFormat.FQ_LORA,
    advanced_parameters=nncf.AdvancedCompressionParameters(
        lora_adapter_rank=rank
    )
)
tune_model(torch_model)
torch_model = nncf.strip(torch_model, strip_format=nncf.StripFormat.DQ)
ov_model = export_to_openvino(torch_model)
