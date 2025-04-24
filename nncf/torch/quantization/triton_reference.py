# AOT ID: ['0_inference']

import torch
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.runtime.triton_heuristics import grid

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p

triton_fp32 = async_compile.triton(
    "triton_fp32",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_fp32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '0E91B58DAB54C915AAF8467E3EDB6871F6D05685FF049BBEEDA70C789216121A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_fp32(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262668288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 128256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = tmp1 + tmp3
    tmp5 = triton_helpers.minimum(tmp2, tmp4)
    tmp6 = tmp5 - tmp1
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp3
    tmp9 = 255.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 * tmp10
    tmp12 = -tmp1
    tmp13 = tmp12 * tmp10
    tmp14 = libdevice.nearbyint(tmp13)
    tmp15 = tmp11 - tmp14
    tmp16 = libdevice.nearbyint(tmp15)
    tmp17 = tmp16 / tmp10
    tl.store(out_ptr0 + (x2), tmp17, None)
""",
    device_str="cuda",
)

triton_fp16 = async_compile.triton(
    "triton_fp16",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_fp16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '0E91B58DAB54C915AAF8467E3EDB6871F6D05685FF049BBEEDA70C789216121A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_fp16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262668288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 128256
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp6 = tmp2 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = triton_helpers.minimum(tmp4, tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 - tmp2
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp5
    tmp13 = 255.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp10 * tmp14
    tmp16 = -tmp2
    tmp17 = tmp16 * tmp14
    tmp18 = libdevice.nearbyint(tmp17)
    tmp19 = tmp15 - tmp18
    tmp20 = libdevice.nearbyint(tmp19)
    tmp21 = tmp20 / tmp14
    tl.store(out_ptr0 + (x2), tmp21, None)
""",
    device_str="cuda",
)

triton_bf16 = async_compile.triton(
    "triton_bf16",
    """
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_bf16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '0E91B58DAB54C915AAF8467E3EDB6871F6D05685FF049BBEEDA70C789216121A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_bf16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262668288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 128256
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp6 = tmp2 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = triton_helpers.minimum(tmp4, tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 - tmp2
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp5
    tmp13 = 255.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp10 * tmp14
    tmp16 = -tmp2
    tmp17 = tmp16 * tmp14
    tmp18 = libdevice.nearbyint(tmp17)
    tmp19 = tmp15 - tmp18
    tmp20 = libdevice.nearbyint(tmp19)
    tmp21 = tmp20 / tmp14
    tl.store(out_ptr0 + (x2), tmp21, None)
""",
    device_str="cuda",
)

async_compile.wait(globals())
del async_compile


def triton_fwd(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2048, 1), (1, 1))
    assert_size_stride(arg1_1, (2048, 128256), (128256, 1))
    assert_size_stride(arg2_1, (2048, 1), (1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        # Topologically Sorted Source Nodes: [output, add, output_1, scale, output_2, neg, mul, zero_point, output_3, output_4, output_5], Original ATen: [aten.clamp, aten.add, aten.sub, aten.reciprocal, aten.mul, aten.neg, aten.round, aten.div]
        if arg1_1.dtype == torch.float32:
            buf0 = empty_strided_cuda((2048, 128256), (128256, 1), torch.float32)
            triton_fp32.run(arg1_1, arg2_1, arg0_1, buf0, 262668288, grid=grid(262668288), stream=stream0)
        elif arg1_1.dtype == torch.float16:
            buf0 = empty_strided_cuda((2048, 128256), (128256, 1), torch.float16)
            triton_fp16.run(arg1_1, arg2_1, arg0_1, buf0, 262668288, grid=grid(262668288), stream=stream0)
        elif arg1_1.dtype == torch.bfloat16:
            buf0 = empty_strided_cuda((2048, 128256), (128256, 1), torch.bfloat16)
            triton_bf16.run(arg1_1, arg2_1, arg0_1, buf0, 262668288, grid=grid(262668288), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
    return (buf0,)


def triton_forward(input_, input_low, input_range, levels):
    return triton_fwd([input_low, input_, input_range])
